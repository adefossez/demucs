# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Download all the released pretrained models and prepare one folder per named model
(i.e. per bag of models), ready to be uploaded as a HuggingFace model repository.

For each named model (e.g. `htdemucs_ft`), the output folder contains:
- the bag definition yaml (list of signatures, weights, segment),
- one `{sig}.safetensors` file per model in the bag, with the weights, and the
  model class / init arguments stored as json in the safetensors metadata,
- one `{sig}.json` sidecar with the full metadata (including training args and metrics),
- a README.md model card stub.

Quantized checkpoints (diffq, e.g. `mdx_q`) contain packed byte buffers that cannot be
represented as safetensors: those are copied as the original `.th` torch checkpoint.

Example:
    uv run tools/export_hf.py --models htdemucs htdemucs_ft --out release_hf
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import shutil
import sys

import torch
import yaml

from demucs.pretrained import REMOTE_ROOT, ROOT_URL, _parse_remote_files  # noqa


def _json_default(value):
    if isinstance(value, Fraction):
        return {"_type": "fraction", "numerator": value.numerator,
                "denominator": value.denominator}
    return str(value)


def download_checkpoint(url: str, cache: Path) -> Path:
    name = url.rsplit('/', 1)[1]
    target = cache / name
    if not target.exists():
        print(f"  Downloading {url}")
        torch.hub.download_url_to_file(url, str(target), progress=True)
    checksum = target.stem.rsplit('-', 1)[1]
    from demucs.repo import check_checksum
    check_checksum(target, checksum)
    return target


def convert_checkpoint(checkpoint: Path, sig: str, out: Path) -> str:
    """Convert a demucs checkpoint to safetensors + json sidecar in `out`.
    Returns the filename holding the weights."""
    from safetensors.torch import save_file
    pkg = torch.load(checkpoint, 'cpu', weights_only=False)
    klass = pkg['klass']
    metadata = {
        'klass': f"{klass.__module__}.{klass.__qualname__}",
        'args': json.dumps(pkg['args'], default=_json_default),
        'kwargs': json.dumps(pkg['kwargs'], default=_json_default),
    }
    sidecar = dict(metadata)
    for key in ['training_args', 'metrics']:
        if key in pkg:
            sidecar[key] = json.loads(json.dumps(pkg[key], default=_json_default))
    with open(out / f"{sig}.json", "w") as file:
        json.dump(sidecar, file, indent=2)

    state = pkg['state']
    if state.get('__quantized'):
        # diffq packed states are not plain tensors, keep the torch checkpoint.
        weights_name = f"{sig}.th"
        shutil.copyfile(checkpoint, out / weights_name)
    else:
        weights_name = f"{sig}.safetensors"
        state = {key: value.contiguous() for key, value in state.items()}
        save_file(state, out / weights_name, metadata=metadata)
    return weights_name


MODEL_CARD = """---
license: mit
tags:
- audio
- music
- music-source-separation
- demucs
---

# {name}

`{name}` pretrained model from [Demucs](https://github.com/adefossez/demucs),
music source separation in the waveform domain.

This is a bag of {count} model(s), applied to the input mix and averaged:

{table}

Each `.safetensors` file contains the model weights, along with the model class and
its init arguments as json in the safetensors metadata (see the `.json` sidecars for
the full training metadata). The `{name}.yaml` file describes how the models are
combined (per source weights and evaluation segment length).
"""


def main():
    parser = argparse.ArgumentParser('export_hf', description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('release_hf'),
                        help='Where to create one folder per model repository.')
    parser.add_argument('--cache', type=Path, default=None,
                        help='Where to store the downloaded checkpoints. Defaults to '
                             'the torch hub cache, reusing existing downloads.')
    parser.add_argument('--models', nargs='*', default=None,
                        help='Only export the given named models (default: all).')
    args = parser.parse_args()

    cache = args.cache or Path(torch.hub.get_dir()) / 'checkpoints'
    cache.mkdir(exist_ok=True, parents=True)
    args.out.mkdir(exist_ok=True, parents=True)

    urls = _parse_remote_files(REMOTE_ROOT / 'files.txt')
    bags = {file.stem: file for file in sorted(REMOTE_ROOT.glob('*.yaml'))}
    names = args.models or sorted(bags)

    used_sigs = set()
    for name in names:
        if name not in bags:
            print(f"error: {name} is not a known model, choose from {sorted(bags)}.",
                  file=sys.stderr)
            sys.exit(1)

    for name in names:
        print(f"Preparing repository for {name}")
        with open(bags[name]) as file:
            bag = yaml.safe_load(file)
        repo = args.out / name
        repo.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(bags[name], repo / f"{name}.yaml")

        rows = []
        for sig in bag['models']:
            used_sigs.add(sig)
            checkpoint = download_checkpoint(urls[sig], cache)
            weights_name = convert_checkpoint(checkpoint, sig, repo)
            rows.append(f"- `{sig}` (`{weights_name}`)")
        (repo / 'README.md').write_text(MODEL_CARD.format(
            name=name, count=len(bag['models']), table="\n".join(rows)))
        print(f"  Wrote {repo}")

    if args.models is None:
        leftover = sorted(set(urls) - used_sigs)
        if leftover:
            print(f"Note: {len(leftover)} remote checkpoints are not referenced by any "
                  f"bag yaml and were not exported: {', '.join(leftover)}")
    print(f"Done. Repositories are in {args.out}, upload them with e.g.\n"
          f"    hf upload <namespace>/<name> {args.out}/<name>")


if __name__ == '__main__':
    main()
