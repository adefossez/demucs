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

Quantized checkpoints (diffq, e.g. `mdx_q`) hold a nested structure of bit-packed int64
tensors and scales rather than a plain state dict: the tensors are stored as-is in the
safetensors file, and the nesting is recorded as json under the `structure` metadata key
(see `_flatten_state` here, and its inverse `demucs.hf._unflatten_state`).

Example:
    uv run tools/export_hf.py --models htdemucs htdemucs_ft --out release_hf --check
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import shutil
import sys

import torch
import yaml

from demucs.hf import hf_repo_name
from demucs.pretrained import REMOTE_ROOT, ROOT_URL, _parse_remote_files  # noqa


def _json_default(value):
    if isinstance(value, Fraction):
        return {"_type": "fraction", "numerator": value.numerator,
                "denominator": value.denominator}
    return str(value)


def _flatten_state(state):
    """Flatten an arbitrarily nested model state (e.g. diffq packed states) into a flat
    `{key: tensor}` dict suitable for safetensors, plus a json-able description of the
    nesting, with tensors referred to by their key. Dicts are stored as lists of pairs
    (safetensors metadata is json, whose object keys are always strings). The rare class
    leaves (e.g. the quantizer class in diffq metadata) are stored as import paths."""
    tensors = {}

    def flatten(value, path):
        if isinstance(value, torch.Tensor):
            tensors[path] = value.detach().clone().contiguous()
            return {"_tensor": path}
        elif isinstance(value, dict):
            return {"_dict": [[key, flatten(item, f"{path}.{key}")]
                              for key, item in value.items()]}
        elif isinstance(value, (list, tuple)):
            kind = "_" + type(value).__name__
            return {kind: [flatten(item, f"{path}.{index}")
                           for index, item in enumerate(value)]}
        elif isinstance(value, type):
            return {"_class": f"{value.__module__}.{value.__qualname__}"}
        elif value is None or isinstance(value, (bool, int, float, str)):
            return value
        else:
            raise TypeError(f"Cannot serialize {path} of type {type(value)}.")

    structure = flatten(state, "state")
    return tensors, structure


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
    state = pkg['state']
    if state.get('__quantized'):
        tensors, structure = _flatten_state(state)
        metadata['structure'] = json.dumps(structure)
    else:
        tensors = {key: value.contiguous() for key, value in state.items()}

    sidecar = dict(metadata)
    for key in ['training_args', 'metrics']:
        if key in pkg:
            sidecar[key] = json.loads(json.dumps(pkg[key], default=_json_default))
    with open(out / f"{sig}.json", "w") as file:
        json.dump(sidecar, file, indent=2)

    weights_name = f"{sig}.safetensors"
    save_file(tensors, out / weights_name, metadata=metadata)
    return weights_name


def check_conversion(checkpoint: Path, sig: str, out: Path):
    """Reload the converted file and check it restores the exact same model
    as the original torch checkpoint."""
    from demucs.hf import load_safetensors_model
    from demucs.states import load_model

    model = load_safetensors_model(out / f"{sig}.safetensors")
    reference = load_model(torch.load(checkpoint, 'cpu', weights_only=False))
    ref_state = reference.state_dict()
    new_state = model.state_dict()
    assert set(ref_state) == set(new_state), f"{sig}: state dict keys differ"
    for key in ref_state:
        assert torch.equal(ref_state[key], new_state[key]), f"{sig}: {key} differs"
    print(f"  Checked {sig}: restored model is identical.")


MODEL_CARD = """---
license: mit
tags:
- audio
- music
- music-source-separation
- demucs
---

# {repo_name}

Weights for the `{name}` pretrained model of
[Demucs](https://github.com/adefossez/demucs), music source separation in the
waveform domain. See the [demucs repository](https://github.com/adefossez/demucs)
for how to use them.

This is a bag of {count} model(s), whose outputs are averaged (see `{name}.yaml`
for the per source weights). Each `.safetensors` file contains the weights of one
model, with its class and init arguments as json in the safetensors metadata, and
the full training metadata in the matching `.json` sidecar.

{table}
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
    parser.add_argument('--check', action='store_true',
                        help='Reload each converted model and check it is identical '
                             'to the one from the original checkpoint.')
    parser.add_argument('--upload', action='store_true',
                        help='Upload each prepared folder to HuggingFace.')
    parser.add_argument('--namespace', default='adefossez',
                        help='HuggingFace namespace to upload to (default: adefossez).')
    parser.add_argument('--private', action='store_true',
                        help='Create the HuggingFace repositories as private.')
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
            if args.check:
                check_conversion(checkpoint, sig, repo)
            rows.append(f"- `{sig}` (`{weights_name}`)")
        repo_name = hf_repo_name(name)
        (repo / 'README.md').write_text(MODEL_CARD.format(
            repo_name=repo_name, name=name, count=len(bag['models']),
            table="\n".join(rows)))
        print(f"  Wrote {repo}")
        if args.upload:
            from huggingface_hub import HfApi
            api = HfApi()
            repo_id = f"{args.namespace}/{repo_name}"
            api.create_repo(repo_id, repo_type='model', private=args.private,
                            exist_ok=True)
            api.upload_folder(folder_path=repo, repo_id=repo_id, repo_type='model')
            print(f"  Uploaded to https://huggingface.co/{repo_id}")

    if args.models is None:
        leftover = sorted(set(urls) - used_sigs)
        if leftover:
            print(f"Note: {len(leftover)} remote checkpoints are not referenced by any "
                  f"bag yaml and were not exported: {', '.join(leftover)}")
    if args.upload:
        print(f"Done. Repositories are in {args.out} and uploaded.")
    else:
        print(f"Done. Repositories are in {args.out}, rerun with --upload "
              "to push them to HuggingFace.")


if __name__ == '__main__':
    main()
