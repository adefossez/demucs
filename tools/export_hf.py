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
(see `_flatten_state` / `_unflatten_state`).

Example:
    uv run tools/export_hf.py --models htdemucs htdemucs_ft --out release_hf --check
"""
import argparse
from fractions import Fraction
import importlib
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


def _unflatten_state(tensors, structure):
    """Inverse of `_flatten_state`."""
    def unflatten(node):
        if isinstance(node, dict):
            if "_tensor" in node:
                return tensors[node["_tensor"]]
            elif "_dict" in node:
                return {key: unflatten(item) for key, item in node["_dict"]}
            elif "_list" in node:
                return [unflatten(item) for item in node["_list"]]
            elif "_tuple" in node:
                return tuple(unflatten(item) for item in node["_tuple"])
            elif "_class" in node:
                module, name = node["_class"].rsplit(".", 1)
                return getattr(importlib.import_module(module), name)
            else:
                raise ValueError(f"Invalid structure node {node}.")
        return node
    return unflatten(structure)


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
    from safetensors import safe_open
    from demucs.states import load_model

    with safe_open(out / f"{sig}.safetensors", framework="pt") as file:
        metadata = file.metadata()
        tensors = {key: file.get_tensor(key) for key in file.keys()}
    if 'structure' in metadata:
        state = _unflatten_state(tensors, json.loads(metadata['structure']))
    else:
        state = tensors
    module, name = metadata['klass'].rsplit(".", 1)
    klass = getattr(importlib.import_module(module), name)
    kwargs = json.loads(metadata['kwargs'])
    if isinstance(kwargs.get('segment'), dict):  # fraction
        kwargs['segment'] = Fraction(kwargs['segment']['numerator'],
                                     kwargs['segment']['denominator'])
    model = load_model({'klass': klass, 'args': json.loads(metadata['args']),
                        'kwargs': kwargs, 'state': state})

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
    parser.add_argument('--check', action='store_true',
                        help='Reload each converted model and check it is identical '
                             'to the one from the original checkpoint.')
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
