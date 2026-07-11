# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Loading pretrained models from the HuggingFace hub.

Each named model (bag of models) lives in its own HuggingFace repository
(see `tools/export_hf.py` for how those are prepared), containing the bag
definition yaml and one safetensors file per model in the bag.
"""
from fractions import Fraction
import importlib
import json
from pathlib import Path
import typing as tp

import yaml

from .apply import BagOfModels, Model
from .states import load_model

DEFAULT_NAMESPACE = "adefossez"


def hf_repo_name(name: str) -> str:
    """Map a demucs model name to its HuggingFace repository name,
    e.g. `htdemucs_ft` -> `HTDemucs-ft`, `mdx_extra_q` -> `Demucs-mdx_extra_q`."""
    if name == 'htdemucs':
        return 'HTDemucs'
    elif name.startswith('htdemucs_'):
        return 'HTDemucs-' + name[len('htdemucs_'):]
    else:
        return 'Demucs-' + name


def _decode_json(value):
    """Decode the json encoding of the model init arguments, in particular
    fractions (e.g. the `segment` param of HTDemucs)."""
    if isinstance(value, dict):
        if value.get("_type") == "fraction":
            return Fraction(value["numerator"], value["denominator"])
        return {key: _decode_json(item) for key, item in value.items()}
    elif isinstance(value, list):
        return [_decode_json(item) for item in value]
    return value


def _unflatten_state(tensors: tp.Dict[str, tp.Any], structure):
    """Rebuild a nested model state (e.g. diffq packed states) from the flat
    safetensors tensor dict and the json structure stored in its metadata.
    Inverse of `_flatten_state` in `tools/export_hf.py`."""
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


def load_safetensors_model(path: tp.Union[str, Path]) -> Model:
    """Load a single model from a safetensors file, with the model class and init
    arguments stored as json in the safetensors metadata."""
    from safetensors import safe_open
    with safe_open(str(path), framework="pt") as file:
        metadata = file.metadata()
        tensors = {key: file.get_tensor(key) for key in file.keys()}
    state: tp.Any
    if 'structure' in metadata:
        state = _unflatten_state(tensors, json.loads(metadata['structure']))
    else:
        state = tensors
    module, name = metadata['klass'].rsplit(".", 1)
    klass = getattr(importlib.import_module(module), name)
    args = _decode_json(json.loads(metadata['args']))
    kwargs = _decode_json(json.loads(metadata['kwargs']))
    return load_model({'klass': klass, 'args': args, 'kwargs': kwargs, 'state': state})


def get_hf_model(name: str) -> BagOfModels:
    """Load a named model from the HuggingFace hub. `name` is a demucs model name
    (e.g. `htdemucs_ft`), potentially prefixed with a namespace to load from
    (e.g. `someuser/htdemucs_ft`), otherwise `adefossez` is used. Files are cached
    in the standard HuggingFace cache folder."""
    from huggingface_hub import hf_hub_download
    namespace = DEFAULT_NAMESPACE
    if '/' in name:
        namespace, name = name.split('/', 1)
    repo_id = f"{namespace}/{hf_repo_name(name)}"
    with open(hf_hub_download(repo_id, f"{name}.yaml")) as file:
        bag = yaml.safe_load(file)
    models = [load_safetensors_model(hf_hub_download(repo_id, f"{sig}.safetensors"))
              for sig in bag['models']]
    return BagOfModels(models, bag.get('weights'), bag.get('segment'))
