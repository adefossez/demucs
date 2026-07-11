# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Importable module names of the dependencies required by `get_model`,
# as checked by `torch.hub` with `importlib.util.find_spec`.
dependencies = ['einops', 'julius', 'torch', 'tqdm', 'yaml']

from demucs.pretrained import get_model
