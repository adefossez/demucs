# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Loading pretrained models.
"""

import logging
from pathlib import Path
import typing as tp

from .hdemucs import HDemucs
from .repo import RemoteRepo, LocalRepo, ModelOnlyRepo, BagOnlyRepo, AnyModelRepo, ModelLoadingError  # noqa
from .states import _check_diffq
from .utils import fatal

logger = logging.getLogger(__name__)
ROOT_URL = "https://dl.fbaipublicfiles.com/demucs/"
REMOTE_ROOT = Path(__file__).parent / 'remote'

SOURCES = ["drums", "bass", "other", "vocals"]
DEFAULT_MODEL = 'htdemucs'


def demucs_unittest():
    model = HDemucs(channels=4, sources=SOURCES)
    return model


def add_model_flags(parser):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("-s", "--sig", help="Locally trained XP signature.")
    group.add_argument("-n", "--name", default="htdemucs",
                       help="Pretrained model name or signature. Default is htdemucs.")
    parser.add_argument("--repo", type=Path,
                        help="Folder containing all pre-trained models for use with -n.")


def _parse_remote_files(remote_file_list) -> tp.Dict[str, str]:
    root: str = ''
    models: tp.Dict[str, str] = {}
    for line in remote_file_list.read_text().split('\n'):
        line = line.strip()
        if line.startswith('#'):
            continue
        elif len(line) == 0:
            continue
        elif line.startswith('root:'):
            root = line.split(':', 1)[1].strip()
        else:
            sig = line.split('-', 1)[0]
            assert sig not in models
            models[sig] = ROOT_URL + root + line
    return models


def get_model(name: str,
              repo: tp.Optional[Path] = None):
    """`name` must be a bag of models name or a pretrained signature
    from the remote AWS model repo or the specified local repo if `repo` is not None.
    Bag of models names are first looked up on the HuggingFace hub, falling back to
    the legacy AWS repo (in particular for single signatures). Names of the form
    `hf://[namespace/]name` are always loaded from the HuggingFace hub.
    """
    if name.startswith('hf://'):
        from .hf import get_hf_model
        bag = get_hf_model(name[len('hf://'):])
        bag.eval()
        return bag
    if name == 'demucs_unittest':
        return demucs_unittest()
    if repo is None:
        from .hf import get_hf_model
        try:
            bag = get_hf_model(name)
        except Exception as exc:
            logger.debug('Could not load %s from the HuggingFace hub (%r), '
                         'falling back to the legacy remote repo.', name, exc)
        else:
            bag.eval()
            return bag
    model_repo: ModelOnlyRepo
    if repo is None:
        models = _parse_remote_files(REMOTE_ROOT / 'files.txt')
        model_repo = RemoteRepo(models)
        bag_repo = BagOnlyRepo(REMOTE_ROOT, model_repo)
    else:
        if not repo.is_dir():
            fatal(f"{repo} must exist and be a directory.")
        model_repo = LocalRepo(repo)
        bag_repo = BagOnlyRepo(repo, model_repo)
    any_repo = AnyModelRepo(model_repo, bag_repo)
    try:
        model = any_repo.get_model(name)
    except ImportError as exc:
        if 'diffq' in exc.args[0]:
            _check_diffq()
        raise

    model.eval()
    return model


def get_model_from_sig(sig: str):
    """Load the model from a locally trained XP, given its dora signature.
    This requires the training dependencies (`pip install demucs[train]`) as well
    as access to the dora folder where the XP was trained."""
    try:
        from .train import get_solver_from_sig
    except ImportError as exc:
        fatal("Loading a model from a signature requires the training dependencies, "
              f"install them with `pip install demucs[train]`. Error was: {exc}")
    solver = get_solver_from_sig(sig, model_only=True)
    if solver.best_state is not None:
        solver.model.load_state_dict(solver.best_state)
    model = solver.model
    model.eval()
    return model


def get_model_from_args(args):
    """
    Load local model package or pre-trained model.
    """
    if getattr(args, 'sig', None) is not None:
        return get_model_from_sig(args.sig)
    if args.name is None:
        args.name = DEFAULT_MODEL
    return get_model(name=args.name, repo=args.repo)
