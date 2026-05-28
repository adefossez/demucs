# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Launch a pool of demucs servers via submitit / Slurm.

Each srun task runs one demucs-server bound to one GPU. By default we pack a
full 8-GPU node per allocation (``tasks_per_node=8``) so Slurm grants
exclusive node access — neighbouring partial-node jobs can't fragment around
us. Bump or shrink that via the TOML config.

Endpoint discovery: each task writes ``<host>:<port>`` to
``<folder>/endpoints/<global_rank>.txt``. The launcher prints them as they
appear and then blocks until every job finishes.

Usage:
    demucs-pool path/to/pool.toml [--folder ./demucs_pool] [--local]
"""
import argparse
import logging
import os
import shutil
import socket
import time
import typing as tp
from pathlib import Path

try:
    import tomllib
except ImportError:  # py < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from pydantic import BaseModel, Field
import submitit


logger = logging.getLogger(__name__)


class SlurmConfig(BaseModel):
    """Slurm allocation parameters. One allocation = one node."""
    partition: str = "default"
    time: int = 1440                            # minutes
    nodes: int = 1                              # how many independent allocations
    tasks_per_node: int = 8                     # one srun task = one server = one GPU
    mem_per_gpu: tp.Optional[int] = 50          # GB; total mem = mem_per_gpu * tasks_per_node
    cpus_per_gpu: int = 8
    job_name: str = "demucs-pool"
    account: tp.Optional[str] = None
    qos: tp.Optional[str] = None
    constraint: tp.Optional[str] = None
    additional_parameters: tp.Dict[str, tp.Any] = Field(default_factory=dict)


class ServerConfig(BaseModel):
    """Per-server flags forwarded to demucs-server inside each srun task."""
    model: str = "htdemucs"
    repo: tp.Optional[str] = None
    max_batch: int = 16
    batch_wait_ms: float = 20.0
    overlap: float = 0.25
    port_base: int = 8765                       # bound port = port_base + SLURM_LOCALID


class PoolConfig(BaseModel):
    slurm: SlurmConfig = Field(default_factory=SlurmConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)


def _worker(server_cfg: ServerConfig, endpoints_dir: Path) -> None:
    """Runs inside one srun task. Slurm has already pinned us to 1 GPU."""
    import torch  # local imports so the launcher can run without GPU deps
    import uvicorn

    from .server import _build_engine, create_app

    local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
    port = server_cfg.port_base + local_rank

    # Slurm exposes only the bound GPU via CUDA_VISIBLE_DEVICES, so cuda:0
    # is correct inside this task regardless of which physical GPU it is.
    args = argparse.Namespace(
        name=server_cfg.model,
        repo=Path(server_cfg.repo) if server_cfg.repo else None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_batch=server_cfg.max_batch,
        batch_wait_ms=server_cfg.batch_wait_ms,
        overlap=server_cfg.overlap,
    )
    engine = _build_engine(args)

    host = socket.gethostbyname(socket.gethostname())
    endpoint = f"{host}:{port}"
    # SLURM_PROCID resets per-allocation, so it collides across nodes when the
    # pool spans several sbatch submissions. host_port is globally unique
    # (two processes cannot bind the same host+port) and doesn't depend on
    # any Slurm env var, so it also works in --local mode.
    endpoints_dir.mkdir(parents=True, exist_ok=True)
    (endpoints_dir / f"{host}_{port}.txt").write_text(endpoint + "\n")

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [{endpoint}] %(levelname)s %(name)s %(message)s",
    )
    logger.info("ready: %s (local_rank=%d, model=%s, device=%s)",
                endpoint, local_rank, server_cfg.model, args.device)

    app = create_app(engine)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning",
                access_log=False)


def _build_executor(folder: Path, cfg: SlurmConfig,
                    local: bool) -> submitit.Executor:
    if local:
        executor = submitit.LocalExecutor(folder=str(folder))
        executor.update_parameters(
            gpus_per_node=cfg.tasks_per_node,
            timeout_min=cfg.time,
        )
        return executor

    executor = submitit.SlurmExecutor(folder=str(folder), max_num_timeout=3)
    extra: tp.Dict[str, tp.Any] = {"gpus-per-task": 1}
    extra.update(cfg.additional_parameters)
    kwargs: tp.Dict[str, tp.Any] = {
        "partition": cfg.partition,
        "time": cfg.time,
        "nodes": 1,
        "ntasks_per_node": cfg.tasks_per_node,
        "cpus_per_task": cfg.cpus_per_gpu,
        "additional_parameters": extra,
        "job_name": cfg.job_name,
        "stderr_to_stdout": True,
    }
    if cfg.mem_per_gpu:
        kwargs["mem"] = f"{cfg.mem_per_gpu * cfg.tasks_per_node}GB"
    if cfg.account:
        kwargs["account"] = cfg.account
    if cfg.qos:
        kwargs["qos"] = cfg.qos
    if cfg.constraint:
        kwargs["constraint"] = cfg.constraint
    executor.update_parameters(**kwargs)
    return executor


def main(argv: tp.Optional[tp.List[str]] = None) -> None:
    parser = argparse.ArgumentParser("demucs-pool")
    parser.add_argument("config", type=Path,
                        help="TOML config file (see pool.example.toml).")
    parser.add_argument("--folder", type=Path, default=Path("./demucs_pool"),
                        help="Submitit logs + endpoints registry (default: ./demucs_pool).")
    parser.add_argument("--local", action="store_true",
                        help="Submitit LocalExecutor instead of Slurm (for dry-runs).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    config = PoolConfig.model_validate(tomllib.loads(args.config.read_text()))
    args.folder.mkdir(parents=True, exist_ok=True)

    # Stale endpoint files from a previous run would mislead us.
    endpoints_dir = args.folder / "endpoints"
    if endpoints_dir.exists():
        stale = sum(1 for _ in endpoints_dir.glob("*.txt"))
        shutil.rmtree(endpoints_dir)
        if stale:
            logger.info("removed %d stale endpoint file(s) from %s",
                        stale, endpoints_dir)
    endpoints_dir.mkdir(parents=True)

    executor = _build_executor(args.folder, config.slurm, args.local)

    jobs: tp.List[submitit.Job] = []
    with executor.batch():
        for _ in range(config.slurm.nodes):
            jobs.append(executor.submit(_worker, config.server, endpoints_dir))

    total_tasks = config.slurm.nodes * config.slurm.tasks_per_node
    master_id = jobs[0].job_id.rsplit("_", 1)[0]
    logger.info(
        "submitted %d allocation(s), %d tasks each = %d server(s); master=%s",
        config.slurm.nodes, config.slurm.tasks_per_node, total_tasks, master_id,
    )
    logger.info("submitit folder: %s", args.folder.resolve())
    logger.info("first log: %s", args.folder / (jobs[0].job_id + "_0_log.out"))

    try:
        _wait_for_endpoints(endpoints_dir, total_tasks, jobs)
        logger.info("all %d server(s) registered an endpoint; blocking on jobs",
                    total_tasks)
        for job in jobs:
            job.result()
    except KeyboardInterrupt:
        logger.warning("interrupted; cancelling jobs")
        for job in jobs:
            job.cancel()
        raise


def _wait_for_endpoints(endpoints_dir: Path, expected: int,
                        jobs: tp.List[submitit.Job]) -> None:
    """Print each endpoint as it appears; raise if every job dies first."""
    seen: tp.Set[str] = set()
    while len(seen) < expected:
        for f in sorted(endpoints_dir.glob("*.txt")):
            if f.stem in seen:
                continue
            seen.add(f.stem)
            print(f.read_text().strip(), flush=True)
        if len(seen) >= expected:
            break
        if all(j.done() and j.state not in ("RUNNING", "PENDING") for j in jobs):
            raise RuntimeError(
                f"all jobs ended before {expected} endpoints registered "
                f"(only {len(seen)} seen)"
            )
        time.sleep(2.0)


if __name__ == "__main__":
    main()
