# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
FastAPI server that batches inference across concurrent clients.

A single inference worker thread drains a global queue, stacks up to MAX_BATCH
fixed-size segments into one model forward pass, and routes each result back to
the originating job. Each job owns its overlap-add accumulator and weight tally
(per-client state), mirroring the ``split=True`` branch of ``apply_model``.

Restricted to HTDemucs for now: HTDemucs pads every chunk to exactly
``segment * samplerate`` samples, which is what makes cross-client batching
trivial. Bag-of-models / Demucs / HDemucs support could be added later.
"""
import argparse
import asyncio
import base64
import logging
import os
import queue
import subprocess
import tempfile
import threading
import time
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torchaudio as ta

from .apply import BagOfModels, TensorChunk
from .audio import AudioFile, convert_audio, i16_pcm, prevent_clip
from .htdemucs import HTDemucs
from .pretrained import DEFAULT_MODEL, get_model
from .utils import center_trim

logger = logging.getLogger(__name__)


DEFAULT_MAX_BATCH = int(os.environ.get("DEMUCS_MAX_BATCH", "8"))
DEFAULT_BATCH_WAIT_MS = float(os.environ.get("DEMUCS_BATCH_WAIT_MS", "20"))
DEFAULT_OVERLAP = 0.25
DEFAULT_TRANSITION_POWER = 1.0


@dataclass
class _Job:
    length: int
    out: "torch.Tensor"        # (sources, channels, length), CPU, float32
    sum_weight: "torch.Tensor" # (length,), CPU, float32
    remaining: int
    done: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    error: tp.Optional[BaseException] = None


@dataclass
class _Segment:
    job: "_Job"
    offset: int
    chunk_length: int          # actual signal length, may be < seg_len at tail
    padded: "torch.Tensor"     # (channels, seg_len), CPU, float32


class Engine:
    """Inference engine with one worker thread and cross-client batching."""

    def __init__(
        self,
        model: HTDemucs,
        device: str,
        max_batch: int = DEFAULT_MAX_BATCH,
        batch_wait_ms: float = DEFAULT_BATCH_WAIT_MS,
        overlap: float = DEFAULT_OVERLAP,
        transition_power: float = DEFAULT_TRANSITION_POWER,
    ):
        if not isinstance(model, HTDemucs):
            raise TypeError(
                "Server currently supports only HTDemucs models. "
                f"Got {type(model).__name__}."
            )
        self.model = model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)

        self.samplerate: int = model.samplerate
        self.audio_channels: int = model.audio_channels
        self.sources: tp.List[str] = list(model.sources)

        self.segment_seconds: float = float(model.segment)
        self.segment_length: int = int(model.samplerate * self.segment_seconds)
        self.stride: int = max(1, int((1 - overlap) * self.segment_length))

        seg = self.segment_length
        weight = torch.cat([
            torch.arange(1, seg // 2 + 1, dtype=torch.float32),
            torch.arange(seg - seg // 2, 0, -1, dtype=torch.float32),
        ])
        self.weight: torch.Tensor = (weight / weight.max()) ** transition_power

        self.max_batch = max_batch
        self.batch_wait = batch_wait_ms / 1000.0

        self._queue: "queue.Queue[_Segment]" = queue.Queue()
        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._loop, name="demucs-infer", daemon=True,
        )
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()

    def separate(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Run separation on ``wav`` (shape ``(channels, length)``) at the model's
        sample rate. Returns ``(sources, channels, length)``.
        """
        if wav.dim() != 2:
            raise ValueError(f"expected 2D (channels, length), got shape {tuple(wav.shape)}")
        if wav.shape[0] != self.audio_channels:
            raise ValueError(
                f"expected {self.audio_channels} channels, got {wav.shape[0]}")

        ref = wav.mean(0)
        mean = ref.mean()
        std = ref.std()
        normed = (wav - mean) / (std + 1e-8)

        job = self._submit(normed)
        job.done.wait()
        if job.error is not None:
            raise RuntimeError(f"inference failed: {job.error}") from job.error

        # sum_weight is guaranteed > 0 everywhere because the triangle weight is
        # strictly positive on every chunk that's accumulated.
        out = job.out / job.sum_weight
        out = out * (std + 1e-8) + mean
        return out

    def _submit(self, wav: torch.Tensor) -> _Job:
        channels, length = wav.shape
        seg_len = self.segment_length
        stride = self.stride
        offsets = list(range(0, length, stride))

        job = _Job(
            length=length,
            out=torch.zeros(len(self.sources), channels, length),
            sum_weight=torch.zeros(length),
            remaining=len(offsets),
        )

        for offset in offsets:
            chunk = TensorChunk(wav, offset, seg_len)
            padded = chunk.padded(seg_len)
            self._queue.put(_Segment(
                job=job,
                offset=offset,
                chunk_length=chunk.length,
                padded=padded,
            ))
        return job

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                first = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            items: tp.List[_Segment] = [first]
            deadline = time.monotonic() + self.batch_wait
            while len(items) < self.max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    items.append(self._queue.get(timeout=remaining))
                except queue.Empty:
                    break
            try:
                self._run_batch(items)
            except BaseException as exc:
                logger.exception("inference batch failed")
                for it in items:
                    with it.job.lock:
                        if it.job.error is None:
                            it.job.error = exc
                        it.job.remaining = 0
                        it.job.done.set()

    def _run_batch(self, items: tp.List[_Segment]) -> None:
        batch = torch.stack([it.padded for it in items], dim=0).to(self.device)
        with torch.no_grad():
            out = self.model(batch)
        out = out.cpu()
        for i, it in enumerate(items):
            chunk_out = center_trim(out[i], it.chunk_length)
            w = self.weight[:it.chunk_length]
            job = it.job
            with job.lock:
                end = it.offset + it.chunk_length
                job.out[..., it.offset:end] += w * chunk_out
                job.sum_weight[it.offset:end] += w
                job.remaining -= 1
                if job.remaining == 0:
                    job.done.set()


def _decode_audio(body: bytes, filename: tp.Optional[str],
                  samplerate: int, channels: int) -> torch.Tensor:
    """Decode arbitrary audio bytes to (channels, length) at ``samplerate``."""
    suffix = Path(filename).suffix if filename else ""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as f:
        f.write(body)
        f.flush()
        path = Path(f.name)
        try:
            return AudioFile(path).read(
                streams=0, samplerate=samplerate, channels=channels,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as ffmpeg_exc:
            try:
                wav, sr = ta.load(str(path))
            except Exception as ta_exc:
                raise ValueError(
                    f"could not decode audio (ffmpeg: {ffmpeg_exc}; "
                    f"torchaudio: {ta_exc})"
                )
            return convert_audio(wav, sr, samplerate, channels)


def _encode_wav(wav: torch.Tensor, samplerate: int) -> bytes:
    # Path-based save is portable across torchaudio backends; pre-dispatcher
    # backends also reject `format=` / `encoding=` kwargs.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        ta.save(f.name, i16_pcm(wav), sample_rate=samplerate)
        return Path(f.name).read_bytes()


def _encode_mp3(wav: torch.Tensor, samplerate: int, bitrate: int) -> bytes:
    import lameenc
    channels, _ = wav.shape
    wav = i16_pcm(wav)
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(samplerate)
    encoder.set_channels(channels)
    encoder.set_quality(2)
    encoder.silence()
    pcm = wav.cpu().transpose(0, 1).numpy().tobytes()
    return encoder.encode(pcm) + encoder.flush()


def _encode_stem(wav: torch.Tensor, samplerate: int, fmt: str, mp3_bitrate: int) -> str:
    wav = prevent_clip(wav, mode="rescale")
    if fmt == "wav":
        data = _encode_wav(wav, samplerate)
    elif fmt == "mp3":
        data = _encode_mp3(wav, samplerate, mp3_bitrate)
    else:
        raise ValueError(f"unknown format {fmt!r}")
    return base64.b64encode(data).decode("ascii")


def create_app(engine: Engine):
    from fastapi import FastAPI, File, HTTPException, Query, UploadFile

    app = FastAPI(title="demucs-server")

    @app.get("/health")
    def health() -> dict:
        return {
            "sources": engine.sources,
            "samplerate": engine.samplerate,
            "audio_channels": engine.audio_channels,
            "segment_seconds": engine.segment_seconds,
            "segment_length": engine.segment_length,
            "max_batch": engine.max_batch,
            "batch_wait_ms": engine.batch_wait * 1000.0,
        }

    @app.post("/separate")
    async def separate(
        file: UploadFile = File(...),
        stems: tp.Optional[str] = Query(
            None,
            description="Comma-separated stem names; default returns all stems.",
        ),
        format: str = Query("wav", regex="^(wav|mp3)$"),
        mp3_bitrate: int = Query(320, ge=32, le=320),
    ) -> dict:
        if stems is None:
            requested = list(engine.sources)
        else:
            requested = [s.strip() for s in stems.split(",") if s.strip()]
            unknown = [s for s in requested if s not in engine.sources]
            if unknown:
                raise HTTPException(
                    400,
                    f"unknown stems {unknown}; available: {engine.sources}",
                )

        body = await file.read()
        filename = file.filename

        def _process() -> dict:
            wav = _decode_audio(body, filename, engine.samplerate, engine.audio_channels)
            out = engine.separate(wav)
            stems_dict = dict(zip(engine.sources, out))
            return {name: _encode_stem(stems_dict[name], engine.samplerate,
                                       format, mp3_bitrate)
                    for name in requested}

        try:
            return await asyncio.get_running_loop().run_in_executor(None, _process)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    return app


def _build_engine(args: argparse.Namespace) -> Engine:
    model = get_model(name=args.name, repo=args.repo)
    # Many "named" models like `htdemucs` are a single-element bag wrapping the
    # actual HTDemucs. Unwrap so cross-client batching applies.
    if isinstance(model, BagOfModels):
        if len(model.models) != 1:
            raise SystemExit(
                "demucs-server currently supports only single HTDemucs models, "
                f"not bags of {len(model.models)}."
            )
        model = model.models[0]
    if not isinstance(model, HTDemucs):
        raise SystemExit(
            f"demucs-server requires HTDemucs; got {type(model).__name__}."
        )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return Engine(
        model=model,
        device=device,
        max_batch=args.max_batch,
        batch_wait_ms=args.batch_wait_ms,
        overlap=args.overlap,
    )


def main(argv: tp.Optional[tp.List[str]] = None) -> None:
    import uvicorn

    parser = argparse.ArgumentParser("demucs-server")
    parser.add_argument("-n", "--name", default=DEFAULT_MODEL,
                        help="Pretrained model name. Must resolve to a single HTDemucs.")
    parser.add_argument("--repo", type=Path, default=None,
                        help="Folder of pre-trained models for use with -n.")
    parser.add_argument("--device", default=None,
                        help="cpu, cuda, cuda:0, ... (default: cuda if available)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-batch", type=int, default=DEFAULT_MAX_BATCH)
    parser.add_argument("--batch-wait-ms", type=float, default=DEFAULT_BATCH_WAIT_MS)
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    engine = _build_engine(args)
    logger.info(
        "demucs-server ready: model=%s device=%s sources=%s segment=%.3fs "
        "max_batch=%d batch_wait_ms=%.1f",
        args.name, engine.device, engine.sources, engine.segment_seconds,
        engine.max_batch, engine.batch_wait * 1000.0,
    )
    app = create_app(engine)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
