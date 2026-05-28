# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Thin client for the demucs FastAPI server.

    from demucs.client import Client

    # Single endpoint
    c = Client("http://localhost:8765")

    # Pool of endpoints discovered from a `demucs-pool` registry folder;
    # one worker is picked uniformly at random per request.
    c = Client("./demucs_pool/endpoints")

    stems = c.separate("song.mp3")                       # filename
    stems = c.separate(wav, samplerate=44100)            # torch.Tensor
    # stems: dict[str, torch.Tensor], each (channels, samples) float32
"""
import base64
import random
import tempfile
import typing as tp
from pathlib import Path

import requests
import torch
import torchaudio as ta


_SourceT = tp.Union[str, Path, "torch.Tensor"]
_TargetT = tp.Union[str, Path]


class Client:
    """HTTP client for demucs-server, with optional folder-based pool discovery.

    ``target`` is either a single URL (``"http://host:port"``) or a path to a
    directory of ``*.txt`` files each containing ``host:port`` (the format
    produced by ``demucs-pool``). In folder mode the directory is re-read on
    every request, so workers can come and go between calls.
    """

    def __init__(
        self,
        target: _TargetT = "http://localhost:8765",
        *,
        scheme: str = "http",
        timeout: float = 600.0,
        rng: tp.Optional[random.Random] = None,
    ):
        target_s = str(target)
        if isinstance(target, str) and target_s.startswith(("http://", "https://")):
            self._dir: tp.Optional[Path] = None
            self._fixed_url: tp.Optional[str] = target_s.rstrip("/")
        else:
            self._dir = Path(target_s)
            self._fixed_url = None
        self.scheme = scheme
        self.timeout = timeout
        self._rng = rng or random.Random()

    def endpoints(self) -> tp.List[str]:
        """Currently visible endpoint URLs, re-read from disk in folder mode."""
        if self._fixed_url is not None:
            return [self._fixed_url]
        assert self._dir is not None
        if not self._dir.is_dir():
            raise FileNotFoundError(f"endpoint folder not found: {self._dir}")
        urls = []
        for f in sorted(self._dir.glob("*.txt")):
            hostport = f.read_text().strip()
            if hostport:
                urls.append(f"{self.scheme}://{hostport}")
        if not urls:
            raise RuntimeError(f"no endpoints registered under {self._dir}")
        return urls

    def _pick_url(self) -> str:
        if self._fixed_url is not None:
            return self._fixed_url
        assert self._dir is not None
        # In folder mode we want a fresh pick per call, so just list and choose.
        # We don't read every file's bytes — pick a filename, then read that one.
        if not self._dir.is_dir():
            raise FileNotFoundError(f"endpoint folder not found: {self._dir}")
        files = sorted(self._dir.glob("*.txt"))
        if not files:
            raise RuntimeError(f"no endpoints registered under {self._dir}")
        f = self._rng.choice(files)
        hostport = f.read_text().strip()
        if not hostport:
            raise RuntimeError(f"empty endpoint file: {f}")
        return f"{self.scheme}://{hostport}"

    def health(self) -> dict:
        """Health from a randomly-picked endpoint (or the only one)."""
        url = self._pick_url()
        r = requests.get(f"{url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def separate(
        self,
        source: _SourceT,
        *,
        samplerate: tp.Optional[int] = None,
        stems: tp.Optional[tp.Sequence[str]] = None,
        format: str = "wav",
        mp3_bitrate: int = 320,
    ) -> tp.Dict[str, torch.Tensor]:
        """Separate ``source`` and return ``{stem: tensor (channels, samples)}``.

        ``source`` is either a path to an audio file (any ffmpeg-readable
        format) or a 2D ``torch.Tensor`` of shape ``(channels, samples)``; in
        the tensor case ``samplerate`` is required and the tensor is encoded
        as WAV before upload.

        ``stems`` filters the response; default returns all stems the server
        knows about. ``format`` selects the transport codec (``"wav"`` or
        ``"mp3"``); it does not affect the returned tensor type.
        """
        filename, file_bytes = self._prepare_payload(source, samplerate)

        params: tp.Dict[str, str] = {"format": format}
        if stems is not None:
            params["stems"] = ",".join(stems)
        if format == "mp3":
            params["mp3_bitrate"] = str(mp3_bitrate)

        url = self._pick_url()
        r = requests.post(
            f"{url}/separate",
            files={"file": (filename, file_bytes)},
            params=params,
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()

        return {name: _decode_audio_bytes(base64.b64decode(b64))
                for name, b64 in data.items()}

    @staticmethod
    def _prepare_payload(source: _SourceT,
                         samplerate: tp.Optional[int]) -> tp.Tuple[str, bytes]:
        if isinstance(source, (str, Path)):
            path = Path(source)
            return path.name, path.read_bytes()
        if isinstance(source, torch.Tensor):
            if samplerate is None:
                raise ValueError(
                    "samplerate is required when source is a torch.Tensor")
            if source.dim() != 2:
                raise ValueError(
                    f"expected a 2D (channels, samples) tensor, "
                    f"got shape {tuple(source.shape)}")
            return "input.wav", _encode_wav_bytes(source, samplerate)
        raise TypeError(
            f"unsupported source type {type(source).__name__}; "
            "expected a path or a torch.Tensor"
        )


def _encode_wav_bytes(wav: torch.Tensor, samplerate: int) -> bytes:
    wav = wav.detach().cpu()
    if wav.dtype.is_floating_point:
        wav = (wav.clamp_(-1, 1) * (2 ** 15 - 1)).short()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        ta.save(f.name, wav, sample_rate=samplerate)
        return Path(f.name).read_bytes()


def _decode_audio_bytes(data: bytes) -> torch.Tensor:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        f.write(data)
        f.flush()
        wav, _ = ta.load(f.name)
    return wav
