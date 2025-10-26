# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (C) 2025 Mixxx Development Team.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Conveniance wrapper to perform STFT and iSTFT"""

import torch as th
from .stft import demucs_stft
from .istft import demucs_istft


def spectro(x, n_fft=512, hop_length=None, pad=0, onnx_exportable=False):
    *other, length = x.shape
    x = x.reshape(-1, length)
    is_mps_xpu = x.device.type in ['mps', 'xpu']
    if is_mps_xpu:
        x = x.cpu()
    
    if onnx_exportable:
        z = demucs_stft(x.view(-1, 1, length))    # z will return 1 more dimension - z.size(-1) will be 2
        _, freqs, frame, dim = z.shape
        assert dim == 2, "STFT should return complex numbers"
        return z.view(*other, freqs, frame, dim)

    z = th.stft(x,
                n_fft * (1 + pad),
                hop_length or n_fft // 4,
                window=th.hann_window(n_fft).to(x),
                win_length=n_fft,
                normalized=True,
                center=True,
                return_complex=True,
                pad_mode='reflect')
    _, freqs, frame = z.shape
    return z.view(*other, freqs, frame)


def ispectro(z, hop_length=None, length=None, pad=0, onnx_exportable=False):
    if onnx_exportable:
        # B, S, -1, Fr, T  (complex)      ----->     # B, S, -1, Fr, T, 2 shape
        *other, freqs, frames, dim = z.shape   # dim is 2
        assert dim == 2, "iSTFT should receive complex numbers"    
        n_fft = 2 * freqs - 2
        z = z.view(-1, freqs, frames, dim)
        win_length = n_fft // (1 + pad)
        is_mps_xpu = z.device.type in ['mps', 'xpu']
        if is_mps_xpu:
            z = z.cpu()
        x = demucs_istft(z[..., 0], z[..., 1], length=length)
        _, length = x.shape
        return x.view(*other, length)

    *other, freqs, frames = z.shape
    n_fft = 2 * freqs - 2
    z = z.view(-1, freqs, frames)
    win_length = n_fft // (1 + pad)
    is_mps_xpu = z.device.type in ['mps', 'xpu']
    if is_mps_xpu:
        z = z.cpu()
    x = th.istft(z,
                 n_fft,
                 hop_length,
                 window=th.hann_window(win_length).to(z.real),
                 win_length=win_length,
                 normalized=True,
                 length=length,
                 center=True)
    _, length = x.shape
    return x.view(*other, length)
