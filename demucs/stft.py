# Copyright (C) 2025 Mixxx Development Team
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# First author is Anmol Mishra.
"""
This module implements a custom STFT process that is compatible with ONNX export.
It uses PyTorch's convolution operations to compute the STFT, avoiding the use of complex
numbers directly, which can be problematic for ONNX export.
"""
import torch
import enum


# Constants set for Demucs
NFFT = 4096                                         # Number of FFT components for the STFT process
HOP_LENGTH = 1024                                   # Number of samples between successive frames in the STFT
WINDOW_TYPE = 'hann'                                # Type of window function used in the STFT
WINDOW_LENGTH = NFFT                                # Length of the window function
NORMALIZED = True                                   # Whether to normalize the window function
CENTER = True                                       # Whether to center the input signal before STFT
PAD_MODE = 'reflect'                                # Select reflect or constant padding mode for the STFT process.

# Enum for window types 
class WindowType(enum.StrEnum):
    BARTLETT = 'bartlett'
    BLACKMAN = 'blackman'
    HAMMING = 'hamming'
    HANN = 'hann'
    KAISER = 'kaiser'

    def __call__(self, window_length):
        match self:
            case WindowType.BARTLETT:
                return torch.bartlett_window(window_length)
            case WindowType.BLACKMAN:
                return torch.blackman_window(window_length)
            case WindowType.HAMMING:
                return torch.hamming_window(window_length)
            case WindowType.HANN:
                return torch.hann_window(window_length)
            case WindowType.KAISER:
                return torch.kaiser_window(window_length, periodic=True, beta=12.0)
            case _:
                raise NotImplementedError(f"Window type {self} doesn't yet have a function.") 

class STFT_Process(torch.nn.Module):
    def __init__(self, n_fft=NFFT, hop_len=HOP_LENGTH, window_type=WINDOW_TYPE, window_length=WINDOW_LENGTH, normalized=NORMALIZED, center=CENTER, pad_mode=PAD_MODE):
        super(STFT_Process, self).__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.window_type = window_type
        self.window_length = window_length
        self.normalized = normalized
        self.center = center
        self.pad_mode = pad_mode
        self.half_n_fft = n_fft // 2  # Precompute once
        
        # Get window function and compute window once
        if self.window_length != self.n_fft:
            raise NotImplementedError(f"The case of window length not equal to n_fft is not implemented in {self.__class__.__name__}.")
        window = WindowType(window_type)(self.window_length).float()

        # STFT forward pass preparation
        time_steps = torch.arange(self.n_fft, dtype=torch.float32).unsqueeze(0)
        frequencies = torch.arange(self.half_n_fft + 1, dtype=torch.float32).unsqueeze(1)
        
        # Calculate omega matrix once
        omega = 2 * torch.pi * frequencies * time_steps / n_fft

        # Calculate window function
        cos_kernel = (torch.cos(omega) * window.unsqueeze(0)).unsqueeze(1)
        sin_kernel = (-torch.sin(omega) * window.unsqueeze(0)).unsqueeze(1)

        # Normalize window if needed
        if normalized:
            cos_kernel = cos_kernel / torch.sqrt(torch.tensor([n_fft], dtype=torch.float32))
            sin_kernel = sin_kernel / torch.sqrt(torch.tensor([n_fft], dtype=torch.float32))

        # Register conv kernels as buffers
        self.register_buffer('cos_kernel', (cos_kernel))
        self.register_buffer('sin_kernel', (sin_kernel))

    def forward(self, x):
        if self.center:
            x_padded = torch.nn.functional.pad(x, (self.half_n_fft, self.half_n_fft), mode=self.pad_mode)
        else:
            x_padded = x

        # Perform convolutions
        real_part = torch.nn.functional.conv1d(x_padded, self.cos_kernel, stride=self.hop_len)
        image_part = torch.nn.functional.conv1d(x_padded, self.sin_kernel, stride=self.hop_len)

        # return real_part, image_part
        return torch.stack((real_part, image_part), dim=-1)

demucs_stft = STFT_Process(
    n_fft=NFFT,
    hop_len=HOP_LENGTH,
    window_type=WINDOW_TYPE,
    window_length=WINDOW_LENGTH,
    normalized=NORMALIZED,
    center=CENTER,
    pad_mode=PAD_MODE
)
