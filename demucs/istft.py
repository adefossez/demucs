# Copyright (C) 2025 Mixxx Development Team
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# First author is Anmol Mishra.
"""
This module implements a custom ISTFT process that is compatible with ONNX export.
It uses PyTorch's convolution operations to compute the inverse STFT, avoiding the use of
complex numbers directly, which can be problematic for ONNX export.
"""
import torch
import enum


# Constants set for Demucs
NFFT = 4096                                         # Number of FFT components for the STFT process
HOP_LENGTH = 1024                                   # Number of samples between successive frames in the STFT
WINDOW_TYPE = 'hann'                                # Type of window function used in the STFT
WINDOW_LENGTH = NFFT                                # Length of the window function
NORMALIZED = True                                   # Whether to normalize the window function
MAX_SIGNAL_LENGTH = int(44100 * 8)                  # Maximum length of the audio signal WITH padding (8 seconds at 44100 Hz)
MAX_FRAMES = MAX_SIGNAL_LENGTH // HOP_LENGTH + 1    # Maximum number of frames for the audio length after STFT processed.
CENTER = True                                       # Whether to center the input signal before STFT

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

class ISTFT_Process(torch.nn.Module):
    def __init__(self, n_fft=NFFT, hop_len=HOP_LENGTH, window_type=WINDOW_TYPE, window_length=WINDOW_LENGTH, normalized=NORMALIZED, max_frames=MAX_FRAMES, center=CENTER):
        super(ISTFT_Process, self).__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.window_type = window_type
        self.window_length = window_length
        self.normalized = normalized
        self.max_frames = max_frames
        self.center = center
        self.half_n_fft = n_fft // 2  # Precompute once

        # Get window function and compute window once
        if self.window_length != self.n_fft:
            raise NotImplementedError(f"The case of window length not equal to n_fft is not implemented in {self.__class__.__name__}.")
        window = WindowType(window_type)(self.window_length).float()

        # Check if center is false
        if not self.center:
            raise NotImplementedError("No centering is not supported in this implementation.")

        # ISTFT forward pass preparation
        # Pre-compute fourier basis
        fourier_basis = torch.fft.fft(torch.eye(n_fft, dtype=torch.float32))
        fourier_basis = torch.vstack([
            torch.real(fourier_basis[:self.half_n_fft + 1, :]),
            torch.imag(fourier_basis[:self.half_n_fft + 1, :])
        ]).float()
        
        # Create forward and inverse basis
        forward_basis = window * fourier_basis[:, None, :]
        inverse_basis = window * torch.linalg.pinv((fourier_basis * n_fft) / hop_len).T[:, None, :]
        
        # Calculate window sum for overlap-add
        n = n_fft + hop_len * (max_frames - 1)
        window_sum = torch.zeros(n, dtype=torch.float32)
        window_normalized = window / window.abs().max()
        
        # Pad window if needed
        total_pad = n_fft - window_normalized.shape[0]
        if total_pad > 0:
            pad_left = total_pad // 2
            pad_right = total_pad - pad_left
            win_sq = torch.nn.functional.pad(window_normalized ** 2, (pad_left, pad_right), mode='constant', value=0)
        else:
            win_sq = window_normalized ** 2
        
        # Calculate overlap-add weights
        for i in range(max_frames):
            sample = i * hop_len
            window_sum[sample: min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]
        
        # Normalize window if needed
        if normalized:
            inverse_basis = inverse_basis * torch.sqrt(torch.tensor([n_fft], dtype=torch.float32))
        
        # Register buffers
        self.register_buffer("forward_basis", forward_basis)
        self.register_buffer("inverse_basis", inverse_basis)
        self.register_buffer("window_sum_inv", n_fft / (window_sum * hop_len + 1e-8))  # Add epsilon to avoid division by zero

    def forward(self, real, imag, length=None):
        # Calculate magnitude and phase from real and imaginary parts
        magnitude = torch.sqrt(real ** 2 + imag ** 2)
        phase = torch.atan2(imag, real + torch.finfo(real.dtype).eps)  # Add epsilon to avoid division by zero

        # Pre-compute trig values
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        
        # Prepare input for transposed convolution
        complex_input = torch.cat((magnitude * cos_phase, magnitude * sin_phase), dim=1)
        
        # Perform transposed convolution
        inverse_transform = torch.nn.functional.conv_transpose1d(
            complex_input,
            self.inverse_basis,
            stride=self.hop_len,
            padding=0,
        )
        
        # Apply window correction
        output_len = inverse_transform.size(-1)
        start_idx = self.half_n_fft
        end_idx = output_len
        
        output =  inverse_transform[:, :, start_idx:end_idx] * self.window_sum_inv[start_idx:end_idx]

        # If length is specified, trim the output to the desired length
        if length:
            pad_len = torch.clamp(torch.tensor(length) - output.size(-1), min=0)

            # Create a zero pad tensor regardless of need
            pad = torch.zeros(
                output.size(0), output.size(1), pad_len,
                dtype=output.dtype, device=output.device
            )

            # Always cat, pad_len will be 0 if not needed
            output = torch.cat([output, pad], dim=-1)

            # Crop in all cases to enforce exact length
            output = output[..., :length]

        output = output.squeeze(dim=1)
        return output

demucs_istft = ISTFT_Process(
    n_fft=NFFT,
    hop_len=HOP_LENGTH,
    window_type=WINDOW_TYPE,
    window_length=WINDOW_LENGTH,
    normalized=NORMALIZED,
    max_frames=MAX_FRAMES,
    center=CENTER,
)
