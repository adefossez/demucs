# Copyright (C) 2025 Mixxx Development Team
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# First author is Anmol Mishra.
"""
Test to compare the spectrogram outputs from PyTorch's STFT and ONNX exportable STFT.
"""
from demucs.spec import spectro
import pytest
import torch

def spectroAPI(x, n_fft=512, hop_length=None, pad=0):
    return NotImplementedError("This function is just a placeholder for the spectro function.")

@pytest.fixture(scope='session')
def input_waveform():
    batch, channels, samples = 1, 2, int(7.8*44100)  # 7.8 seconds of audio at 44100 Hz
    yield torch.randn(batch, channels, samples)

def test_spectrogram_shape_pytorch(input_waveform):
    """ Test the shape of the spectrogram output using PyTorch's STFT """
    b, c, s = input_waveform.shape
    n_fft, hop_length = 4096, 1024
    spec = spectro(input_waveform, n_fft=n_fft, hop_length=hop_length, pad=0, onnx_exportable=False)
    
    assert spec.shape == (b, c, n_fft // 2 + 1, s // hop_length + 1)  # (batch, channels, freq_bins, time_steps)

def test_spectrogram_shape_onnx_exportable(input_waveform):
    """ Test the shape of the spectrogram output using ONNX exportable STFT """
    b, c, s = input_waveform.shape
    n_fft, hop_length = 4096, 1024
    spec = spectro(input_waveform, n_fft=n_fft, hop_length=hop_length, pad=0, onnx_exportable=True)
    
    assert spec.shape == (b, c, n_fft // 2 + 1, s // hop_length + 1, 2)  # (batch, channels, freq_bins, time_steps, 2)

def test_compare_spectrograms(input_waveform):
    """ Compare the spectrograms from PyTorch and ONNX exportable STFT """
    n_fft, hop_length = 4096, 1024
    spec_pytorch = torch.view_as_real(
        spectro(input_waveform, n_fft=n_fft, hop_length=hop_length, pad=0, onnx_exportable=False)
    )   # Convert to real tensor for comparison
    spec_onnx = spectro(input_waveform, n_fft=n_fft, hop_length=hop_length, pad=0, onnx_exportable=True)

    # Calculate the mean difference between the two spectrograms
    mean_diff = torch.abs(spec_pytorch - spec_onnx).mean().item()
    print("STFT Result: Mean Difference =", mean_diff)

    assert mean_diff < 1e-4, "Spectrograms do not match!"
