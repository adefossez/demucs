# Copyright (C) 2025 Mixxx Development Team
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# First author is Anmol Mishra.
"""
Test to compare the iSTFT outputs from PyTorch's iSTFT and ONNX exportable iSTFT.
"""
from demucs.spec import ispectro
import pytest
import torch


def ispectroAPI(z, hop_length=None, length=None, pad=0):
    return NotImplementedError("This function is just a placeholder for the ispectro function.")

@pytest.fixture(scope='session')
def input_spectrogram():
    """ Fixture to generate a random spectrogram for testing """
    batch, sources, channels = 1, 4, 2
    samples, n_fft, hop_length = int(7.8 * 44100), 4096, 1024
    freq_bins, time_steps = n_fft//2 + 1 , samples//hop_length + 1

    z = torch.randn(batch, sources, channels, freq_bins, time_steps, 2)  # (batch, sources, channels, freq_bins, time_steps, 2)
    return z, samples, hop_length

def test_istft_shape_pytorch(input_spectrogram):
    """ Test the shape of the iSTFT output using PyTorch's iSTFT """
    z, samples, hop_length = input_spectrogram
    batch, sources, channels = z.shape[0], z.shape[1], z.shape[2]
    x = ispectro(torch.view_as_complex(z), hop_length=hop_length, length=samples, onnx_exportable=False)

    assert x.shape == (batch, sources, channels, samples)  # (batch, sources, channels, samples)

def test_istft_shape_onnx_compatible(input_spectrogram):
    """ Test the shape of the iSTFT output using ONNX compatible iSTFT """
    z, samples, hop_length = input_spectrogram
    batch, sources, channels = z.shape[0], z.shape[1], z.shape[2]
    x = ispectro(z, hop_length=hop_length, length=samples, onnx_exportable=True)

    assert x.shape == (batch, sources, channels, samples)  # (batch, sources, channels, samples)

def test_compare_istfts(input_spectrogram):
    """ Compare the iSTFTs from PyTorch and ONNX compatible iSTFT """
    z, samples, hop_length = input_spectrogram

    x_pytorch = ispectro(torch.view_as_complex(z), hop_length=hop_length, length=samples-hop_length, onnx_exportable=False)
    x_onnx = ispectro(z, hop_length=hop_length, length=samples-hop_length, onnx_exportable=True)

    # Calculate the mean difference between the two iSTFTs
    mean_diff = torch.abs(x_pytorch - x_onnx).mean().item()
    print("\niSTFT Result: Mean Difference =", mean_diff)

    assert mean_diff < 1e-4, "iSTFTs do not match!"
