# Copyright (C) 2025 Mixxx Development Team
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# First author is Anmol Mishra.
"""
Test for the HTDemucs model ONNX export branch functionality.
"""

from demucs.htdemucs import HTDemucs
import pytest
import torch

@pytest.fixture
def htdemucs_model():
    return HTDemucs(sources=['drums', 'bass', 'other', 'vocals'])

@pytest.fixture
def input_waveform():
    batch, channels, samples = 1, 2, int(44100 * 7.8)  # 7.8 seconds of audio at 44100 Hz is the expected input waveform for HTDemucs
    return torch.randn(batch, channels, samples)

def test_onnx_flag(htdemucs_model, input_waveform):
    # Run forward pass with default complex tensor path
    htdemucs_model.onnx_exportable = False  # The default state
    output = htdemucs_model(input_waveform)

    # Now set onnx_exportable to True to get the ONNX path output
    htdemucs_model.onnx_exportable = True
    output_onnx = htdemucs_model(input_waveform)
    
    # Check if the outputs are identical
    assert torch.allclose(output, output_onnx, atol=1e-5), "Outputs should be identical for onnx_exportable=False and True"