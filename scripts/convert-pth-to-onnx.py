#!/usr/bin/env python

import torch
from torch.nn import functional as F
import argparse
from pathlib import Path
from demucs.pretrained import get_model
from demucs.htdemucs import HTDemucs

DEMUCS_MODEL = "htdemucs"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Demucs PyTorch models to ONNX')
    parser.add_argument("dest_dir", type=str, help="destination path for the converted model")

    args = parser.parse_args()

    dir_out = Path(args.dest_dir)
    dir_out.mkdir(parents=True, exist_ok=True)

    # Load the appropriate model
    model = get_model(DEMUCS_MODEL)
    model_name = DEMUCS_MODEL

    # Check if model is an instance of BagOfModels
    if isinstance(model, HTDemucs):
        core_model = model
    elif hasattr(model, 'models') and isinstance(model.models[0], HTDemucs):
        core_model = model.models[0]  # Select the first model in BagOfModels
    else:
        raise TypeError("Unsupported model type")

    # Set the model to onnx export mode
    core_model.onnx_exportable = True

    # Prepare a dummy input tensor
    dummy_waveform = torch.randn(1, 2, 343980)
    training_length = int(core_model.segment * core_model.samplerate)
    dummy_waveform = F.pad(dummy_waveform, (0, training_length - dummy_waveform.shape[-1]))
    dummy_input = (dummy_waveform)

    # Define output file name
    onnx_file_path = dir_out / f"{model_name}.onnx"
    print(f"Converting {model_name} to ONNX format...")

    # Export the core model to ONNX
    try:
        torch.onnx.export(
            core_model,
            dummy_input,
            onnx_file_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
        )
        print(f"Model successfully converted to ONNX format at {onnx_file_path}")
    except Exception as e:
        print("Error during ONNX export:", e)
