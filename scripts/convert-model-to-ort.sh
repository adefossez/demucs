#!/bin/bash

# copied from https://github.com/olilarkin/ort-builder

python -m onnxruntime.tools.convert_onnx_models_to_ort $1 --enable_type_reduction