# Exporting HTDemucs Model

## Convert PyTorch model to ONNX and ORT

Convert Demucs PyTorch model to ONNX:

```python
python ./scripts/convert-pth-to-onnx.py ./onnx-models
```

Optionally, convert ONNX model to ORT:

```python
python -m onnxruntime.tools.convert_onnx_models_to_ort ./onnx-models --enable_type_reduction 
```
