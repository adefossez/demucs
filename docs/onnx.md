# Exporting HTDemucs Model

## Convert PyTorch model to ONNX and ORT

Convert Demucs PyTorch model to ONNX:
```
$ python ./scripts/convert-pth-to-onnx.py ./scripts/demucs-onnx
```

Then, convert ONNX to ORT:
```
$ ./scripts/convert-model-to-ort.sh 
```