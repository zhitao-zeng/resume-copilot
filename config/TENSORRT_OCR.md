# A100 TensorRT OCR assets

The production GPU OCR mode keeps PP-OCRv6 Small detection and orientation on
CPU, uses PP-OCRv6 Medium FP32 TensorRT recognition for every line, and uses two
Small FP32 recognition engines as numeric witnesses.

The checked-in files are source code, staging scripts, and SHA256 manifests.
TensorRT engines and runtime binaries stay in the Git-ignored `models_slim/`
directory because they are large and hardware/runtime specific.

## Pinned inputs

- TensorRT: `10.13.3.9`
- Target: NVIDIA A100, FP32
- Primary Small:
  `../embodied-ai/models/ppocrv6-small-ort/rec.onnx`
- Secondary Small:
  `../embodied-ai/models/ppocrv6-small-finetune-ort/rec.onnx`
- Medium:
  `../embodied-ai/models/ppocrv6-medium-ort/rec.onnx`
- Detector, classifier, and keys:
  `../embodied-ai/models/ppocrv6-small-ort/`

Build engines on an A100 with `tools/build_resume_ocr_tensorrt.py`, using the
paths above and `--precision fp32`. The production bundle needs only:

- `primary-rec.engine`
- `secondary-rec.engine`
- `medium-rec.engine`
- `keys.txt`

Set `PPOCRV6_TRT_SOURCE_DIR` to that output directory and run
`config/stage_ppocrv6_trt_a100.sh`. Set `TENSORRT_RUNTIME_SOURCE_DIR` to the
pinned inference-only runtime and run `config/stage_tensorrt_runtime.sh`.
`config/build.sh` runs all staging checks before Docker build and push.

The inference runtime includes the NVIDIA TensorRT license and only the Python
bindings, `libnvinfer`, plugin library, and ONNX parser required by the
bindings. Builder resources are intentionally excluded from the image.

Runtime defaults are:

```text
RAPID_OCR_BACKEND=tensorrt
RAPID_OCR_DEVICE=cuda
PPOCRV6_TRT_RECOGNITION_MODE=medium-primary
PPOCRV6_TRT_RECOGNITION_PREPROCESS=native-height-pad
PPOCRV6_TRT_WITNESS_PREPROCESS=standard-resize
```

`native-height-pad` preserves the source raster for detected line crops no
taller than the recognizer's 48 px input. Taller or over-wide crops keep the
standard proportional resize path. Set the value to `standard-resize` for an
immediate, model-free rollback. The two Small numeric witnesses intentionally
keep `standard-resize`: using the same native-height transform for all three
heads created correlated date errors and defeated the purpose of consensus.
A number is therefore retained only when native-height Medium and two
standard-resize Small heads still agree exactly.

Use `PPOCRV6_TRT_RECOGNITION_MODE=small-primary` for exact CPU-primary output
parity. Missing/incompatible TensorRT assets and runtime inference errors fall
back to CPU PP-OCRv6 Small.
