# PP-StructureV3 runtime bundle

The production raster-document parser uses four official PaddleOCR models and
disables every unrelated PP-StructureV3 module.

| Model | Role | Staged bytes |
|---|---|---:|
| `PP-DocLayout_plus-L` | document layout and reading order | 130,397,511 |
| `PP-LCNet_x1_0_textline_ori` | text-line orientation | 6,851,610 |
| `PP-OCRv5_server_det` | general text detection | 88,339,141 |
| `PP-OCRv5_server_rec` | Chinese/English text recognition | 85,215,919 |

Provenance:

- Source namespace: PaddlePaddle official models, downloaded through PaddleX
  into `official_models` on 2026-08-15.
- Runtime: `paddlepaddle-gpu==3.3.0`, `paddleocr==3.7.0`,
  `paddlex==3.7.2`.
- Isolation: the Paddle stack lives in `/opt/ppstructure-venv`; this prevents
  its CUDA 12.6 Python libraries from replacing the CUDA 12.8 dependencies of
  the vLLM/PyTorch base environment.
- License: Apache-2.0, as declared in each official model card.
- Integrity: every staged runtime file is pinned in
  `ppstructure-v3-layout-ocr.sha256`.

The image intentionally excludes `PP-Chart2Table`, table structure models,
formula recognition, seal recognition, document orientation, and unwarping.
The excluded chart model alone is about 1.4 GB and is not required when chart
recognition is disabled.

The isolated Paddle wheel is pruned during the same Docker installation layer:
LLM-only FlashAttention/FlashMask kernels and compile-time C++/CUDA headers are
removed. The four-model A100 inference smoke is rerun after pruning; Paddle's
linked CUDA, cuDNN, cuBLAS, cuSOLVER, cuSPARSE, and NVRTC libraries remain.
The pinned vLLM base already ships those libraries for CUDA 12.8, so the
Paddle cu126 virtual environment symlinks to that runtime rather than storing a
second copy. The exact mixed-runtime image is validated on A100 after build.

To stage from another verified cache:

```bash
PPSTRUCTURE_MODEL_SOURCE_DIR=/path/to/official_models \
  bash config/stage_ppstructure_v3.sh
```
