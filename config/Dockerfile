# syntax=docker/dockerfile:1.7
# The original upstream base tag was removed from Harbor. Pin the last known
# working self-contained image so vLLM/CUDA remain available and reproducible.
FROM harbor-contest.4pd.io/zengzhitao/resume-copilot@sha256:43d76292407f3b3fcf55c3215ee068a0e07fd18c79bda3324235243476c0436e

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONIOENCODING=utf-8 \
    LAYOUT_ORDER_ENGINE=ppstructure \
    PPSTRUCTURE_DEVICE=gpu:0 \
    PPSTRUCTURE_PYTHON=/opt/ppstructure-venv/bin/python \
    PPSTRUCTURE_MODEL_DIR=/root/app/models/ppstructure-v3/official_models \
    PADDLE_PDX_CACHE_HOME=/root/app/models/ppstructure-v3 \
    DISABLE_MODEL_SOURCE_CHECK=True \
    PPOCRV6_MODEL_DIR=/root/app/models/ppocrv6-small-ort \
    PPOCRV6_TRT_MODEL_DIR=/root/app/models/ppocrv6-trt-a100 \
    RAPID_OCR_DEVICE=cuda \
    RAPID_OCR_BACKEND=tensorrt \
    PPOCRV6_TRT_RECOGNITION_MODE=medium-primary \
    PPOCRV6_TRT_RECOGNITION_PREPROCESS=native-height-pad \
    PPOCRV6_TRT_WITNESS_PREPROCESS=standard-resize \
    RAPID_OCR_CPU_THREADS=2 \
    RAPID_OCR_ENGINE_MAX_SIDE=1600 \
    OCR_HARD_TIMEOUT_SECONDS=60 \
    LLM_COMPOSER_CALL_TIMEOUT_SECONDS=120 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    PYTHONPATH=/root/app/core:/opt/tensorrt-runtime-10.13.3.9 \
    LD_LIBRARY_PATH=/opt/tensorrt-runtime-10.13.3.9/tensorrt_libs:${LD_LIBRARY_PATH}

WORKDIR /root/app

RUN if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
      sed -i \
        -e 's|http://deb.debian.org/debian-security|https://mirrors.tuna.tsinghua.edu.cn/debian-security|g' \
        -e 's|http://deb.debian.org/debian|https://mirrors.tuna.tsinghua.edu.cn/debian|g' \
        /etc/apt/sources.list.d/debian.sources; \
    elif [ -f /etc/apt/sources.list ]; then \
      sed -i \
        -e 's|http://archive.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' \
        -e 's|http://security.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' \
        /etc/apt/sources.list; \
    fi

RUN apt-get -o Acquire::ForceIPv4=true -o Acquire::Retries=3 -o Acquire::http::Timeout=30 update \
    && apt-get -o Acquire::ForceIPv4=true -o Acquire::Retries=3 -o Acquire::http::Timeout=30 install -y --no-install-recommends \
    fonts-noto-cjk fonts-wqy-zenhei fonts-wqy-microhei \
    tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-eng \
    libreoffice-writer \
    curl libglib2.0-0 libgl1 libpango-1.0-0 libpangoft2-1.0-0 \
    && fc-cache -f && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip uninstall -y onnxruntime-gpu onnxruntime >/dev/null 2>&1 || true; \
    pip install -r requirements.txt \
      -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    && pip install --no-deps rapidocr==3.9.1 \
      -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    && python -c "from rapidocr.utils.typings import OCRVersion; assert hasattr(OCRVersion, 'PPOCRV6')"

# Paddle and the vLLM base image intentionally use separate Python
# environments. Paddle 3.3 pulls CUDA 12.6 Python libraries while the current
# PyTorch/vLLM image uses CUDA 12.8; isolating them prevents either resolver
# from replacing the other's GPU runtime packages.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /opt/ppstructure-venv \
    && /opt/ppstructure-venv/bin/pip install paddlepaddle-gpu==3.3.0 \
      -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
      --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple/ \
    && /opt/ppstructure-venv/bin/pip install \
      "paddleocr[doc-parser]==3.7.0" \
      "paddlex[ocr]==3.7.2" \
      -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    && /opt/ppstructure-venv/bin/python -c "import importlib.metadata as m; from pathlib import Path; assert m.version('paddlepaddle-gpu') == '3.3.0'; assert m.version('paddleocr') == '3.7.0'; assert m.version('paddlex') == '3.7.2'; assert (Path('/opt/ppstructure-venv/lib/python3.11/site-packages/paddle/base/libpaddle.so')).is_file()" \
    # The resume pipeline is convolution/layout inference only. Paddle's GPU
    # wheel also ships optional LLM FlashAttention kernels and C++ headers;
    # none is linked by libpaddle or loaded by the validated StructureV3 path.
    # Remove them in the installation layer so the registry does not retain
    # roughly 0.8 GiB of unreachable runtime payload.
    && psp_site=/opt/ppstructure-venv/lib/python3.11/site-packages \
    && find "$psp_site/paddle/libs" -maxdepth 1 -type f \
      \( -name 'libflashattn*.so' -o -name 'libflashmask*.so' \) -delete \
    && find "$psp_site/paddle/include" -type f -delete \
    && find "$psp_site/paddle/include" -depth -type d -empty -delete \
    && find "$psp_site/nvidia" -type f -path '*/include/*' -delete \
    && find "$psp_site/nvidia" -depth -type d -empty -delete \
    # The pinned vLLM base already contains the complete CUDA 12.x/cuDNN 9
    # runtime. CUDA minor-version compatibility lets the cu126 Paddle wheel
    # reuse those cu128 libraries, avoiding a second multi-gigabyte copy.
    && find "$psp_site/nvidia" -depth -delete \
    && find "$psp_site/cusparselt" -depth -delete \
    && ln -s /opt/conda/lib/python3.11/site-packages/nvidia "$psp_site/nvidia" \
    && test ! -e "$psp_site/paddle/libs/libflashattn.so" \
    && test ! -e "$psp_site/paddle/libs/libflashattnv3.so" \
    && test ! -e "$psp_site/paddle/libs/libflashmaskv2.so" \
    && test -L "$psp_site/nvidia" \
    && test -s "$psp_site/nvidia/cudnn/lib/libcudnn.so.9" \
    && test -s "$psp_site/nvidia/cublas/lib/libcublas.so.12"

# Application code and model artifacts change frequently; copy them after the
# expensive OS/Python dependency layers so iterative fixes keep the cache.
# core/*.py matches only top-level files, so core/v3 (and any future
# core/v4) was silently absent from every image built here.  Nothing
# failed at build or boot time because the default pipeline is v2; the
# gap only surfaced as ModuleNotFoundError under RESUME_PIPELINE_VERSION=v3.
# Copy the package whole, then assert the subpackages are present.
COPY core ./core
COPY main.py cli.py drafts.py eval_runner.py http_compat.py ./
COPY config/start.sh ./start.sh
COPY eval_cases.jsonl ./
COPY models_slim/ppocrv6-small-ort ./models/ppocrv6-small-ort
COPY models_slim/ppocrv6-trt-a100 ./models/ppocrv6-trt-a100
COPY models_slim/ppstructure-v3 ./models/ppstructure-v3
COPY models_slim/tensorrt-runtime-10.13.3.9 /opt/tensorrt-runtime-10.13.3.9
COPY config/ppocrv6-small-ort.sha256 /tmp/ppocrv6-small-ort.sha256
COPY config/ppocrv6-trt-a100-fp32.sha256 /tmp/ppocrv6-trt-a100-fp32.sha256
COPY config/ppocrv6-tensorrt-runtime-10.13.3.9.sha256 /tmp/ppocrv6-tensorrt-runtime.sha256
COPY config/ppstructure-v3-layout-ocr.sha256 /tmp/ppstructure-v3-layout-ocr.sha256

# Build-time guard: a pipeline package that the runtime can select must be
# importable in the image.  Fail the build, not a production request.
RUN test -f ./core/v3/__init__.py || (echo "core/v3 missing from image" && exit 1)

RUN test -s /root/app/models/ppocrv6-small-ort/det.onnx \
    && test -s /root/app/models/ppocrv6-small-ort/rec.onnx \
    && test -s /root/app/models/ppocrv6-small-ort/cls.onnx \
    && test -s /root/app/models/ppocrv6-small-ort/keys.txt \
    && cd /root/app/models \
    && sha256sum -c /tmp/ppocrv6-small-ort.sha256 \
    && sha256sum -c /tmp/ppocrv6-trt-a100-fp32.sha256 \
    && sha256sum -c /tmp/ppstructure-v3-layout-ocr.sha256 \
    && test ! -e /root/app/models/ppstructure-v3/official_models/PP-Chart2Table \
    && cd /opt \
    && sha256sum -c /tmp/ppocrv6-tensorrt-runtime.sha256 \
    && python -c "import tensorrt as trt; assert trt.__version__ == '10.13.3.9'" \
    && rm -f \
      /tmp/ppocrv6-small-ort.sha256 \
      /tmp/ppocrv6-trt-a100-fp32.sha256 \
      /tmp/ppstructure-v3-layout-ocr.sha256 \
      /tmp/ppocrv6-tensorrt-runtime.sha256

RUN RAPID_OCR_BACKEND=onnxruntime RAPID_OCR_DEVICE=cpu python -c "import numpy as np; import resume_io as r; r._init_rapid_ocr(); assert r._RAPID_OCR_VERSION == 'v6'; assert r._RAPID_OCR_MODEL_TYPE == 'small'; assert r._RAPID_OCR is not None; r._RAPID_OCR(np.full((64, 256, 3), 255, dtype=np.uint8)); print('PP-OCRv6 Small smoke passed:', r._RAPID_OCR_PROVIDER)"

ARG BUILD_COMMIT=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.revision="${BUILD_COMMIT}" \
      org.opencontainers.image.created="${BUILD_DATE}"
ENV RESUME_BUILD_COMMIT=${BUILD_COMMIT} \
    RESUME_BUILD_DATE=${BUILD_DATE}

EXPOSE 80 8000

RUN chmod +x /root/app/start.sh
CMD ["/root/app/start.sh"]
