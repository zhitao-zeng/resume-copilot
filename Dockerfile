FROM python:3.11-slim

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONIOENCODING=utf-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/root/app/core \
    PPOCRV6_MODEL_DIR=/root/app/models/ppocrv6-small-ort \
    RAPID_OCR_DEVICE=cpu \
    RAPID_OCR_CPU_THREADS=2 \
    RAPID_OCR_ENGINE_MAX_SIDE=1600 \
    OCR_HARD_TIMEOUT_SECONDS=60 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2

WORKDIR /root/app

RUN sed -i \
    -e 's|http://deb.debian.org/debian-security|https://mirrors.tuna.tsinghua.edu.cn/debian-security|g' \
    -e 's|http://deb.debian.org/debian|https://mirrors.tuna.tsinghua.edu.cn/debian|g' \
    /etc/apt/sources.list.d/debian.sources

RUN apt-get -o Acquire::ForceIPv4=true -o Acquire::Retries=3 -o Acquire::http::Timeout=30 update \
    && apt-get -o Acquire::ForceIPv4=true -o Acquire::Retries=3 -o Acquire::http::Timeout=30 install -y --no-install-recommends \
    curl fonts-noto-cjk fonts-wqy-zenhei fonts-wqy-microhei \
    tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-eng \
    libreoffice-writer \
    libglib2.0-0 libgl1 libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0 \
    && fc-cache -f \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip uninstall -y onnxruntime-gpu onnxruntime >/dev/null 2>&1 || true; \
    pip install --no-cache-dir -r requirements.txt \
      -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    && pip install --no-cache-dir --no-deps rapidocr==3.9.1 \
      -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    && python -c "from rapidocr.utils.typings import OCRVersion; assert hasattr(OCRVersion, 'PPOCRV6')"

COPY core ./core
COPY main.py drafts.py http_compat.py ./
COPY start_api.sh ./
COPY models_slim/ppocrv6-small-ort ./models/ppocrv6-small-ort
COPY config/ppocrv6-small-ort.sha256 /tmp/ppocrv6-small-ort.sha256

RUN test -s /root/app/models/ppocrv6-small-ort/det.onnx \
    && test -s /root/app/models/ppocrv6-small-ort/rec.onnx \
    && test -s /root/app/models/ppocrv6-small-ort/cls.onnx \
    && test -s /root/app/models/ppocrv6-small-ort/keys.txt \
    && cd /root/app/models \
    && sha256sum -c /tmp/ppocrv6-small-ort.sha256 \
    && rm -f /tmp/ppocrv6-small-ort.sha256

RUN python -c "import numpy as np; import resume_io as r; r._init_rapid_ocr(); assert r._RAPID_OCR_VERSION == 'v6'; assert r._RAPID_OCR_MODEL_TYPE == 'small'; assert r._RAPID_OCR is not None; r._RAPID_OCR(np.full((64, 256, 3), 255, dtype=np.uint8)); print('PP-OCRv6 Small smoke passed:', r._RAPID_OCR_PROVIDER)"

ARG BUILD_COMMIT=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.revision="${BUILD_COMMIT}" \
      org.opencontainers.image.created="${BUILD_DATE}"
ENV RESUME_BUILD_COMMIT=${BUILD_COMMIT} \
    RESUME_BUILD_DATE=${BUILD_DATE}

RUN chmod +x /root/app/start_api.sh && mkdir -p /root/app/output

EXPOSE 8001
CMD ["/root/app/start_api.sh"]
