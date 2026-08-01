FROM python:3.11-slim

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONIOENCODING=utf-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/root/app/core

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
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple/

COPY core ./core
COPY main.py drafts.py http_compat.py ./
COPY start_api.sh ./
COPY models_slim ./models

RUN chmod +x /root/app/start_api.sh && mkdir -p /root/app/output

EXPOSE 8001
CMD ["/root/app/start_api.sh"]
