# The original upstream base tag was removed from Harbor. Pin the last known
# working self-contained image so vLLM/CUDA remain available and reproducible.
FROM harbor-contest.4pd.io/zengzhitao/resume-copilot@sha256:43d76292407f3b3fcf55c3215ee068a0e07fd18c79bda3324235243476c0436e

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONIOENCODING=utf-8

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

# Python 源码 → core/（PYTHONPATH 保证 import 兼容）
COPY core/*.py ./core/
# 根目录入口文件
COPY main.py cli.py drafts.py eval_runner.py http_compat.py ./
COPY config/start.sh ./start.sh
COPY requirements.txt ./
COPY eval_cases.jsonl ./
COPY models_slim ./models
ENV PYTHONPATH=/root/app/core

RUN apt-get -o Acquire::ForceIPv4=true -o Acquire::Retries=3 -o Acquire::http::Timeout=30 update \
    && apt-get -o Acquire::ForceIPv4=true -o Acquire::Retries=3 -o Acquire::http::Timeout=30 install -y --no-install-recommends \
    fonts-noto-cjk fonts-wqy-zenhei fonts-wqy-microhei \
    tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-eng \
    libreoffice-writer \
    curl libglib2.0-0 libgl1 libpango-1.0-0 libpangoft2-1.0-0 \
    && fc-cache -f && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    && pip cache purge

EXPOSE 80 8000

RUN chmod +x /root/app/start.sh
CMD ["/root/app/start.sh"]
