FROM harbor-contest.4pd.io/lizhuoju/menu-translate:base

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONIOENCODING=utf-8

WORKDIR /root/app

RUN if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
      sed -i \
        -e 's|http://deb.debian.org/debian-security|https://mirrors.tuna.tsinghua.edu.cn/debian-security|g' \
        -e 's|http://deb.debian.org/debian|https://mirrors.tuna.tsinghua.edu.cn/debian|g' \
        /etc/apt/sources.list.d/debian.sources; \
    fi

# Python 源码 → core/（PYTHONPATH 保证 import 兼容）
COPY core/*.py ./core/
# 根目录入口文件
COPY main.py cli.py drafts.py eval_runner.py http_compat.py ./
COPY start_api.sh ./
COPY requirements.txt ./
COPY eval_cases.jsonl ./
COPY models_slim ./models
ENV PYTHONPATH=/root/app/core

RUN apt-get -o Acquire::ForceIPv4=true -o Acquire::Retries=3 -o Acquire::http::Timeout=30 update \
    && apt-get -o Acquire::ForceIPv4=true -o Acquire::Retries=3 -o Acquire::http::Timeout=30 install -y --no-install-recommends \
    fonts-noto-cjk fonts-wqy-zenhei fonts-wqy-microhei \
    tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-eng \
    libreoffice-writer \
    curl libglib2.0-0 libgl1 libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0 \
    && fc-cache -f && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    && pip cache purge

EXPOSE 80 8000

RUN chmod +x /root/app/start_api.sh
CMD ["/root/app/start_api.sh"]
