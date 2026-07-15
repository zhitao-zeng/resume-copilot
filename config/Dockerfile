FROM harbor-contest.4pd.io/lizhuoju/menu-translate:base

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONIOENCODING=utf-8

WORKDIR /root/app

# Python 源码 → core/（PYTHONPATH 保证 import 兼容）
COPY core/*.py ./core/
# 根目录入口文件
COPY main.py cli.py drafts.py eval_runner.py http_compat.py ./
COPY config/*.sh ./
COPY config/requirements.txt ./
COPY eval_cases.jsonl ./
COPY models_slim ./models
ENV PYTHONPATH=/root/app/core

RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-noto-cjk fonts-wqy-zenhei fonts-wqy-microhei \
    tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-eng \
    curl \
    && fc-cache -f && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    && pip install Flask==2.3.0 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    && pip cache purge

EXPOSE 80 8000

RUN chmod +x /root/app/start.sh
CMD ["/root/app/start.sh"]
