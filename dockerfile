FROM python:3.10

RUN echo '[global]\nindex-url=https://mirrors.aliyun.com/pypi/simple/\n' >> /etc/pip.conf

ENV TZ=Asia/Shanghai \
    DEBIAN_FRONTEND=noninteractive

COPY requirements.txt /root/requirements.txt
WORKDIR /root

RUN pip install --no-cache-dir -r /root/requirements.txt && \
    rm -rf /var/lib/apt/lists/*
COPY ./code /root/code

WORKDIR /root/code
CMD ["python", "demo/generateAPI.py"]