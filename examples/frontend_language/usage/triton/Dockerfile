FROM nvcr.io/nvidia/tritonserver:24.01-py3

RUN apt-get update && apt-get install -y --only-upgrade ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

RUN git clone https://github.com/sgl-project/sglang.git

WORKDIR /opt/sglang
RUN pip install --upgrade pip && \
    pip install -e "python[all]" && \
    pip install datasets
