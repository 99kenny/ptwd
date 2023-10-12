FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y \
    curl \
    git \
    python3.8 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/99kenny/ptwd.git
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
RUN cd ptwd && pip install -r requirements.txt