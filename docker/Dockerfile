FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-dev libopenmpi-dev git wget \
    python3 python3-dev python3-pip python3-setuptools python3-wheel \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN echo "export PATH=/usr/local/cuda/bin:$PATH" >> /etc/bash.bashrc \
    && echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> /etc/bash.bashrc

RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel packaging mpi4py \
    && pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install flash-attn==0.2.8 \
    && pip3 install lpips 

WORKDIR /home/
RUN pip3 install -e git+https://github.com/openai/consistency_models.git@main#egg=consistency_models \
    && ln -s /usr/bin/python3 /usr/bin/python

ARG USERNAME=dhein
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
