FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED 1


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    git \
    build-essential && \
    rm -rf /var/lib/apt/lists/*


WORKDIR /app

RUN pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121


RUN pip install --no-cache-dir \
    torch_scatter \
    torch_sparse \
    torch_cluster \
    torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cu121.html


RUN pip install --no-cache-dir torch_geometric


RUN pip install --no-cache-dir pandas numpy scikit-learn matplotlib seaborn


CMD ["/bin/bash"]
