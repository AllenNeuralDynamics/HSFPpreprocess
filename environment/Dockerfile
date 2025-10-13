# hash:sha256:af37ae02a7dcffeb1f3f01718a46a632e989c169bcdbbbdcb157c5e5d8e502c7
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    matplotlib==3.9.4 \
    natsort==8.4.0 \
    numpy==2.0.2 \
    pandas==2.3.3 \
    pillow \
    scikit-image \
    scipy==1.13.1 \
    toml==0.10.2
