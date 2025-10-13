# hash:sha256:bf9287f8cb497d922464cc6c5c4b81f501ae55560f1805f437f59183b54e5065
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    matplotlib \
    numpy \
    pandas \
    scipy
