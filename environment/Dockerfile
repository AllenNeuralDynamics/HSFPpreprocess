# hash:sha256:a49cb97cb5ef77d32f83b5dcb8a323ae6239a8fe570b969f931ba0ea19bf0427
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    allensdk \
    matplotlib==3.9.4 \
    natsort==8.4.0 \
    numpy==2.0.2 \
    opencv-python-headless==4.12.0.88 \
    pandas==2.3.3 \
    pillow==11.3.0 \
    scikit-image==0.24.0 \
    scikit-learn \
    scipy==1.13.1 \
    seaborn \
    toml==0.10.2
