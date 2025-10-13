# hash:sha256:ab53c6146c7d4311aa1809cacf626ede58a2d472cea215435b27ab06a90c42a8
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    cv2 \
    matplotlib==3.9.4 \
    natsort==8.4.0 \
    numpy==2.0.2 \
    opencv-python-headless \
    pandas==2.3.3 \
    pillow==11.3.0 \
    scikit-image==0.24.0 \
    scipy==1.13.1 \
    toml==0.10.2
