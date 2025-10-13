# hash:sha256:b892c00b005eca910bf7e0ca7149f0c9caa024a2419a4a852c30d9d23537823f
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    allensdk==0.16.3 \
    matplotlib==3.9.4 \
    natsort==8.4.0 \
    numpy==2.0.2 \
    opencv-python-headless==4.12.0.88 \
    pandas==2.3.3 \
    pillow==11.3.0 \
    scikit-image==0.24.0 \
    scikit-learn==1.6.1 \
    scipy==1.13.1 \
    seaborn==0.13.2 \
    toml==0.10.2
