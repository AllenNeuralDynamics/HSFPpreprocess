# hash:sha256:786bb3635d0188cfc9a4b0717285b0699d291c7cfc24ed168c8eb50e67bbdc44
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    h5py==3.14.0 \
    matplotlib==3.9.4 \
    natsort==8.4.0 \
    numpy==1.25.2 \
    opencv-python-headless==4.7.0.72 \
    pandas==2.3.3 \
    pillow==11.3.0 \
    scikit-image==0.24.0 \
    scikit-learn==1.6.1 \
    scipy==1.13.1 \
    seaborn==0.13.2 \
    tables==3.8.0 \
    toml==0.10.2
