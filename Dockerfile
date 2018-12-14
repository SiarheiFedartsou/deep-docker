FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    DOWNLOAD_MODEL="wget -P /models" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
# ==================================================================
# tools
# ------------------------------------------------------------------
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        ca-certificates \
        cmake \
        wget \
        git \
        vim \
        zsh \
        && \
# ==================================================================
# oh my zsh
# ------------------------------------------------------------------
    wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true \
    && \
# ==================================================================
# python
# ------------------------------------------------------------------
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.6 \
        python3.6-dev \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.6 ~/get-pip.py && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python && \
    $PIP_INSTALL \
        setuptools \
        && \
    $PIP_INSTALL \
        numpy==1.14.5 \
        scipy==1.1.0 \
        pandas==0.23.4 \
        cloudpickle==0.5.3 \
        scikit-learn==0.20.0 \
        matplotlib==2.2.2 \
        Cython==0.28.5 \
        && \
# ==================================================================
# jupyter
# ------------------------------------------------------------------
    $PIP_INSTALL \
        jupyter==1.0.0 \
        && \
# ==================================================================
# pytorch
# ------------------------------------------------------------------
    $PIP_INSTALL \
    	torch==1.0.0 \
        torchvision==0.2.1 \
        torchtext==0.3.1 \
        && \
# ==================================================================
# tensorflow
# ------------------------------------------------------------------
    $PIP_INSTALL \
        tensorflow-gpu==1.10.0 \
        && \
# ==================================================================
# keras
# ------------------------------------------------------------------
    $PIP_INSTALL \
        h5py==2.8.0 \
        keras==2.2.2 \
        && \
# ==================================================================
# opencv
# ------------------------------------------------------------------
    $PIP_INSTALL \
        opencv-python==3.4.4.19 \
        && \
# ==================================================================
# gradient boosting
# ------------------------------------------------------------------
    $PIP_INSTALL \
        xgboost==0.80 \
        lightgbm==2.1.2 \
        catboost==0.9.1.1 \
        && \
# ==================================================================
# nlp
# ------------------------------------------------------------------
    $PIP_INSTALL \
        nltk==3.4 \
        spacy==2.0.18 \
        gensim==3.6.0 \
        && \
# ==================================================================
# utilities
# ------------------------------------------------------------------
    $PIP_INSTALL \
        albumentations==0.1.8 \
        && \
# ==================================================================
# visualization
# ------------------------------------------------------------------
    $PIP_INSTALL \
        plotly==3.4.2 \
        seaborn==0.9.0 \
        && \
# ==================================================================
# model's zoo
# ------------------------------------------------------------------
    $PIP_INSTALL \
        pretrainedmodels==0.7.4 \
        && \
    $DOWNLOAD_MODEL https://download.pytorch.org/models/resnet18-5c106cde.pth && \
    $DOWNLOAD_MODEL https://download.pytorch.org/models/resnet34-333f7ec4.pth && \
    $DOWNLOAD_MODEL https://download.pytorch.org/models/resnet50-19c8e357.pth && \
    $DOWNLOAD_MODEL https://download.pytorch.org/models/vgg11-bbd30ac9.pth && \
    $DOWNLOAD_MODEL https://download.pytorch.org/models/vgg13-c768596a.pth && \
    $DOWNLOAD_MODEL https://download.pytorch.org/models/vgg16-397923af.pth && \
    $DOWNLOAD_MODEL https://download.pytorch.org/models/vgg19-dcbb9e9d.pth && \
    $DOWNLOAD_MODEL https://download.pytorch.org/models/vgg11_bn-6002323d.pth && \
    $DOWNLOAD_MODEL https://download.pytorch.org/models/vgg13_bn-abd245e5.pth && \
    $DOWNLOAD_MODEL https://download.pytorch.org/models/vgg16_bn-6c64b313.pth && \
    $DOWNLOAD_MODEL https://download.pytorch.org/models/vgg19_bn-c79401a0.pth && \
    $DOWNLOAD_MODEL http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth && \
    $DOWNLOAD_MODEL http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth && \
    $DOWNLOAD_MODEL http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth && \
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*
    
ENV TORCH_MODEL_ZOO=/models
EXPOSE 8888 6006
