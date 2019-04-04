FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python3 -m pip --no-cache-dir install --upgrade" && \
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
        ffmpeg \
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
        python3-distutils \
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
        numpy==1.16.2 \
        scipy==1.2.1 \
        pandas==0.24.2 \
        cloudpickle==0.8.1 \
        scikit-learn==0.20.3 \
        matplotlib==3.0.3 \
        Cython==0.29.6 \
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
    	torch==1.0.1 \
        torchvision==0.2.2 \
        torchtext==0.3.1 \
        && \
# ==================================================================
# apex
# ------------------------------------------------------------------
    $GIT_CLONE https://github.com/NVIDIA/apex.git ~/apex && \
    cd ~/apex && \
    $PIP_INSTALL -v --global-option="--cpp_ext" --global-option="--cuda_ext" . && \
# ==================================================================
# tensorflow
# ------------------------------------------------------------------
    $PIP_INSTALL \
        tensorflow-gpu==1.13.1 \
        && \
# ==================================================================
# keras
# ------------------------------------------------------------------
    $PIP_INSTALL \
        h5py==2.9.0 \
        keras==2.2.4 \
        && \
# ==================================================================
# opencv
# ------------------------------------------------------------------
    $PIP_INSTALL \
        opencv-python==4.0.0.21 \
        && \
# ==================================================================
# audio
# ------------------------------------------------------------------
    $PIP_INSTALL \
        librosa==0.6.3 \
        && \
# ==================================================================
# gradient boosting
# ------------------------------------------------------------------
    $PIP_INSTALL \
        xgboost==0.82 \
        lightgbm==2.2.3 \
        catboost==0.13.1 \
        && \
# ==================================================================
# nlp
# ------------------------------------------------------------------
    $PIP_INSTALL \
        nltk==3.4 \
        spacy==2.1.3 \
        gensim==3.7.1 \
        pytorch-pretrained-bert==0.6.1 \
        && \
# ==================================================================
# utilities
# ------------------------------------------------------------------
    $PIP_INSTALL \
        albumentations==0.2.2 \
        tqdm==4.31.1 \
        kaggle==1.5.3 \
        pyarrow==0.12.1 \
        && \
# ==================================================================
# visualization
# ------------------------------------------------------------------
    $PIP_INSTALL \
        plotly==3.7.1 \
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
    # $DOWNLOAD_MODEL http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth && \
    # $DOWNLOAD_MODEL http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth && \
    # $DOWNLOAD_MODEL http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth && \
    # $DOWNLOAD_MODEL http://data.lip6.fr/cadene/pretrainedmodels/resnext101_32x4d-29e315fa.pth && \
    # $DOWNLOAD_MODEL http://data.lip6.fr/cadene/pretrainedmodels/resnext101_64x4d-e77a0586.pth && \
    # $DOWNLOAD_MODEL http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth && \
    # $DOWNLOAD_MODEL http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth && \
    # $DOWNLOAD_MODEL http://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth && \
    # $DOWNLOAD_MODEL http://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth && \
    # $DOWNLOAD_MODEL http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth && \
    # $DOWNLOAD_MODEL http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth && \
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*
    
ENV TORCH_MODEL_ZOO=/models
COPY .zshrc /root/.zshrc
EXPOSE 8888 6006

