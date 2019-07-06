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
        numpy==1.16.4 \
        scipy==1.3.0 \
        pandas==0.24.2 \
        cloudpickle==0.8.1 \
        scikit-learn==0.21.2 \
        matplotlib==3.1.0 \
        Cython==0.29.10 \
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
    	torch==1.1.0 \
        torchvision==0.3.0 \
        torchtext==0.3.1 \
        && \
# # ==================================================================
# # apex
# # ------------------------------------------------------------------
#     $GIT_CLONE https://github.com/NVIDIA/apex.git ~/apex && \
#     cd ~/apex && \
#     $PIP_INSTALL -v --global-option="--cpp_ext" --global-option="--cuda_ext" . && \
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
        opencv-python==4.1.0.25 \
        && \
# # ==================================================================
# # audio
# # ------------------------------------------------------------------
#     $PIP_INSTALL \
#         librosa==0.6.3 \
#         && \
#     $GIT_CLONE https://github.com/pytorch/audio.git ~/torchaudio && \
#     cd ~/torchaudio && \
#     python setup.py install && \
# ==================================================================
# mmdetection
# ------------------------------------------------------------------
    $GIT_CLONE git clone https://github.com/open-mmlab/mmdetection.git ~/mmdetection && \
    cd ~/mmdetection && \
    ./compile.sh && \
    python setup.py install && \
# ==================================================================
# gradient boosting
# ------------------------------------------------------------------
    $PIP_INSTALL \
        xgboost==0.90 \
        lightgbm==2.2.3 \
        catboost==0.15.1 \
        && \
# ==================================================================
# nlp
# ------------------------------------------------------------------
    $PIP_INSTALL \
        nltk==3.4.3 \
        spacy==2.1.4 \
        gensim==3.7.3 \
        pytorch-pretrained-bert==0.6.2 \
        && \
# ==================================================================
# utilities
# ------------------------------------------------------------------
    $PIP_INSTALL \
        albumentations==0.2.3 \
        tqdm==4.32.1 \
        fastprogress=0.1.21 \
        kaggle==1.5.4 \
        pyarrow==0.13.0 \
        && \
# ==================================================================
# visualization
# ------------------------------------------------------------------
    $PIP_INSTALL \
        plotly==3.10.0 \
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
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*
    
ENV TORCH_MODEL_ZOO=/models
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
COPY .zshrc /root/.zshrc
EXPOSE 8888 6006

