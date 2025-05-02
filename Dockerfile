
ARG UBUNTU_VERSION=20.04

ARG ARCH=
ARG CUDA=11.3
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}.1-base-ubuntu${UBUNTU_VERSION} as base

ARG ARCH
ARG CUDA
ARG CUDNN=8.1
ARG CUDNN_MAJOR_VERSION=8
ARG LIB_DIR_PREFIX=x86_64
ARG LIBNVINFER=6.0.1-1
ARG LIBNVINFER_MAJOR_VERSION=6

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN echo $GROUP_ID $USER_ID
RUN addgroup --gid $GROUP_ID user
RUN useradd -l -u $USER_ID -g $GROUP_ID user
RUN mkdir /home/user
RUN chown -R user.user /home/user

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 35
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub


# The following is to automatically select a locale during installation

RUN export DEBIAN_FRONTEND=noninteractive; \
    export DEBCONF_NONINTERACTIVE_SEEN=true; \
    echo 'tzdata tzdata/Areas select Etc' | debconf-set-selections; \
    echo 'tzdata tzdata/Zones/Etc select US' | debconf-set-selections; \
    apt-get update -qqy \
 && apt-get install -qqy --no-install-recommends \
        tzdata \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Needed for string substitution
SHELL ["/bin/bash", "-c"]

# install tensorflow dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-${CUDA/./-} \
        # There appears to be a regression in libcublas10=10.2.2.89-1 which
        # prevents cublas from initializing in TF. See
        # https://github.com/tensorflow/tensorflow/issues/9489#issuecomment-562394257
        libcublas-${CUDA/./-} \
        cuda-nvrtc-${CUDA/./-} \
        libcufft-${CUDA/./-} \
        libcurand-${CUDA/./-} \
        libcusolver-${CUDA/./-} \
        libcusparse-${CUDA/./-} \
        curl \
        #libcudnn8=${CUDNN}+cuda${CUDA} \
        libcudnn8 \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip


# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
     python3 \
     python3-pip \
     libexpat1-dev \
     libicu-dev \
     libigraph0-dev

RUN python3 -m pip --no-cache-dir install --upgrade \
     pip \
     setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which python3) /usr/local/bin/python

RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN apt-get update && apt-get install -y build-essential autoconf gsl-bin libgsl-dev wget unzip vim

RUN pip install matplotlib==3.3.4
# OpenCV
RUN apt-get update  && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install opencv-python==4.5.3.56

RUN pip install Pillow==8.3.2
RUN pip install pytorch-fid==0.2.0
RUN pip install scikit-image==0.17.2
RUN pip install scipy==1.5.4
RUN pip install tensorboardX==2.4
RUN pip install timm==0.4.12
RUN pip install thop
RUN pip install h5py

# Downgrading protobuf
RUN pip install protobuf==3.20.*

USER user
WORKDIR /scratch/OSwinMRI/Code/


