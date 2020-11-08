FROM tensorflow/tensorflow:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
build-essential \
libssl-dev \
libsm6 \
libxext6 \
libxrender-dev \
curl \
unzip \
git \
openssl

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl -OL https://github.com/Kitware/CMake/releases/download/v3.18.4/cmake-3.18.4.tar.gz && \
tar -zxvf cmake-3.18.4.tar.gz && \
cd cmake-3.18.4 && \
./bootstrap && \
make && \
make install

RUN /usr/local/bin/pip install -U pip
RUN /usr/local/bin/pip --no-cache-dir install \
wheel \
Pillow \
matplotlib \
numpy \
pandas \
scipy \
sklearn \
tqdm \
argparse \
boto3 \
mtcnn \
Cython \
contextlib2 \
lxml \
jupyter \
jupyterlab \
easydict \
seaborn \
pyarrow \
imgaug \
python-shogi \
tensorflow-addons \
tensorflow-datasets

RUN /usr/local/bin/pip --no-cache-dir install \
kaggle \
opencv-python
