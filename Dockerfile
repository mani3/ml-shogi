FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && apt-get install -y --no-install-recommends \
libsm6 \
libxext6 \
libxrender-dev \
curl \
unzip \
git \
openssl

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

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

RUN curl -sL https://deb.nodesource.com/setup_12.x | bash
RUN apt-get install -y nodejs
RUN jupyter labextension install jupyterlab_tensorboard


