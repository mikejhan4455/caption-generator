# Captiong Generator

An Image caption generator using Deep Learning.

## Getting Started

Good habit

```sh
$ sudo apt-get update
$ sudo apt-get upgrade -y
```

```sh
$ sudo apt-get install linux-headers-$(uname -r)
```



## Prerequisites

### Python3

Install python3 and pip3

```shell
$ sudo apt-get install python3
$ sudo apt-get install python3-pip
```

### Anaconda

```shell
$ wget https://repo.continuum.io/archive/Anaconda3-2018.12-Linux-x86_64.sh
$ chmod +x Anaconda3-2018.12-Linux-x86_64.sh```
$ ./Anaconda3-2018.12-Linux-x86_64.sh
$ source ~/.bashrc
$ conda install python=3.6
```

after anaconda installation,  Install python libraries in conda virtural environment

```shell
$ conda create -n image-caption python=3 anaconda
$ conda activate image-caption
(image-caption)$ conda install keras tensorflow-gpu==1.9.0 pydot Pillow matplotlib
```

- Notice 1: Make sure which `pip` are you using
- Notice 2: Be sure that your Tensorflow version is compatible with your Cuda one
- Notice 3: Better use `conda install` to avoid package dependence problem

Install GraphViz to use plot_model()
```sh
$ sudo apt-get install graphviz -y
```

### Cuda 9.0 & CuDnn

Install cuda drive, Toolkit, examples (Optional)

```sh
$ ./cuda_9.0.176_384.81_linux-run
```

Install CuDnn

Register an nvidia developer account and [download cudnn here](https://developer.nvidia.com/cudnn)

```sh
$ sudo dpkg -i libcudnn7_7.4.1.5-1+cuda9.0_amd64.deb
```



### Datasets

- [Flick8K dataset (summit to get download link)](https://forms.illinois.edu/sec/1713398)
- [MS COCO dataset (2015)](http://cocodataset.org/#download)





