# OSwinMRI

## Overview

This is the github repository for the implementation of the OSwinMRI cascaded reconstruction network as described in the paper:

[**Multi-channel MRI reconstruction using cascaded Swinμ transformers with overlapped attention**](https://doi.org/10.1088/1361-6560/adb933)




## Installation and Usage:

*OSwinMRI is not intened for clinical use.*

The simplest method of using this code to perform inference on the Calgary Campinas and M4Raw datasets using the trained models
is by building a Docker image using the provided Dockerfile and running the code within a container.

The instructions below are for an Ubuntu cli with docker installed.


### Step 1:

Clone/download the code from this repository.

### Step 2:

Download the model checkpoints from Zenodo [Link](https://zenodo.org/records/15319907) and extract them to the `trained_models` directory.

### Step 3:

`cd` to the code directory and build the docker image using:

```
docker build -t oswinu --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) .

```

### Step 4:

Download the Calgary Campinas and M4Raw multi-coil datasets.

Calgary Campinas: https://www.ccdataset.com/download

M4Raw: https://doi.org/10.5281/zenodo.8056074



### Step 5:

Run the docker container using:

```

docker run --shm-size=32g --gpus all -it -p 8888:8888 \
-v /path_to_data:/scratch/OSwinMRI/Data \
-v /path_to_code:/scratch/OSwinMRI/Code oswinu:latest bash

```

### Step 6:

Run inference using the provided checkpoints and options files. Output will be saved in the `mri_recon` directory.

```console
# Calgary Campinas Dataset
# R = 5
python main_test_mrrec.py --opt ./options/CC_359/test/test_oswinv2u_r5.json --test_data *path to directory containing Calgary Campinas HDF5 volumes*
# R = 10
python main_test_mrrec.py --opt ./options/CC_359/test/test_oswinv2u_r10.json --test_data *path to directory containing Calgary Campinas HDF5 volumes*

# M4Raw Dataset
# R = 2
python main_test_mrrec.py --opt ./options/M4_Raw/test/test_oswinv2u_r2.json --test_data *path to directory containing M4Raw HDF5 volumes of T1-weighted acquistions*
# R = 3
python main_test_mrrec.py --opt ./options/M4_Raw/test/test_oswinv2u_r3.json --test_data *path to directory containing M4Raw HDF5 volumes of T1-weighted acquistions*

```

## Citation and Acknowledgement:

If you use the code in this repository for your own research, please cite OSwinMRI using the following BiBTeX entry:

```
@article{rahman2025multi,
  title={Multi-channel MRI reconstruction using cascaded Swin$\mu$ transformers with overlapped attention},
  author={Rahman, Tahsin and Bilgin, Ali and Cabrera, Sergio D},
  journal={Physics in Medicine \& Biology},
  volume={70},
  number={7},
  pages={075002},
  year={2025},
  publisher={IOP Publishing}
}
```

This project is released uder the Apache 2.0 license. The implementation is based on the following projects: [SwinMR](https://github.com/ayanglab/SwinMR), [Swin Transformer](https://github.com/microsoft/Swin-Transformer), and [HAT](https://github.com/XPixelGroup/HAT).
