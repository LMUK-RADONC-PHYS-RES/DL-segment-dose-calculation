# Multi-model Study of Fast VMAT Segment Dose Calculation with Deep Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](docker/)

This repository contains the source code of models/preprocessing, model weights, and Dockerfile for the paper **"Multi-model Study of Fast VMAT Segment Dose Calculation with Deep Learning"**.

## Repository Contents
* **`docker/`**: Dockerfile for setting up a reproducible environment.
* **`preprocessing/`**: Scripts for BEV cuboid/patient coordinate (four physical inputs) processing.
* **`model/`**: Source code and weights for the CNN-ConvLSTM, CNN-Mamba, DoTA(pytorch), C3D, DeepDose-C3D.
* **`train/`**: Shared training utils (`utils.py`) and per-model entry points (`CNN_ConvLSTM/train.py`, `CNN_Mamba/train.py`).
* **`inference/`**: Shared inference pipeline (`pipeline.py`) and per-model entry points (`CNN_ConvLSTM/inference.py`, `CNN_Mamba/inference.py`).


## Model Architectures

![Model architectures](example_figs/models.png)

## Inference

```bash
# CNN-ConvLSTM
python inference/CNN_ConvLSTM/inference.py \
    --patient PATIENT_ID \
    --sim-root /path/to/simulation \
    --seg-dir /path/to/segments \
    --ct-root /path/to/ct \
    --model-weights /path/to/weights.pth

# CNN-Mamba
python inference/CNN_Mamba/inference.py \
    --patient PATIENT_ID \
    --sim-root /path/to/simulation \
    --seg-dir /path/to/segments \
    --ct-root /path/to/ct \
    --model-weights /path/to/weights.pth
```

## DL dose calculation challenge

An associated open challenge is available at the [DoseRad 2026 Grand Challenge](https://doserad2026.grand-challenge.org/).
