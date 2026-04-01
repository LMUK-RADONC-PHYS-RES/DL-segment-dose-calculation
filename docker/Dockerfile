FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common ca-certificates curl git build-essential cmake pkg-config \
    libssl-dev libffi-dev libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 ffmpeg \
    libhdf5-dev \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv \
 && python3.11 -m venv /opt/venv \
 && /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0

RUN pip install \
    numpy==1.26.4 scipy pandas matplotlib pillow opencv-python-headless h5py seaborn \
    tensorboard traitlets tornado pyzmq ipython grpcio protobuf nibabel jsonschema \
    requests networkx packaging filelock kiwisolver markupsafe fonttools pyparsing \
    contourpy cycler pooch pytz pyyaml wcwidth werkzeug zipp tqdm sympy tabulate

RUN pip install \
    cupy-cuda12x==13.6.0 \
    absl-py==2.1.0 antlr4-python3-runtime==4.9.3 asttokens==2.4.1 \
    awkward==2.6.7 awkward-cpp==37 cfgv==3.4.0 comm==0.2.2 cramjam==2.8.3 \
    dacite==1.8.1 debugpy==1.8.1 decorator==5.1.1 distlib==0.3.8 execnet==2.1.1 \
    executing==2.0.1 fastrlock==0.8.3 fsspec==2024.6.1 identify==2.5.36 \
    importlib-metadata==8.3.0 importlib-resources==6.5.2 iniconfig==2.0.0 \
    interpolation==2.2.7 ipykernel==6.29.4 \
    itk==5.4.0 itk-core==5.4.0 itk-filtering==5.4.0 itk-io==5.4.0 \
    itk-numerics==5.4.0 itk-registration==5.4.0 itk-segmentation==5.4.0 \
    jedi==0.19.1 jupyter-client==8.6.2 jupyter-core==5.7.2 \
    lightning-utilities==0.11.2 llvmlite==0.43.0 markdown==3.6 matplotlib-inline==0.1.7 \
    monai==1.2.0 nest-asyncio==1.6.0 nodeenv==1.9.1 numba==0.60.0 omegaconf==2.3.0 \
    parso==0.8.4 pexpect==4.9.0 platformdirs==4.2.2 pluggy==1.5.0 pre-commit==3.7.1 \
    prompt-toolkit==3.0.47 ptyprocess==0.7.0 pure-eval==0.2.2 pydicom==3.0.1 \
    pygments==2.18.0 pymedphys==0.40.0 pytest==8.2.2 pytest-xdist==3.6.1 \
    python-dateutil==2.9.0.post0 simpleitk==2.3.1 six==1.16.0 stack-data==0.6.3 \
    tensorboard-data-server==0.7.2 tomlkit==0.12.5 torchmetrics==1.4.0.post0 \
    tzdata==2024.1 uproot==5.3.10 virtualenv==20.26.2 xlstm==1.0.3 causal-conv1d>=1.4.0 \
    mamba-ssm==2.2.5