#!/bin/bash
# Commands to setup a new virtual environment and install all the necessary packages
pip install --upgrade pip

python -m venv icl
source icl/bin/activate

pip install numpy matplotlib jupyter ipykernel jupyterlab tqdm scikit-image pandas matplotlib seaborn einops torchvision torch==2.1.0 wandb