#!/bin/bash
module load python/3.9
rm -rf $STORE/mytf
python -m venv $STORE/mytf
source $STORE/mytf/bin/activate
python3 -m pip install tensorflow[and-cuda]
pip install jupyterlab-nvdashboard
pip install -U tensorboard-plugin-profile
pip install nvidia-pyindex
pip install nvidia-cuda-cupti