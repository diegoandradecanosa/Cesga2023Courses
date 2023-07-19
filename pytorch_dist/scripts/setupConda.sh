#!/bin/bash
set -x
rm -rf $STORE/mytorchdist
module purge
module load cesga/system miniconda3/22.11
mkdir -p $STORE/mytorchdist
tar -xzf /tmp/mytorchdist.tar.gz -C $STORE/mytorchdist
source activateconda.sh
which python
conda-unpack
python -m pip install 'urllib3==1.26.16' --user





