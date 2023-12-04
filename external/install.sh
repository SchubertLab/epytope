#!/usr/bin/env bash

cd /external/tcellmatch
conda run -n epytope_numpy195 pip install -e .
conda run -n epytope_torch11 conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda run -n deepTCR conda uninstall tensorflow-gpu -y
conda run -n deepTCR conda install tensorflow==2.1.0
cd /external/STAPLER
conda run -n epytope_stapler pip install stitchr IMGTgeneDL
conda run -n epytope_stapler pip install -e .
conda run -n epytope_stapler pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16
conda run -n epytope_stapler stitchrdl -s human
conda run -n epytope_stapler pip install x-transformers==0.22.3
cd /epytope
conda run -n epytope pip install -e .
