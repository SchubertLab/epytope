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
conda run -n epytope conda install -c conda-forge ncurses -y
conda run -n epytope conda install libgcc==7.2.0
conda run -n epytope pip install -e .
cd /benchmark_TCRprediction
conda run -n epytope pip install -e .
cd /external/TITAN
conda run -n epytope_torch11_1 pip install -e .
conda run -n epytope_torch11_1 pip install biopython==1.81 pytoda==1.1.3
cd /external/ANARCI
conda run -n epytope_torch21 pip install stitchr IMGTgeneDL
conda run -n epytope_torch21 conda install -c conda-forge openmm pdbfixer -y
conda run -n epytope_torch21 conda install -c bioconda hmmer=3.3.2 -y
conda run -n epytope_torch21 pip install muscle
conda run -n epytope_torch21 python setup.py install
conda run -n epytope_torch21 stitchrdl -s human
conda run -n epytope_torch21 pip install numpy==1.24.1
