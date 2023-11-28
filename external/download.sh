#!/usr/bin/env bash

mkdir /external
cd /external
#TEIM
git clone https://github.com/pengxingang/TEIM.git
#epiTCR
git clone https://github.com/ddiem-ri-4D/epiTCR.git
unzip /epiTCR/models/rdforestWithoutMHCModel.pickle.zip -d /external/epiTCR/models
unzip /epiTCR/models/rdforestWithMHCModel.pickle.zip -d /external/epiTCR/models
unzip /epiTCR/data/hlaCovertPeudoSeq/HLAWithPseudoSeq.csv.zip -d /external/epiTCR/data/hlaCovertPeudoSeq
#AttnTAP
git clone https://github.com/Bioinformatics7181/AttnTAP.git
#TULIP-TCR
git clone https://github.com/barthelemymp/TULIP-TCR.git
#TEINet
git clone https://github.com/jiangdada1221/TEINet.git
cd /external/TEINet
mkdir models
cd models
pip install gdown && gdown https://drive.google.com/uc?id=12pVozHhRcGyMBgMlhcjgcclE3wlrVO32
#ERGO
cd external
git clone https://github.com/louzounlab/ERGO.git
#ERGO2
git clone https://github.com/IdoSpringer/ERGO-II.git
#bertrand
git clone https://github.com/SFGLab/bertrand.git
cd /external/bertrand
mkdir models
gdown https://drive.google.com/uc?id=1FywbDbzhhYbwf99MdZrpYQEbXmwX9Zxm
unzip /bertrand-checkpoint.zip -d /bertrand/models
#ATM-TCR
cd /external
git clone https://github.com/Lee-CBG/ATM-TCR.git
#pMTnet - works
git clone https://github.com/tianshilu/pMTnet.git
#PanPep
git clone https://github.com/bm2-lab/PanPep.git
#iTCep
git clone https://github.com/kbvstmd/iTCep.git
#DLpTCR
git clone https://github.com/JiangBioLab/DLpTCR
#STAPLER
git clone https://github.com/NKI-AI/STAPLER.git
cd /external/STAPLER
mkdir model
cd model
mkdir finetuned_model_refactored
cd finetuned_model_refactored
wget https://files.aiforoncology.nl/stapler/model/finetuned_model_refactored/train_checkpoint_epoch-50-loss-0.000-val-ap0.461_refactored.ckpt
wget https://files.aiforoncology.nl/stapler/model/finetuned_model_refactored/train_checkpoint_epoch-53-loss-0.000-val-ap0.504_refactored.ckpt
wget https://files.aiforoncology.nl/stapler/model/finetuned_model_refactored/train_checkpoint_epoch-68-loss-0.000-val-ap0.477_refactored.ckpt
wget https://files.aiforoncology.nl/stapler/model/finetuned_model_refactored/train_checkpoint_epoch-83-loss-0.000-val-ap0.526_refactored.ckpt
wget https://files.aiforoncology.nl/stapler/model/finetuned_model_refactored/train_checkpoint_epoch-97-loss-0.000-val-ap0.503_refactored.ckpt
cd ..
mkdir pretrained_model
cd pretrained_model
wget https://files.aiforoncology.nl/stapler/model/pretrained_model/pre-cdr3_combined_epoch%3D437-train_mlm_loss%3D0.702.ckpt
#tcellmatch
cd /external
git clone https://github.com/theislab/tcellmatch.git
cd tcellmatch
wget -O models.zip https://figshare.com/ndownloader/files/43082557
unzip models.zip
mv msb199416-sup-0005-DatasetEV4 models
#NetTCR-2.2
cd /external
git clone https://github.com/mnielLab/NetTCR-2.2.git
git clone https://github.com/oxpig/ANARCI.git
#ImRex
git clone https://github.com/pmoris/ImRex.git

conda env create -f /epytope/external/environment.yml
conda env create -f /external/ImRex/environment.yml
conda env create -f /epytope/external/ergo.yml
conda env create -f /epytope/external/atm.yml
conda env create -f /epytope/external/tcellmatch.yml
conda env create -f /epytope/external/titan.yml
conda create --name epytope_stapler python=3.9
cd /external/tcellmatch
conda run -n epytope_numpy195 && pip install -e .
