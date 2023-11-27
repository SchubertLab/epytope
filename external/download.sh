#!/bin/sh

mkdir /external
cd external
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
