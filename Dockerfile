FROM continuumio/miniconda3

RUN apt-get update
RUN apt-get update && apt-get upgrade -y && apt-get install gcc --yes
RUN apt-get update && apt-get install -y git
RUN apt-get update && apt-get install vim -y

WORKDIR /
RUN git clone -b ac_docker2 https://github.com/SchubertLab/epytope.git
RUN apt-get -y update && apt-get install unzip
RUN conda env create -f /epytope/environment.yml
SHELL ["/bin/bash", "-c"]

COPY files /files
WORKDIR /
RUN git clone https://github.com/pengxingang/TEIM.git
#epiTCR - works
WORKDIR /
RUN git clone https://github.com/ddiem-ri-4D/epiTCR.git
RUN unzip /epiTCR/models/rdforestWithoutMHCModel.pickle.zip -d /epiTCR/models
RUN unzip /epiTCR/models/rdforestWithMHCModel.pickle.zip -d /epiTCR/models
RUN unzip /epiTCR/data/hlaCovertPeudoSeq/HLAWithPseudoSeq.csv.zip -d /epiTCR/data/hlaCovertPeudoSeq
#AttnTAP - works
WORKDIR /
RUN git clone https://github.com/Bioinformatics7181/AttnTAP.git
#TULIP-TCR - works
WORKDIR /
RUN git clone https://github.com/barthelemymp/TULIP-TCR.git
#TEINet - needs GPU
WORKDIR /
RUN git clone https://github.com/jiangdada1221/TEINet.git
WORKDIR /TEINet
RUN mkdir models
WORKDIR /TEINet/models
RUN pip install gdown && gdown https://drive.google.com/uc?id=12pVozHhRcGyMBgMlhcjgcclE3wlrVO32
#ERGO - works
WORKDIR /
RUN git clone https://github.com/louzounlab/ERGO.git
#ERGO2 - works
WORKDIR /
RUN git clone https://github.com/IdoSpringer/ERGO-II.git
#bertrand - works
WORKDIR /
RUN git clone https://github.com/SFGLab/bertrand.git
#ImRex - works
WORKDIR /
RUN git clone https://github.com/pmoris/ImRex.git
RUN conda env create -f /ImRex/environment.yml
#TITAN
WORKDIR /
RUN git clone https://github.com/PaccMann/TITAN.git
WORKDIR /TITAN/datasets
RUN mkdir imgt
WORKDIR /TITAN/datasets/imgt
RUN wget -O J_segment_sequences.fasta https://www.imgt.org/download/V-QUEST/IMGT_V-QUEST_reference_directory/Homo_sapiens/TR/TRBJ.fasta
WORKDIR /TITAN/datasets/imgt
RUN wget -O V_segment_sequences.fasta https://www.imgt.org/download/V-QUEST/IMGT_V-QUEST_reference_directory/Homo_sapiens/TR/TRBV.fasta
#ATM-TCR - works
WORKDIR /
RUN git clone https://github.com/Lee-CBG/ATM-TCR.git
#pMTnet - works
WORKDIR /
RUN git clone https://github.com/tianshilu/pMTnet.git
#PanPep
WORKDIR /
RUN git clone https://github.com/bm2-lab/PanPep.git
#iTCep - works
WORKDIR /
RUN git clone https://github.com/kbvstmd/iTCep.git
#DLpTCR
WORKDIR /
RUN git clone https://github.com/JiangBioLab/DLpTCR
#STAPLER
WORKDIR /
RUN git clone https://github.com/NKI-AI/STAPLER.git
WORKDIR /STAPLER
RUN mkdir model
WORKDIR /STAPLER/model
RUN mkdir finetuned_model_refactored
WORKDIR /STAPLER/model/finetuned_model_refactored
RUN wget https://files.aiforoncology.nl/stapler/model/finetuned_model_refactored/train_checkpoint_epoch-50-loss-0.000-val-ap0.461_refactored.ckpt
RUN wget https://files.aiforoncology.nl/stapler/model/finetuned_model_refactored/train_checkpoint_epoch-53-loss-0.000-val-ap0.504_refactored.ckpt
RUN wget https://files.aiforoncology.nl/stapler/model/finetuned_model_refactored/train_checkpoint_epoch-68-loss-0.000-val-ap0.477_refactored.ckpt
RUN wget https://files.aiforoncology.nl/stapler/model/finetuned_model_refactored/train_checkpoint_epoch-83-loss-0.000-val-ap0.526_refactored.ckpt
RUN wget https://files.aiforoncology.nl/stapler/model/finetuned_model_refactored/train_checkpoint_epoch-97-loss-0.000-val-ap0.503_refactored.ckpt
WORKDIR /STAPLER/model
RUN mkdir pretrained_model
WORKDIR /STAPLER/model/pretrained_model
RUN wget https://files.aiforoncology.nl/stapler/model/pretrained_model/pre-cdr3_combined_epoch%3D437-train_mlm_loss%3D0.702.ckpt
#tcellmatch
WORKDIR /
RUN git clone https://github.com/theislab/tcellmatch.git
WORKDIR /tcellmatch
RUN wget -O models.zip https://figshare.com/ndownloader/files/43082557
RUN unzip models.zip
RUN mv msb199416-sup-0005-DatasetEV4 models
#bertrand
WORKDIR /bertrand
RUN mkdir /models
WORKDIR /
RUN pip install gdown && gdown https://drive.google.com/uc?id=1FywbDbzhhYbwf99MdZrpYQEbXmwX9Zxm
RUN unzip /bertrand-checkpoint.zip -d /bertrand/models
#envs
RUN conda env create -f /epytope/ergo.yml
RUN conda env create -f /epytope/atm.yml
RUN conda env create -f /epytope/tcellmatch.yml
RUN conda env create -f /epytope/titan.yml
RUN conda create --name epytope_stapler python=3.9
WORKDIR /tcellmatch
SHELL ["conda", "run", "-n", "epytope_numpy195", "/bin/bash", "-c"]
RUN pip install -e .
SHELL ["conda", "run", "-n", "epytope_torch11", "/bin/bash", "-c"]
RUN conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
WORKDIR /STAPLER
SHELL ["conda", "run", "-n", "epytope_stapler", "/bin/bash", "-c"]
RUN pip install stitchr
RUN pip install IMGTgeneDL
RUN pip install -e .
RUN pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16
RUN stitchrdl -s human
RUN pip install x-transformers==0.22.3
WORKDIR /TITAN
SHELL ["conda", "run", "-n", "epytope_torch11_1", "/bin/bash", "-c"]
RUN pip install -e .
SHELL ["conda", "run", "-n", "deepTCR", "/bin/bash", "-c"]
RUN pip install xlrd==1.2.0 && pip install keras==2.3.1
WORKDIR /epytope
SHELL ["conda", "run", "-n", "epytope", "/bin/bash", "-c"]
RUN pip install -e .
EXPOSE 8001
#ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "epytope"]
ENTRYPOINT ["tail", "-f", "/dev/null"]
