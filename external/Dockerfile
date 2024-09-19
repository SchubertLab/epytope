FROM continuumio/miniconda3

RUN apt-get update
RUN apt-get update && apt-get upgrade -y && apt-get install gcc --yes
RUN apt-get update && apt-get install -y git
RUN apt-get update && apt-get install vim -y
RUN apt-get -y update && apt-get install unzip

WORKDIR /
RUN git clone -b ac_docker https://github.com/SchubertLab/epytope.git
SHELL ["/bin/bash", "-c"]

WORKDIR /
RUN /bin/sh -c "/epytope/external/download.sh"
#TITAN
#WORKDIR /
#RUN git clone https://github.com/PaccMann/TITAN.git
#WORKDIR /TITAN/datasets
#RUN mkdir imgt
#WORKDIR /TITAN/datasets/imgt
#RUN wget -O J_segment_sequences.fasta https://www.imgt.org/download/V-QUEST/IMGT_V-QUEST_reference_directory/Homo_sapiens/TR/TRBJ.fasta
#WORKDIR /TITAN/datasets/imgt
#RUN wget -O V_segment_sequences.fasta https://www.imgt.org/download/V-QUEST/IMGT_V-QUEST_reference_directory/Homo_sapiens/TR/TRBV.fasta
#envs
RUN conda env create -f /epytope/external/environment.yml
RUN conda env create -f /ImRex/environment.yml
RUN conda env create -f /epytope/external/ergo.yml
RUN conda env create -f /epytope/external/atm.yml
RUN conda env create -f /epytope/external/tcellmatch.yml
RUN conda env create -f /epytope/external/titan.yml
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
WORKDIR /ANARCI
SHELL ["conda", "run", "-n", "epytope_stapler", "/bin/bash", "-c"]
RUN python setup.py install
WORKDIR /TITAN
SHELL ["conda", "run", "-n", "epytope_torch11_1", "/bin/bash", "-c"]
RUN pip install -e .
WORKDIR /epytope
SHELL ["conda", "run", "-n", "epytope", "/bin/bash", "-c"]
RUN pip install -e .
EXPOSE 8001
ENTRYPOINT ["tail", "-f", "/dev/null"]
