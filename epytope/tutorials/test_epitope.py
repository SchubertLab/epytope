import sys
import os
sys.path.append('../..')

from epytope.Core import Peptide, Allele
from epytope.Core import Peptide, Allele, TCREpitope, ImmuneReceptorChain, ImmuneReceptor
from epytope.IO import IRDatasetAdapterFactory
from epytope.TCRSpecificityPrediction import TCRSpecificityPredictorFactory
import pandas as pd

os.system('export LD_LIBRARY_PATH=/home/icb/anna.chernysheva/miniconda3/lib:$LD_LIBRARY_PATH')

peptide = Peptide("SYFPEITHI")
allele = Allele("HLA-A*02:01")
epitope_1 = TCREpitope(peptide=peptide, allele=allele)

epitope_2 = TCREpitope(peptide="EAAGIGILTV", allele=None)

path_data = '../../../McPAS-TCR.csv'
tcr_repertoire = IRDatasetAdapterFactory("mcpas-tcr")

tcr_repertoire.from_path(path_data)

predictor = TCRSpecificityPredictorFactory("panpep")
results = predictor.predict(tcr_repertoire, [epitope_1] * len(tcr_repertoire.receptors), repository="/home/icb/anna.chernysheva/PanPep", conda="panpen", pairwise=False)
results.to_csv("out_single.csv")

predictor = TCRSpecificityPredictorFactory("ergo-I")
results = predictor.predict(tcr_repertoire, [epitope_1, epitope_2], repository="/home/icb/anna.chernysheva/PanPep", conda="panpen", pairwise=True)
results.to_csv("out_pair.csv")
