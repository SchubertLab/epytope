import sys
sys.path.append('/home/icb/anna.chernysheva/epytope')

from epytope.Core import Peptide, Allele
from epytope.Core import Peptide, Allele, TCREpitope, ImmuneReceptorChain, ImmuneReceptor
from epytope.IO import IRDatasetAdapterFactory
from epytope.TCRSpecificityPrediction import TCRSpecificityPredictorFactory
import pandas as pd


peptide = Peptide("SYFPEITHI")
allele = Allele("HLA-A*02:01")
epitope_1 = TCREpitope(peptide=peptide, allele=allele)

epitope_2 = TCREpitope(peptide="EAAGIGILTV", allele=None)

path_data = '/home/icb/anna.chernysheva/McPAS-TCR.csv'
tcr_repertoire = IRDatasetAdapterFactory("mcpas-tcr")

tcr_repertoire.from_path(path_data)

predictor = TCRSpecificityPredictorFactory("ergo-I")
results = predictor.predict(tcr_repertoire, [epitope_1] * len(tcr_repertoire.receptors), repository="/home/icb/anna.chernysheva/ERGO", conda="ergo", pairwise=False)
print(results[:20, 4:])

predictor = TCRSpecificityPredictorFactory("ergo-I")
results = predictor.predict(tcr_repertoire, [epitope_1, epitope_2], repository="/home/icb/anna.chernysheva/ERGO", conda="ergo", pairwise=True)
print(results[:20, 4:])
