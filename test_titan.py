import sys
sys.path.append('/epytope')

from epytope.Core import Peptide, Allele
from epytope.Core import Peptide, Allele, TCREpitope, ImmuneReceptorChain, ImmuneReceptor
from epytope.IO import IRDatasetAdapterFactory
from epytope.TCRSpecificityPrediction import TCRSpecificityPredictorFactory
import pandas as pd


peptide = Peptide("SYFPEITHI")
allele = Allele("HLA-A*02:01")
epitope_1 = TCREpitope(peptide=peptide, allele=allele)

epitope_2 = TCREpitope(peptide="EAAGIGILT", allele=None)

path_data = '/files/vdjbd.tsv'
tcr_repertoire = IRDatasetAdapterFactory("vdjdb")

tcr_repertoire.from_path(path_data)
tcr_repertoire.receptors = tcr_repertoire.receptors[:100]



predictor = TCRSpecificityPredictorFactory("titan")
results = predictor.predict(tcr_repertoire, [epitope_1] * len(tcr_repertoire.receptors), repository="/TITAN", conda="epytope_torch11", pairwise=False)
print(results)
