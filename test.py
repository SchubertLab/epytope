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

path_data = '/files/McPAS-TCR.csv'
tcr_repertoire = IRDatasetAdapterFactory("mcpas-tcr")

tcr_repertoire.from_path(path_data)
tcr_repertoire.receptors = tcr_repertoire.receptors[:100]

predictor = TCRSpecificityPredictorFactory("teim")
results = predictor.predict(tcr_repertoire, [epitope_1] * len(tcr_repertoire.receptors), repository="/TEIM", pairwise=False)
print(results)
predictor = TCRSpecificityPredictorFactory("epitcr")
results = predictor.predict(tcr_repertoire, [epitope_1] * len(tcr_repertoire.receptors), repository="/epiTCR", pairwise=False)
print(results)
predictor = TCRSpecificityPredictorFactory("attntap")
results = predictor.predict(tcr_repertoire, [epitope_1] * len(tcr_repertoire.receptors), repository="/AttnTAP", pairwise=False)
print(results)
#predictor = TCRSpecificityPredictorFactory("teinet")
#results = predictor.predict(tcr_repertoire, [epitope_1] * len(tcr_repertoire.receptors), repository="/TEINet", model="/teinet_data.pth", pairwise=False)
#print(results)
predictor = TCRSpecificityPredictorFactory("tulip-tcr")
results = predictor.predict(tcr_repertoire, [epitope_1] * len(tcr_repertoire.receptors), repository="/TULIP-TCR", processor="cpu", pairwise=False)
print(results)
predictor = TCRSpecificityPredictorFactory("bertrand")
results = predictor.predict(tcr_repertoire, [epitope_1] * len(tcr_repertoire.receptors), repository="/bertrand", pairwise=False)
print(results)
predictor = TCRSpecificityPredictorFactory("ERGO-I")
results = predictor.predict(tcr_repertoire, [epitope_1] * len(tcr_repertoire.receptors), repository="/ERGO", conda="ergo", cuda="cpu", pairwise=False)
print(results)
predictor = TCRSpecificityPredictorFactory("ERGO-II")
results = predictor.predict(tcr_repertoire, [epitope_1] * len(tcr_repertoire.receptors), repository="/ERGO-II", conda="ergo", dataset="mcpas", pairwise=False)
print(results)
predictor = TCRSpecificityPredictorFactory("imrex")
results = predictor.predict(tcr_repertoire, [epitope_1] * len(tcr_repertoire.receptors), repository="/ImRex", conda="deepTCR", pairwise=False)
print(results)
predictor = TCRSpecificityPredictorFactory("atm-tcr")
results = predictor.predict(tcr_repertoire, [epitope_1] * len(tcr_repertoire.receptors), repository="/ATM-TCR", conda="atm", pairwise=False)
print(results)
predictor = TCRSpecificityPredictorFactory("pmtnet")
results = predictor.predict(tcr_repertoire, [epitope_1] * len(tcr_repertoire.receptors), repository="/pMTnet", conda="ergo", pairwise=False)
print(results)
#predictor = TCRSpecificityPredictorFactory("panpep") cuda needed
#results = predictor.predict(tcr_repertoire, [epitope_1] * len(tcr_repertoire.receptors), repository="/PanPep", conda="atm", pairwise=False)
#print(results)
#predictor = TCRSpecificityPredictorFactory("titan")
#results = predictor.predict(tcr_repertoire, [epitope_1] * len(tcr_repertoire.receptors), repository="/TITAN", conda="ergo", pairwise=False)
#print(results)
predictor = TCRSpecificityPredictorFactory("itcep")
results = predictor.predict(tcr_repertoire, [epitope_1] * len(tcr_repertoire.receptors), repository="/iTCep", conda="ergo", pairwise=False)
print(results)
predictor = TCRSpecificityPredictorFactory("stapler")
results = predictor.predict(tcr_repertoire, [epitope_1] * len(tcr_repertoire.receptors), repository="/STAPLER", conda="ergo", pairwise=False)
print(results)
