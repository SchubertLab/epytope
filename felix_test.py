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
epitope_2 = TCREpitope(peptide="EAAGIGILTV", allele=allele)

path_data = '../prak/data/vdjdb.tsv'
tcr_repertoire = IRDatasetAdapterFactory("vdjdb")
tcr_repertoire.from_path(path_data)
tcr_repertoire.receptors = tcr_repertoire.receptors[:20]

for name, version in TCRSpecificityPredictorFactory.available_methods().items():
    print(name, ",".join(version))

reqs = {
    "ergo-i": {"repository": "../ERGO"},
}

choices = {
    "ergo-i": {"dataset": ["vdjdb", "mcpas"]}
}

epitopes_pairwise = [epitope_1, epitope_2]

for name, req_model in reqs.items():
    print(name)
    
    predictor = TCRSpecificityPredictorFactory(name)
    results = predictor.predict(tcr_repertoire, epitopes_pairwise, pairwise=True, **req_model)
    
    assert len(results)== len(tcr_repertoire.receptors), "Results have wrong length"
    for epitope in epitopes_pairwise:
        assert epitope in results, "Epitope not in result"
        assert name in [el.lower() for el in results[epitope].columns.tolist()], "Method not in results"
        assert results[epitope].iloc[:, 0].isna().sum() < len(results), "Method always yield NaN"
        
print("### All pairwise Tests succeded")

epitopes_list = [epitope_1, epitope_2] * 10

for name, req_model in reqs.items(): 
    print(name)
    predictor = TCRSpecificityPredictorFactory(name)
    results = predictor.predict(tcr_repertoire, epitopes_list, pairwise=False, **req_model)
    
    assert len(results)== len(tcr_repertoire.receptors), "Results have wrong length"
    assert name in [el.lower() for el in results["Method"].columns]
    assert results["Method"].iloc[:, 0].isna().sum() < len(results), "Method always yield NaN"
    for i, epitope in enumerate(epitopes_list):
        assert results.at[i, ("Epitope", "Peptide")] == epitope.peptide, f"Wrong epitope at position {i}"
        assert results.at[i, ("Epitope", "MHC")] == epitope.allele, f"Wrong MHC at position {i}"
    
print("### All non-pairwise Tests succeded")