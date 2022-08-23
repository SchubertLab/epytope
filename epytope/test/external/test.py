import unittest
import pandas as pd
from tempfile import NamedTemporaryFile
from epytope.Core import AntigenImmuneReceptor
from epytope.Core import TCREpitope
from epytope.Core import ImmuneReceptorChain
from epytope.TCRSpecificityPrediction import TCRSpecificityPredictorFactory, ML
import os
import scirpy as ir
from epytope.IO.FileReader import process_dataset_TCR


epitope1 = TCREpitope("FLKEKGGL", mhc="HLA-B*08")
epitope2 = TCREpitope("SQLLNAKYL", mhc="HLA-B*08")
epitopes = [epitope1, epitope2]
IEDB = "/home/mahmoud/Documents/Github/GoBi/TCR/epytope/Data/TCR/tcell_receptor_table_export_1660640162.csv"
repository = "/home/mahmoud/Documents/epytope/epytope/epytope/TCRSpecificityPrediction/Models/ERGO-II"
mo = TCRSpecificityPredictorFactory("ergo-ii")
df = ir.datasets.wu2020().obs
# get all TCR seqs in scirpy format
df = process_dataset_TCR(df=df, source="scirpy")
df = df[['TRA', 'TRB', "TRAV", "TRAJ", "TRBV", "TRBJ", "T-Cell-Type", "Species", "Antigen.species", "Tissue"]]
df2 = pd.DataFrame({"Peptide": [str(pep) for pep in epitopes],
                    "MHC": [pep.mhc for pep in epitopes]})
# map each TCR seq to each epitope in the epitopes list
df = pd.merge(df, df2, how='cross')
df = df[["Receptor_ID", 'TRA', 'TRB', "TRAV", "TRAJ", "TRBV", "TRBJ", "T-Cell-Type", "Peptide", "MHC", "Species",
         "Antigen.species", "Tissue"]]
print("Testing scirpy\n")
mo.predict_from_dataset(repository= "/home/mahmoud/Documents/epytope/epytope/epytope/TCRSpecificityPrediction/Models/ERGO-II",
                        df=df)