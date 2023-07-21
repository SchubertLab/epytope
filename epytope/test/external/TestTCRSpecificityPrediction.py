__author__ = 'albahah'


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


class TestTCRSpecificityPredictionClass(unittest.TestCase):

    def setUp(self):
        TRA1 = ImmuneReceptorChain(chain_type="TRA", v_gene="TRAV12-1*01", d_gene="", j_gene="TRAJ23*01",
                                   cdr3="VVRAGKLI")
        TRB1 = ImmuneReceptorChain(chain_type="TRB", v_gene="TRBV6-3*01", d_gene="", j_gene="TRBJ2-4*01",
                                   cdr3="ASGQGNFDIQY")
        TRA2 = ImmuneReceptorChain(chain_type="TRA", v_gene="TRAV9-2*01", d_gene="", j_gene="TRAJ43*01",
                                   cdr3="ALSDPVNDMR")
        TRB2 = ImmuneReceptorChain(chain_type="TRB", v_gene="TRBV11-2*01", d_gene="", j_gene="TRBJ1-5*01",
                                   cdr3="ASSLRGRGDQPQH")
        epitope1 = TCREpitope("FLRGRAYGL", mhc="HLA-B*08:01")
        epitope2 = TCREpitope("HSKRKCDEL", mhc="HLA-B*08:01")
        TCR1 = AntigenImmuneReceptor(receptor_id="1", chains=[TRA1, TRB1], cell_type="CD8")
        TCR2 = AntigenImmuneReceptor(receptor_id="2", chains=[TRA2, TRB2], cell_type="CD8")
        self.TCRs = [TCR1, TCR2]
        self.epitopes = [epitope1, epitope2]
        self.peptide = []
        self.dataset = pd.DataFrame({"Receptor_ID": 1, "TRA": "CAVSAASGGSYIPTF", "TRB": "CASSFSGNTGELFF", "TRAV": "TRAV3", "TRAJ": "TRAJ6",
                                "TRBV": "TRBV12-3", "TRBJ": "TRBJ2-2", "T-Cell-Type": "CD8", "Peptide": "RAKFKQLL",
                                "MHC": "HLA-B*08", "Species": "", "Antigen.species": "", "Tissue": ""}, index=[0])
        self.TCR = ""
        self.vdjdb = "/home/mahmoud/Downloads/vdjdb/vdjdb_full.txt"
        self.McPAS = "/home/mahmoud/Downloads/McPAS-TCR.csv"
        self.IEDB = "/home/mahmoud/Downloads/tcell_receptor_table_export_1660640162.csv"
        self.repository = {"ERGO-II": "/home/mahmoud/Documents/BA/ERGOII/ERGO-II",
              "TITAN": "/home/mahmoud/Documents/BA/TITAN/TITAN",
              "ImRex": "/home/mahmoud/Documents/BA/IMRex/ImRex",
              "NetTCR2": "/home/mahmoud/Documents/BA/test/NetTCR-2.0",
              "pMTnet": "/home/mahmoud/Documents/BA/test/pMTnet",
              "ATM_TCR": "/home/mahmoud/Documents/BA/test/ATM-TCR"}
        self.pMTnet_interpreter = "/home/mahmoud/anaconda3/envs/pmtnet/bin/python"

    def test_TCR_specificity_prediction_multiple_input(self):
        for m in TCRSpecificityPredictorFactory.available_methods():
            mo = TCRSpecificityPredictorFactory(m)
            print("\nTesting", mo.name)
            print("Test binding specificity for each TCR to each epitope")
            mo.predict(peptides=self.epitopes, TCRs=self.TCRs, repository=self.repository[mo.name], all=True,
                       trained_on="vdjdb", trained_model="/home/mahmoud/Documents/BA/TITAN/TITAN/public/trained_model",
                       nettcr_chain="ab", pMTnet_interpreter=self.pMTnet_interpreter)
            print("Test binding specificity for TCRs to the corresponding epitopes in the same passed order\n")
            mo.predict(peptides=self.epitopes, TCRs=self.TCRs, repository=self.repository[mo.name], all=False,
                       trained_on="vdjdb", trained_model="/home/mahmoud/Documents/BA/TITAN/TITAN/public/trained_model",
                       nettcr_chain="b", pMTnet_interpreter=self.pMTnet_interpreter)

    def test_TCR_specificity_prediction_single_input(self):
        for m in TCRSpecificityPredictorFactory.available_methods():
            mo = TCRSpecificityPredictorFactory(m)
            print("\nTesting", mo.name)
            mo.predict(peptides=self.epitopes[0], TCRs=self.TCRs[0], repository=self.repository[mo.name], all=False,
                       trained_on="vdjdb", trained_model="/home/mahmoud/Documents/BA/TITAN/TITAN/public/trained_model",
                       nettcr_chain="ab", pMTnet_interpreter=self.pMTnet_interpreter)

    def test_TCR_specificity_prediction_dataset(self):
        tmp_file = NamedTemporaryFile(delete=False)
        self.dataset.to_csv(tmp_file.name, sep=",", index=False)
        for m in TCRSpecificityPredictorFactory.available_methods():
            mo = TCRSpecificityPredictorFactory(m)
            print("\nTesting", mo.name)
            mo.predict_from_dataset(repository=self.repository[mo.name], path=tmp_file.name,
                                    trained_model="/home/mahmoud/Documents/BA/TITAN/TITAN/public/trained_model",
                                    nettcr_chain="ab", pMTnet_interpreter=self.pMTnet_interpreter)
            print("Testing on vdjdb")
            mo.predict_from_dataset(repository=self.repository[mo.name], path=self.vdjdb, source="vdjdb",
                                    trained_model="/home/mahmoud/Documents/BA/TITAN/TITAN/public/trained_model",
                                    nettcr_chain="b", pMTnet_interpreter=self.pMTnet_interpreter)
            print("Testing on McPAS")
            mo.predict_from_dataset(repository=self.repository[mo.name], path=self.McPAS, source="mcpas", trained_on="McPAS",
                                    trained_model="/home/mahmoud/Documents/BA/TITAN/TITAN/public/trained_model",
                                    nettcr_chain="a", pMTnet_interpreter=self.pMTnet_interpreter)
            print("Testing on IEDB\n")
            mo.predict_from_dataset(repository=self.repository[mo.name], path=self.IEDB, source="IEDB",
                                    trained_model="/home/mahmoud/Documents/BA/TITAN/TITAN/public/trained_model",
                                    nettcr_chain="ab", pMTnet_interpreter=self.pMTnet_interpreter)
        os.remove(tmp_file.name)

    def test_wrong_input(self):
        with self.assertRaises(ValueError):
            mo = TCRSpecificityPredictorFactory("ergo-ii")
            mo.predict(peptides=self.peptide, TCRs=self.TCRs, repository=self.repository[mo.name], all=True,
                       trained_on="vdjdb", trained_model="/home/mahmoud/Documents/BA/TITAN/TITAN/public/trained_model",
                       nettcr_chain="ab", pMTnet_interpreter=self.pMTnet_interpreter)
            mo.predict(peptides=self.epitopes, TCRs=self.TCR, repository=self.repository[mo.name], all=True,
                       trained_on="vdjdb", trained_model="/home/mahmoud/Documents/BA/TITAN/TITAN/public/trained_model",
                       nettcr_chain="ab", pMTnet_interpreter=self.pMTnet_interpreter)

    def test_merging_and_filtering(self):
        mo = TCRSpecificityPredictorFactory("ergo-ii")
        result1 = mo.predict(peptides=self.epitopes[0], TCRs=self.TCRs[0], repository=self.repository[mo.name], all=False,
                             trained_on="vdjdb",
                             trained_model="/home/mahmoud/Documents/BA/TITAN/TITAN/public/trained_model",
                             nettcr_chain="ab", pMTnet_interpreter=self.pMTnet_interpreter)
        result2 = mo.predict(peptides=self.epitopes[1], TCRs=self.TCRs[1], repository=self.repository[mo.name], all=False,
                             trained_on="vdjdb",
                             trained_model="/home/mahmoud/Documents/BA/TITAN/TITAN/public/trained_model",
                             nettcr_chain="ab", pMTnet_interpreter=self.pMTnet_interpreter)
        print("\n\nTest merging")
        print(result1.merge_results(result2))
        print("\n\nTest filtering")
        com = lambda x, y: x > y
        thr = 0.7
        expression = (mo.name, com, thr)
        result = mo.predict(peptides=self.epitopes, TCRs=self.TCRs, repository=self.repository[mo.name], all=False,
                            trained_on="vdjdb",
                            trained_model="/home/mahmoud/Documents/BA/TITAN/TITAN/public/trained_model",
                            nettcr_chain="ab", pMTnet_interpreter=self.pMTnet_interpreter)
        print(f"\nresult before filtering:\n{result}\n\nresult after filtering:\n{result.filter_result(expression)}")

    def test_scirpy_format(self):
        df = ir.datasets.wu2020().obs
        # get all TCR seqs in scirpy format
        df = process_dataset_TCR(df=df, source="scirpy")
        df = df[["Receptor_ID", 'TRA', 'TRB', "TRAV", "TRAJ", "TRBV", "TRBJ", "T-Cell-Type", "Species",
                 "Antigen.species", "Tissue"]]
        df2 = pd.DataFrame({"Peptide": [str(pep) for pep in self.epitopes],
                            "MHC": [pep.mhc for pep in self.epitopes]})
        # map each TCR seq to each epitope in the epitopes list
        df = pd.merge(df, df2, how='cross')
        print("\n\nTesting scirpy")
        outputs = []
        for m in TCRSpecificityPredictorFactory.available_methods():
            mo = TCRSpecificityPredictorFactory(m)
            outputs.append(mo.predict_from_dataset(repository=self.repository[mo.name], df=df,
                                              trained_model="/home/mahmoud/Documents/BA/TITAN/TITAN/public/trained_model",
                                              nettcr_chain="ab", pMTnet_interpreter=self.pMTnet_interpreter))
        print(pd.concat(outputs, axis=1))


if __name__ == '__main__':
    unittest.main()
