__author__ = 'schubert'


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
        TRA1 = ImmuneReceptorChain(chain_type="TRA", v_gene="TRAV26-1", d_gene="", j_gene="TRAJ43",
                                 cdr3="CIVRAPGRADMRF")
        TRB1 = ImmuneReceptorChain(chain_type="TRB", v_gene="TRBV13", d_gene="", j_gene="TRBJ1-5",
                                cdr3="CASSYLPGQGDHYSNQPQHF")
        TRA2 = ImmuneReceptorChain(chain_type="TRA", v_gene="TRAV20", d_gene="", j_gene="TRAJ28", cdr3="CAVPSGAGSYQLTF")
        TRB2 = ImmuneReceptorChain(chain_type="TRB", v_gene="TRBV13", d_gene="", j_gene="TRBJ1-5",
                                   cdr3="CASSFEPGQGFYSNQPQHF")
        epitope1 = TCREpitope("FLKEKGGL", mhc="HLA-B*08")
        epitope2 = TCREpitope("SQLLNAKYL", mhc="HLA-B*08")
        TCR1 = AntigenImmuneReceptor(receptor_id="", chains=[TRA1, TRB1], cell_type="CD8")
        TCR2 = AntigenImmuneReceptor(receptor_id="", chains=[TRA2, TRB2], cell_type="CD8")
        self.TCRs = [TCR1, TCR2]
        self.epitopes = [epitope1, epitope2]
        self.peptide = []
        self.dataset = pd.DataFrame({"TRA": "CAVSAASGGSYIPTF", "TRB": "CASSFSGNTGELFF", "TRAV": "TRAV3", "TRAJ": "TRAJ6",
                                "TRBV": "TRBV12-3", "TRBJ": "TRBJ2-2", "T-Cell-Type": "CD8", "Peptide": "RAKFKQLL",
                                "MHC": "HLA-B*08", "Species": "", "Antigen.species": "", "Tissue": ""}, index=[0])
        self.TCR = ""
        self.path = ""
        for root, _, _ in os.walk(os.path.expanduser('~')):
            if root.endswith("/epytope/Data/TCR"):
                self.path = root
                break
        self.vdjdb = os.path.join(self.path, "vdjdb_full.txt")
        self.McPAS = os.path.join(self.path, "McPAS-TCR.csv")
        self.IEDB = os.path.join(self.path, "tcell_receptor_table_export_1660640162.csv")

    def test_TCR_specificity_prediction_multiple_input(self):
        for m in TCRSpecificityPredictorFactory.available_methods():
            mo = TCRSpecificityPredictorFactory(m)
            print("\nTesting", mo.name)
            print("Test binding specificity for each TCR to each epitope")
            mo.predict(peptides=self.epitopes, TCRs=self.TCRs, all=True, trained_on="vdjdb")
            print("Test binding specificity for TCRs to the corresponding epitopes in the same passed order\n")
            mo.predict(peptides=self.epitopes, TCRs=self.TCRs, all=False, trained_on="vdjdb")

    def test_TCR_specificity_prediction_single_input(self):
        for m in TCRSpecificityPredictorFactory.available_methods():
            mo = TCRSpecificityPredictorFactory(m)
            print("\nTesting", mo.name)
            mo.predict(peptides=self.epitopes[0], TCRs=self.TCRs[0], all=False, trained_on="vdjdb")

    def test_TCR_specificity_prediction_dataset(self):
        tmp_file = NamedTemporaryFile(delete=False)
        self.dataset.to_csv(tmp_file.name, sep=",", index=False)
        for m in TCRSpecificityPredictorFactory.available_methods():
            mo = TCRSpecificityPredictorFactory(m)
            print("\nTesting", mo.name)
            mo.predict_from_dataset(path=tmp_file.name)
            print("Testing on vdjdb")
            mo.predict_from_dataset(path=self.vdjdb, source="vdjdb")
            print("Testing on McPAS")
            mo.predict_from_dataset(path=self.McPAS, source="mcpas")
            print("Testing on IEDB\n")
            mo.predict_from_dataset(path=self.IEDB, source="IEDB")
        os.remove(tmp_file.name)

    def test_wrong_input(self):
        with self.assertRaises(ValueError):
            mo = TCRSpecificityPredictorFactory("ergo-ii")
            mo.predict(peptides=self.peptide, TCRs=self.TCRs, all=True, trained_on="vdjdb")
            mo.predict(peptides=self.epitopes, TCRs=self.TCR, all=True, trained_on="vdjdb")

    def test_merging_and_filtering(self):
        mo = TCRSpecificityPredictorFactory("ergo-ii")
        result1 = mo.predict(peptides=self.epitopes[0], TCRs=self.TCRs[0], all=False, trained_on="vdjdb")
        result2 = mo.predict(peptides=self.epitopes[1], TCRs=self.TCRs[1], all=False, trained_on="vdjdb")
        print("\n\nTest merging")
        print(result1.merge_results(result2))
        print("\n\nTest filtering")
        com = lambda x, y: x > y
        thr = 0.7
        expression = (mo.name.upper(), com, thr)
        result = mo.predict(peptides=self.epitopes, TCRs=self.TCRs, all=False, trained_on="vdjdb")
        print(f"\nresult before filtering:\n{result}\n\nresult after filtering:\n{result.filter_result(expression)}")

    def test_scirpy_format(self):
        mo = TCRSpecificityPredictorFactory("ergo-ii")
        df = ir.datasets.wu2020().obs
        # get all TCR seqs in scirpy format
        df = process_dataset_TCR(df=df, source="scirpy")
        df = df[['TRA', 'TRB', "TRAV", "TRAJ", "TRBV", "TRBJ", "T-Cell-Type", "Species", "Antigen.species", "Tissue"]]
        df2 = pd.DataFrame({"Peptide": [str(pep) for pep in self.epitopes],
                            "MHC": [pep.mhc for pep in self.epitopes]})
        # map each TCR seq to each epitope in the epitopes list
        df["key"] = 1
        df2["key"] = 1
        df = df.merge(df2, on="key")[['TRA', 'TRB', "TRAV", "TRAJ", "TRBV", "TRBJ", "T-Cell-Type", "Peptide", "MHC",
                                      "Species", "Antigen.species", "Tissue"]]
        print("\n\nTesting scirpy")
        mo.predict_from_dataset(df=df)


if __name__ == '__main__':
    unittest.main()
