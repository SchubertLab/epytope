__author__ = 'albahah'


import unittest
import pandas as pd
from epytope.Core import AntigenImmuneReceptor
from epytope.Core import ImmuneReceptorChain
from epytope.TCRSimilarityMeasurement import SimilarityTools, TCRSimilarityMeasurementFactory
from epytope.IO.FileReader import process_dataset_TCR


class TestTCRSimilarityMeasurementClass(unittest.TestCase):

    def setUp(self):
        TRA1 = ImmuneReceptorChain(chain_type="TRA", v_gene="TRAV7-3*01", d_gene="", j_gene="TRAJ33*01",
                                 cdr3="CAVSLDSNYQLIW", nuc_seq="tgtgcagtgagcctcgatagcaactatcagttgatctgg")
        TRB1 = ImmuneReceptorChain(chain_type="TRB", v_gene="TRBV13-1", d_gene="", j_gene="TRAJ33*01",
                                cdr3="CASSDFDWGGDAETLYF", nuc_seq="tgtgccagcagtgatttcgactggggaggggatgcagaaacgctgtatttt")
        TRA2 = ImmuneReceptorChain(chain_type="TRA", v_gene="TRAV6D-6*01", d_gene="", j_gene="TRAJ56*01",
                                   cdr3="CALGDRATGGNNKLTF", nuc_seq="tgtgctctgggtgacagggctactggaggcaataataagctgactttt")
        TRB2 = ImmuneReceptorChain(chain_type="TRB", v_gene="TRBV29*01", d_gene="", j_gene="TRBJ1-1*01",
                                   cdr3="CASSPDRGEVFF")
        TRA3 = ImmuneReceptorChain(chain_type="TRA", v_gene="TRAV6D-6", d_gene="", j_gene="TRAJ49*01",
                                   cdr3="CALGSNTGYQNFYF")
        TRB3 = ImmuneReceptorChain(chain_type="TRB", v_gene="TRBV29", d_gene="", j_gene="TRBJ1-5*01",
                                   cdr3="CASTGGGAPLF")
        TRA4 = ImmuneReceptorChain(chain_type="TRA", v_gene="TRAV6-4*01", d_gene="", j_gene="TRAJ34*02",
                                   cdr3="CALAPSNTNKVVF")
        TRB4 = ImmuneReceptorChain(chain_type="TRB", v_gene="TRBV2*01", d_gene="", j_gene="TRBJ2-7*01",
                                   cdr3="CASSQDPGDYEQYF")
        TCR1 = AntigenImmuneReceptor(receptor_id="1", chains=[TRA1, TRB1], species="mouse")
        TCR2 = AntigenImmuneReceptor(receptor_id="2", chains=[TRA2, TRB2], species="mouse")
        TCR3 = AntigenImmuneReceptor(receptor_id="3", chains=[TRA3, TRB3], species="mouse")
        TCR4 = AntigenImmuneReceptor(receptor_id="4", chains=[TRA4, TRB4], species="mouse")
        TCR5 = AntigenImmuneReceptor(receptor_id="1", chains=[TRB4], species="mouse")
        self.rep1 = [TCR1, TCR2]
        self.rep2 = [TCR3, TCR4]
        self.rep3 = []
        self.rep4 = "TCR"
        self.rep5 = TCR1
        self.rep6 = [TCR1, ""]
        self.rep7 = [TCR5, TCR1]
        self.path1 = "/home/mahmoud/Documents/example.csv"
        self.df1 = pd.read_csv("/home/mahmoud/Downloads/dash.csv").head(2)
        self.path2 = "/home/mahmoud/Documents/example2.csv"
        self.df2 = pd.read_csv("/home/mahmoud/Downloads/dash.csv").iloc[2:5, ]

    def test_wrong_input(self):
        mo = TCRSimilarityMeasurementFactory("tcrdist3")
        with self.assertRaises(ValueError):
            # an empty repertoire or None
            mo.compute_distance(rep1=self.rep3, organism="mouse", seq_type="ALPHABETA")
            mo.compute_distance(rep1=None, organism="mouse", seq_type="ALPHABETA")
            # the second repertoire is empty
            mo.compute_distance(rep1=self.rep2, rep2=[], organism="mouse", seq_type="ALPHABETA")
            # the second repertoire has a non AntigenImmuneReceptor object
            mo.compute_distance(rep1=self.rep2, rep2=self.rep6, organism="mouse", seq_type="ALPHABETA")
            # the second repertoire is not an AntigenImmuneReceptor object
            mo.compute_distance(rep1=self.rep2, rep2="", organism="mouse", seq_type="ALPHABETA")
            # a repertoire with a wrong instance type
            mo.compute_distance(rep1=self.rep4, organism="mouse", seq_type="ALPHABETA")
            mo.compute_distance(rep1="", organism="mouse", seq_type="ALPHABETA")
            # a list of different objects
            mo.compute_distance(rep1=self.rep6, organism="mouse", seq_type="ALPHABETA")
            mo.compute_distance(rep1=self.rep2, rep2=self.rep6, organism="mouse", seq_type="ALPHABETA")
            # a wrong metric name
            mo.compute_distance_from_dataset(df1=self.df1, organism="mouse", seq_type="ALPHABETA", metric="dist")
            # TCRs without cdr3 alpha or beta
            mo.compute_distance(rep1=self.rep7, organism="mouse", seq_type="ALPHABETA")
            # wrong seq_type
            mo.compute_distance(rep1=self.rep1, organism="mouse", seq_type="ALP")
        with self.assertRaises(FileNotFoundError):
            # a None path to a dataset
            mo.compute_distance_from_dataset(organism="mouse", seq_type="ALPHABETA")
            # a wrong path to the second dataset
            mo.compute_distance_from_dataset(path1=self.path1, path2="", organism="mouse", seq_type="ALPHABETA")

    def test_TCR_similarity_one_repertoire(self):
        for m in TCRSimilarityMeasurementFactory.available_methods():
            mo = TCRSimilarityMeasurementFactory(m)
            print("\nTesting", mo.name)
            print("Test similarity computation for one repertoire")
            print("Test similarity for alpha seqs")
            print("default TCRrep object\n")
            mo.compute_distance(rep1=self.rep1, organism="mouse", seq_type="ALPHA")
            print("\nTesting another metric for alpha seqs\n")
            mo.compute_distance(rep1=self.rep1, organism="mouse", seq_type="ALPHA", default=False,
                                metric='nw_hamming_metric', open=2, extend=2)
            print("\nTest similarity for beta seqs")
            print("default TCRrep object\n")
            mo.compute_distance(rep1=self.rep1, organism="mouse", seq_type="BETA")
            print("\nTesting another metric for beta seqs\n")
            mo.compute_distance(rep1=self.rep1, organism="mouse", seq_type="ALPHA", default=False,
                                metric='nw_metric', open=1, extend=1)
            print("\nTesting similarity for alpha beta seqs separately")
            print("default TCRrep object")
            mo.compute_distance(rep1=self.rep1, organism="mouse", seq_type="ALPHABETA")
            print("\nTesting another metric for alpha beta seqs")
            mo.compute_distance(rep1=self.rep1, organism="mouse", seq_type="ALPHABETA", default=False,
                                metric="nb_vector_tcrdist")
            print("\nTesting similarity for both alpha beta seqs combined")
            mo.compute_distance(rep1=self.rep1, organism="mouse", seq_type="BOTH", default=False,
                                metric='nb_vector_editdistance')

    def test_TCR_similarity_two_repertoires(self):
        for m in TCRSimilarityMeasurementFactory.available_methods():
            mo = TCRSimilarityMeasurementFactory(m)
            print("\nTest similarity computation for tow repertoires\n")
            print("Test similarity for alpha seqs")
            print("default TCRrep object\n")
            mo.compute_distance(rep1=self.rep1, rep2=self.rep2, organism="mouse", seq_type="ALPHA")
            print("\nTesting another metric for alpha seqs\n")
            mo.compute_distance(rep1=self.rep1, rep2=self.rep2, organism="mouse", seq_type="ALPHA", default=False,
                                metric='nw_hamming_metric', open=2, extend=2)
            print("\nTest similarity for beta seqs")
            print("default TCRrep object\n")
            mo.compute_distance(rep1=self.rep1, rep2=self.rep2, organism="mouse", seq_type="BETA")
            print("\nTesting another metric for beta seqs\n")
            mo.compute_distance(rep1=self.rep1, rep2=self.rep2, organism="mouse", seq_type="ALPHA", default=False,
                                metric='nw_metric', open=1, extend=1)
            print("\nTesting similarity for alpha beta seqs separately")
            print("default TCRrep object")
            mo.compute_distance(rep1=self.rep1, rep2=self.rep2, organism="mouse", seq_type="ALPHABETA")
            print("\nTesting another metric for alpha beta seqs")
            mo.compute_distance(rep1=self.rep1, rep2=self.rep2, organism="mouse", seq_type="ALPHABETA", default=False,
                                metric="nb_vector_tcrdist")
            print("\nTesting similarity for both alpha beta seqs combined")
            mo.compute_distance(rep1=self.rep1, rep2=self.rep2, organism="mouse", seq_type="BOTH", default=False,
                                metric='nb_vector_hamming_distance')

    def test_TCR_similarity_computation_one_dataset(self):
        for m in TCRSimilarityMeasurementFactory.available_methods():
            mo = TCRSimilarityMeasurementFactory(m)
            print("Testing similarity for one dataset")
            print("Test similarity for alpha seqs")
            mo.compute_distance_from_dataset(df1=self.df1, organism="mouse", seq_type="ALPHA", source="dash")
            print("\nTesting another metric for alpha seqs\n")
            mo.compute_distance_from_dataset(df1=self.df1, organism="mouse", seq_type="ALPHA", default=False,
                                             metric='nw_hamming_metric', open=2, extend=2, source="dash")
            print("\nTest similarity for beta seqs")
            print("default TCRrep object\n")
            mo.compute_distance_from_dataset(df1=self.df1, organism="mouse", seq_type="BETA", source="dash")
            print("\nTesting another metric for beta seqs\n")
            mo.compute_distance_from_dataset(df1=self.df1, organism="mouse", seq_type="ALPHA", default=False,
                                             metric='nw_metric', open=1, extend=1, source="dash")
            print("\nTesting similarity for alpha beta seqs separately")
            print("default TCRrep object")
            mo.compute_distance_from_dataset(df1=self.df1, organism="mouse", seq_type="ALPHABETA", source="dash")
            print("\nTesting another metric for alpha beta seqs")
            mo.compute_distance_from_dataset(df1=self.df1, organism="mouse", seq_type="ALPHABETA", default=False,
                                             metric="nb_vector_tcrdist", source="dash")
            print("\nTesting similarity for both alpha beta seqs combined")
            mo.compute_distance_from_dataset(df1=self.df1, organism="mouse", seq_type="BOTH", default=False,
                                             metric='nb_vector_editdistance', source="dash")

    def test_TCR_similarity_computation_two_datasets(self):
        for m in TCRSimilarityMeasurementFactory.available_methods():
            mo = TCRSimilarityMeasurementFactory(m)
            print("Testing similarity for two datasets")
            print("Test similarity for alpha seqs")
            print("default TCRrep object\n")
            mo.compute_distance_from_dataset(df1=self.df1, df2=self.df2, organism="mouse", seq_type="ALPHA",
                                             source="dash")
            print("\nTesting another metric for alpha seqs\n")
            mo.compute_distance_from_dataset(df1=self.df1, df2=self.df2, organism="mouse", seq_type="ALPHA",
                                             default=False, metric='nw_hamming_metric', open=2, extend=2, source="dash")
            print("\nTest similarity for beta seqs")
            print("default TCRrep object\n")
            mo.compute_distance_from_dataset(df1=self.df1, df2=self.df2, organism="mouse", seq_type="BETA",
                                             source="dash")
            print("\nTesting another metric for beta seqs\n")
            mo.compute_distance_from_dataset(df1=self.df1, df2=self.df2, organism="mouse", seq_type="ALPHA",
                                             default=False, metric='nw_metric', open=1, extend=1, source="dash")
            print("\nTesting similarity for alpha beta seqs separately")
            print("default TCRrep object")
            mo.compute_distance_from_dataset(df1=self.df1, df2=self.df2, organism="mouse", seq_type="ALPHABETA",
                                             source="dash")
            print("\nTesting another metric for alpha beta seqs")
            mo.compute_distance_from_dataset(df1=self.df1, df2=self.df2, organism="mouse", seq_type="ALPHABETA",
                                             default=False, metric="nb_vector_tcrdist", source="dash")
            print("\nTesting similarity for both alpha beta seqs combined")
            print("\nTesting another metric for both alpha beta seqs combined")
            mo.compute_distance_from_dataset(df1=self.df1, df2=self.df2, organism="mouse", seq_type="BOTH",
                                             default=False, metric='nb_vector_editdistance', source="dash")

    def test_TCR_similarity_computation_dataset_from_path(self):
        for m in TCRSimilarityMeasurementFactory.available_methods():
            mo = TCRSimilarityMeasurementFactory(m)
            print("Testing similarity for one dataset given the path to it")
            print("Test similarity for alpha seqs")
            print("default TCRrep object\n")
            mo.compute_distance_from_dataset(path1=self.path1, organism="mouse", seq_type="ALPHA", source="dash")
            print("\nTesting another metric for alpha seqs\n")
            mo.compute_distance_from_dataset(path1=self.path1, organism="mouse", seq_type="ALPHA", default=False,
                                             metric='nw_hamming_metric', source="dash", open=2, extend=2)
            print("\nTest similarity for beta seqs")
            print("default TCRrep object\n")
            mo.compute_distance_from_dataset(path1=self.path1, organism="mouse", seq_type="BETA", source="dash")
            print("\nTesting another metric for beta seqs\n")
            mo.compute_distance_from_dataset(path1=self.path1, organism="mouse", seq_type="ALPHA", default=False,
                                             metric='nw_metric', open=1, extend=1, source="dash")
            print("\nTesting similarity for alpha beta seqs separately")
            print("default TCRrep object")
            mo.compute_distance_from_dataset(path1=self.path1, organism="mouse", seq_type="ALPHABETA", source="dash")
            print("\nTesting another metric for alpha beta seqs")
            mo.compute_distance_from_dataset(path1=self.path1, path2=self.path2, organism="mouse", seq_type="ALPHABETA",
                                             default=False, metric="nb_vector_tcrdist", source="dash")
            print("\nTesting similarity for both alpha beta seqs combined")
            mo.compute_distance_from_dataset(path1=self.path1, path2=self.path2, organism="mouse", seq_type="BOTH",
                                             default=False, metric='nb_vector_editdistance', source="dash")

    def test_merging_and_filtering(self):
        mo = TCRSimilarityMeasurementFactory("tcrdist3")
        result1 = mo.compute_distance(rep1=self.rep1, organism="mouse", seq_type="BETA")
        result2 = mo.compute_distance(rep1=self.rep1, organism="mouse", seq_type="ALPHA")
        print("\n\nTest merging")
        print(result1.merge_results(result2))
        print("\n\nTest filtering")
        com = lambda x, y: x > y
        thr = 30
        expression = (mo.name, com, thr)
        result = mo.compute_distance_from_dataset(path1=self.path1, organism="mouse", seq_type="BETA", source="dash")
        print(f"\nresult before filtering:\n{result}\n\nresult after filtering:\n"
              f"{result.filter_result(expression, 'cdr3_b')}")


if __name__ == '__main__':
    unittest.main()
