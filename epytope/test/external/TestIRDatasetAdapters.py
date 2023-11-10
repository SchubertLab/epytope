__author__ = "chernysheva, drost"

import unittest
import argparse
import yaml
import os
import sys
import warnings


from epytope.Core import TCREpitope
from epytope.IO.IRDatasetAdapter import IRDataset
from epytope.IO import IRDatasetAdapterFactory


class TestIRDatasetAdapters(unittest.TestCase):
    predictors= "all"
    config_yaml = None
    options_config = None

    @classmethod
    def setUpClass(cls):
        super(TestIRDatasetAdapters, cls).setUpClass()
        # testing setup
        if cls.adapters == "all":
            cls.adapters = list(IRDatasetAdapterFactory.available_methods())
        else:
            cls.adapters = cls.adapters.split(",")

        # Sample TCRs
        cls.path_data = {"vdjdb": os.path.join(os.path.dirname(__file__), "../../Data/examples/test_vdjdb_example.tsv"),
                     "mcpas-tcr": os.path.join(os.path.dirname(__file__), "../../Data/examples/test_mcpas_example.csv"),
                     "iedb": os.path.join(os.path.dirname(__file__), "../../Data/examples/test_iedb_example.csv"),
                     "scirpy": os.path.join(os.path.dirname(__file__), "../../Data/examples/test_adata_example.h5ad"),
                     "airr": [os.path.join(os.path.dirname(__file__), "../../Data/examples/test_airr_example_alpha.tsv"), os.path.join(os.path.dirname(__file__), "../../Data/examples/test_airr_example_beta.tsv")]}


    def test_adapter(self):
        for name in self.adapters:
            tcr_repertoire = IRDatasetAdapterFactory(name)
            tcr_repertoire.from_path(self.path_data[name])
            print(tcr_repertoire.to_pandas())
            result = tcr_repertoire.to_pandas()
            self.assertEqual(len(result), 16, f"{name}: Length result is different from 20")
            self.assertLess(result["VDJ_chain_type"].isna().sum(), len(result), f"{name}: Method always yield NaN")
            self.assertLess(result["VDJ_cdr3"].isna().sum(), len(result), f"{name}: Method always yield NaN")
            self.assertLess(result["VJ_chain_type"].isna().sum(), len(result), f"{name}: Method always yield NaN")
            self.assertLess(result["VJ_cdr3"].isna().sum(), len(result), f"{name}: Method always yield NaN")
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapters", type=str, default="all")
    args = parser.parse_args()

    TestIRDatasetAdapters.adapters = args.adapters
    unittest.main(argv=sys.argv[:1])
