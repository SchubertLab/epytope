__author__ = "albahah,drost"

import unittest
import argparse
import yaml
import os
import warnings

from epytope.Core import TCREpitope
from epytope.IO.IRDatasetAdapter import IRDataset


class TestTCRSpecificityPrediction(unittest.TestCase):
    predictors = "all"
    config_yaml = None
    options_yaml = None

    @classmethod
    def setUp(cls):
        # testing setup
        if cls.predictors == "all":
            cls.predictors = list(TCRSpecificityPredictorFactory.available_methods())
        else:
            cls.predictors = cls.predictors.split(",")

        if cls.config_yaml is None or not os.path.exists(cls.config_yaml):
            raise ValueError("Please provide a valid path to a configuration file for your predictors."
                             f"Currently: {cls.config_yaml}")
        with open(cls.config_yaml, "r") as yaml_file:
            cls.config_yaml = yaml.safe_load(yaml_file)

        if cls.options_yaml is not None or not os.path.exists(cls.config_yaml):
            raise ValueError("Please provide a valid path to an option file for your predictors."
                             f"Currently: {cls.options_yaml}")
        else:
            with open(cls.options_yaml, "r") as yaml_file:
                cls.options_yaml = yaml.safe_load(yaml_file)

        # Sample Eptiopes
        epitope1 = TCREpitope("FLRGRAYGL", allele="HLA-B*08:01")
        epitope2 = TCREpitope("HSKRKCDEL", allele="HLA-A*02:01")
        cls.epitopes = [epitope1, epitope2]

        # Sample TCRs
        path_data = os.path.join(os.path.dirname(__file__), "../../Data/examples/test_tcrs.csv")
        cls.tcr_repertoire = IRDataset()
        cls.tcr_repertoire.from_path(path_data)

    def test_tcr_pairwise(self):
        epitopes_pairwise = self.epitopes
        for name in self.predictors:
            predictor = TCRSpecificityPredictorFactory(name)
            config_predictor = None if name not in self.config_yaml else self.config_yaml[name]
            results = predictor.predict(self.tcr_repertoire, epitopes_pairwise, pairwise=True, **config_predictor)

            self.assertEqual(len(results), len(tcr_repertoire.receptors), "Results have wrong length")
            for epitope in epitopes_pairwise:
                self.assertIn(epitope, results, f"{name}: Epitope not in result")
                self.assertIn(name, [el.lower() for el in results[epitope].columns.tolist()],
                              f"{name}: Method not in results")
                self.assertLess(results[epitope].iloc[:, 0].isna().sum(), len(results),
                                f"{name}: Method always yield NaN")

    def test_tcr_list(self):
        for name in self.predictors:
            predictors = TCRSpecificityPredictorFactory(name)

    def test_tcr_empty_input(self):
        for name in self.predictors:
            predictor = TestTCRSpecificityPredictionClass(name)

    def test_tcr_single_input(self):
        for name in self.predictors:
            predictor = TestTCRSpecificityPredictionClass(name)

    def test_tcr_options(self):
        if self.options_yaml is None:
            warnings.warn("Skipping Test for predictors options as --options_yaml was not set.")
            return
        for name in self.predictors:
            predictor = TestTCRSpecificityPredictionClass(name)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictors", default="all")
    parser.add_argument("--config_yaml")
    parser.add_argument("--options_yaml", default=None)
    args = parser.parse_args()

    TestTCRSpecificityPrediction.predictors = args.predictos
    TestTCRSpecificityPrediction.config_yaml = args.config_yaml
    TestTCRSpecificityPrediction.options_yaml = args.options_yaml
    unittest.main()
