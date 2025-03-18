__author__ = "albahah,drost"

import unittest
import argparse
import yaml
import re
import os
import sys
import warnings

from epytope.Core import TCREpitope, Allele
from epytope.IO.IRDatasetAdapter import IRDataset
from epytope.TCRSpecificityPrediction import TCRSpecificityPredictorFactory


class TestTCRSpecificityPrediction(unittest.TestCase):
    predictors= "all"
    config_yaml = None
    options_config = None

    @classmethod
    def setUpClass(cls):
        super(TestTCRSpecificityPrediction, cls).setUpClass()
        # testing setup
        if cls.predictors == "all":
            cls.predictors = list(TCRSpecificityPredictorFactory.available_methods())
        else:
            cls.predictors = cls.predictors.split(",")

        _var_matcher = re.compile(r"\${([^}^{]+)}")
        _tag_matcher = re.compile(r"[^$]*\${([^}^{]+)}.*")
        def _path_constructor(_loader, node):
            def replace_fn(match):
                envparts = f"{match.group(1)}:".split(":")
                return os.environ.get(envparts[0], envparts[1])

            return _var_matcher.sub(replace_fn, node.value)
        yaml.add_implicit_resolver("!envvar", _tag_matcher, None, yaml.SafeLoader)
        yaml.add_constructor("!envvar", _path_constructor, yaml.SafeLoader)

        if cls.config_yaml is None or not os.path.exists(cls.config_yaml):
            raise ValueError("Please provide a valid path to a configuration file for your predictors."
                             f"Currently: {cls.config_yaml}")
        with open(cls.config_yaml, "r") as yaml_file:
            cls.config_yaml = yaml.safe_load(yaml_file)

        if cls.options_yaml is not None and not os.path.exists(cls.options_yaml):
            raise ValueError("Please provide a valid path to an option file for your predictors."
                             f"Currently: {cls.options_yaml}")
        if cls.options_yaml is not None:
            with open(cls.options_yaml, "r") as yaml_file:
                cls.options_yaml = yaml.safe_load(yaml_file)

        # Sample Eptiopes

        epitope1 = TCREpitope("FLRGRAYGL", allele=Allele("HLA-B*08:01"))
        epitope2 = TCREpitope("GILGFVFTL", allele=Allele("HLA-A*02:01"))
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

            self.assertEqual(len(results), len(self.tcr_repertoire.receptors), "Results have wrong length")
            for epitope in epitopes_pairwise:
                self.assertIn(epitope, results, f"{name}: Epitope not in result")
                self.assertIn(name, [el.lower() for el in results[epitope].columns.tolist()],
                              f"{name}: Method not in results")
                self.assertLess(results[epitope].iloc[:, 0].isna().sum(), len(results),
                                f"{name}: Method always yield NaN")

    def test_tcr_list(self):
        epitopes_list = self.epitopes * (len(self.tcr_repertoire.receptors)//2)
        for name in self.predictors:
            predictor = TCRSpecificityPredictorFactory(name)
            config_predictor = None if name not in self.config_yaml else self.config_yaml[name]
            results = predictor.predict(self.tcr_repertoire, epitopes_list, pairwise=False, **config_predictor)
    
            self.assertEqual(len(results), len(self.tcr_repertoire.receptors), f"{name}: Results have wrong length")
            self.assertIn(name, [el.lower() for el in results["Method"].columns], f"{name}: Method missing in results")
            self.assertLess(results["Method"].iloc[:, 0].isna().sum(), len(results), f"{name}: Method always yield NaN")
            for i, epitope in enumerate(epitopes_list):
                self.assertEqual(results.at[i, ("Epitope", "Peptide")], epitope.peptide, f"{name}: Wrong epitope at position {i}")
                self.assertEqual(results.at[i, ("Epitope", "MHC")], epitope.allele, f"{name}: Wrong MHC at position {i}")

    def test_tcr_single_input(self):
        for name in self.predictors:
            predictor = TCRSpecificityPredictorFactory(name)
            config_predictor = None if name not in self.config_yaml else self.config_yaml[name]
            results = predictor.predict(self.tcr_repertoire.receptors[:1], self.epitopes[0], pairwise=True, **config_predictor)
            self.assertEqual(len(results), 1, f"{name}: Length result is different from 1")
            self.assertIn(self.epitopes[0], results, f"{name}: Epitope not in result")
            self.assertIn(name, [el.lower() for el in results[self.epitopes[0]].columns.tolist()], f"{name}: Method not in results")
            self.assertLess(results[self.epitopes[0]].iloc[:, 0].isna().sum(), 1, f"{name}: Method always yield NaN")
            
            results = predictor.predict(self.tcr_repertoire.receptors[:1],self.epitopes[0], pairwise=False, **config_predictor)
            self.assertEqual(len(results), 1, f"{name}: Results have wrong length")
            self.assertIn(name, [el.lower() for el in results["Method"].columns], f"{name}: Method missing in results")
            self.assertLess(results["Method"].iloc[:, 0].isna().sum(), 1, f"{name}: Method always yield NaN")
            self.assertEqual(results.at[0, ("Epitope", "Peptide")], self.epitopes[0].peptide, f"{name}: Wrong epitope at position 0")
            self.assertEqual(results.at[0, ("Epitope", "MHC")], self.epitopes[0].allele, f"{name}: Wrong MHC at position 0")
            
    def form_options(self, config, base_name):
        base_config = config.copy()
        options = base_config.pop("options")

        all_configs = {base_name: base_config}
        for opt_name, opt_dict in options.items():
            all_configs_new = {}
            for old_name, old_config in all_configs.items():
                for add_name, add_value in opt_dict.items():
                    add_config = {opt_name: add_value}
                    all_configs_new[f"{old_name}_{add_name}"] = {**old_config, **add_config}
            all_configs = all_configs_new
        return all_configs

    def test_tcr_options(self):
        if self.options_yaml is None:
            warnings.warn("Skipping Test for predictors options as --options_yaml was not set.")
            return
        
        epitopes_pairwise = self.epitopes
        epitopes_list = self.epitopes * (len(self.tcr_repertoire.receptors)//2)
        for name in self.predictors:
            predictor = TCRSpecificityPredictorFactory(name)
            if name not in self.options_yaml:
                continue
            config_predictor = self.options_yaml[name]
            configs = self.form_options(config_predictor, name)
            for opt_name, config in configs.items():
                results = predictor.predict(self.tcr_repertoire, epitopes_pairwise, pairwise=True, **config_predictor)
                self.assertEqual(len(results), len(self.tcr_repertoire.receptors), "Results have wrong length")
                for epitope in epitopes_pairwise:
                    self.assertIn(epitope, results, f"{opt_name}: Epitope not in result")
                    self.assertIn(name, [el.lower() for el in results[epitope].columns.tolist()],
                                  f"{opt_name}: Method not in results")
                    self.assertLess(results[epitope].iloc[:, 0].isna().sum(), len(results),
                                    f"{opt_name}: Method always yield NaN")
                
                results = predictor.predict(self.tcr_repertoire, epitopes_list, pairwise=False, **config_predictor)
                self.assertEqual(len(results), len(self.tcr_repertoire.receptors), f"{opt_name}: Results have wrong length")
                self.assertIn(name, [el.lower() for el in results["Method"].columns], f"{opt_name}: Method missing in results")
                self.assertLess(results["Method"].iloc[:, 0].isna().sum(), len(results), f"{opt_name}: Method always yield NaN")
                for i, epitope in enumerate(epitopes_list):
                    self.assertEqual(results.at[i, ("Epitope", "Peptide")], epitope.peptide, f"{opt_name}: Wrong epitope at position {i}")
                    self.assertEqual(results.at[i, ("Epitope", "MHC")], epitope.allele, f"{opt_name}: Wrong MHC at position {i}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictors", type=str, default="all")
    parser.add_argument("--config_yaml", type=str)
    parser.add_argument("--options_yaml", type=str, default=None)
    args = parser.parse_args()

    TestTCRSpecificityPrediction.predictors = args.predictors
    TestTCRSpecificityPrediction.config_yaml = args.config_yaml
    TestTCRSpecificityPrediction.options_yaml = args.options_yaml
    unittest.main(argv=sys.argv[:1])
