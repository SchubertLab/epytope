__author__ = 'schubert'

import unittest

from epytope.Core.Allele import Allele
from epytope.Core.Peptide import Peptide
from epytope.EpitopePrediction import EpitopePredictorFactory
from epytope.CleavagePrediction import CleavageSitePredictorFactory
from epytope.EpitopeAssembly.EpitopeAssembly import EpitopeAssemblyWithSpacer


class SpacerDesignTestCase(unittest.TestCase):
    """
        Unittest for OptiTope
    """

    def setUp(self):
        epis = """GHRMAWDM
                 VYEADDVI""".split("\n")

        self.epis = [Peptide(x.strip()) for x in epis]
        self.alleles = [Allele("HLA-A*02:01", prob=0.5)]

    def test_standart_functions(self):
        """
        Tests default functions
        needs GLPK installed
        :return:
        """
        epi_pred = EpitopePredictorFactory("Syfpeithi")
        cl_pred = CleavageSitePredictorFactory("PCM")

        sbws = EpitopeAssemblyWithSpacer(self.epis, cl_pred, epi_pred, self.alleles, solver="glpk")
        sol = sbws.solve()
        print(sol)
        assert all(i == str(j) for i, j in zip(["GHRMAWDM", "WWQW", "VYEADDVI"], sol))

    def test_unsupported_allele_length_combination(self):
        """
        Tests default functions
        needs GLPK installed
        :return:
        """
        epi_pred = EpitopePredictorFactory("Syfpeithi")
        cl_pred = CleavageSitePredictorFactory("PCM")
        alleles = [Allele("HLA-A*02:01", prob=0.5), Allele("HLA-A*26:01", prob=0.5)]
        sbws = EpitopeAssemblyWithSpacer(self.epis, cl_pred, epi_pred, alleles, solver="glpk")
        sol = sbws.solve()
        print(sol)
        assert all(i == str(j) for i, j in zip(["GHRMAWDM", "WWRW", "VYEADDVI"], sol))

    def test_unsupported_allele_length_combination_exception(self):
        """
        Tests default functions
        needs GLPK installed
        :return:
        """
        epi_pred = EpitopePredictorFactory("Syfpeithi")
        cl_pred = CleavageSitePredictorFactory("PCM")
        alleles = [Allele("HLA-A*26:01", prob=0.5)]
        sbws = EpitopeAssemblyWithSpacer(self.epis, cl_pred, epi_pred, alleles, solver="glpk", en=8)
        self.assertRaises(ValueError, sbws.solve)


if __name__ == '__main__':
    unittest.main()
