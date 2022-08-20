"""
Unit test for TCREpitope class
"""

import unittest

from epytope.Core.TCREpitope import TCREpitope


class TestTCREpitopeClass(unittest.TestCase):
    def setUp(self):
        self.epitope = TCREpitope(seq="RAKFKQLL", mhc="HLA-B*08")

    def test1_peptide_construction_novariants(self):
        self.assertEqual(str(self.epitope), "RAKFKQLL", 'incorrect sequence')
        self.assertEqual(self.epitope.mhc, "HLA-B*08", 'incorrect mhc')

    def test2_invalid_peptide_seq(self):
        with self.assertRaises(ValueError):
            TCREpitope(seq="RAKFBKQLL", mhc="HLA-B*08")


if __name__ == '__main__':
    unittest.main()