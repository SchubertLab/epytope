"""
Unit test for ImmuneReceptorChain class
"""

import unittest

from epytope.Core.ImmuneReceptorChain import ImmuneReceptorChain


class TestImmuneReceptorChainClass(unittest.TestCase):
    def setUp(self):
        self.chain = ImmuneReceptorChain(chain_id="57", chain_type="TRA", v_gene="TRAV26-1", d_gene=None,
                                         j_gene="TRAJ37", cdr3="IVVRSSNTGKLI", cdr1="TISGNEY", cdr2="GLKNN")

    def test1_chain_construction_novariants(self):
        self.assertEqual(self.chain.chain_id, "57", 'incorrect Id')
        self.assertEqual(self.chain.cdr3, "IVVRSSNTGKLI", 'incorrect cdr3')
        self.assertEqual(self.chain.cdr2, "GLKNN", 'incorrect cdr2')
        self.assertEqual(self.chain.cdr1, "TISGNEY", "incorrect cdr2")
        self.assertIsNone(self.chain.d_gene)
        self.assertEqual(self.chain.v_gene, "TRAV26-1", "incorrect v_gene")
        self.assertEqual(self.chain.j_gene, "TRAJ37", "incorrect j_gene")
        self.assertEqual(self.chain.chain_type, "TRA", "incorrect chain_type")

    def test2_invalid_cdr_seqs(self):
        with self.assertRaises(ValueError):
            # invalid cdr3
            ImmuneReceptorChain(chain_id="57", chain_type="TRA", v_gene="TRAV26-1", d_gene=None,
                                                j_gene="TRAJ37", cdr3="IVVRSSBTGKLI", cdr1="TISGNEY", cdr2="GLKNN")
            # invalid cdr2
            ImmuneReceptorChain(chain_id="57", chain_type="TRA", v_gene="TRAV26-1", d_gene=None,
                                                j_gene="TRAJ37", cdr3="IVVRSSNTGKLI", cdr1="TISGNEY", cdr2="GBKNN")
            # invalid cdr1
            ImmuneReceptorChain(chain_id="57", chain_type="TRA", v_gene="TRAV26-1", d_gene=None,
                                                j_gene="TRAJ37", cdr3="IVVRSSNTGKLI", cdr1="TBBSGNEY", cdr2="GLKNN")

    def test3_string_representation(self):
        self.assertEqual(str(self.chain), f"TRA\nV_gene: TRAV26-1\nJ_gene: TRAJ37\nCDR1: TISGNEY\nCDR2: GLKNN\nCDR3: "
                                          f"IVVRSSNTGKLI", "incorrect representation")


if __name__ == '__main__':
    unittest.main()

