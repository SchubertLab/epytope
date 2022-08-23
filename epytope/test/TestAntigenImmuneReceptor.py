"""
Unit test for AntigenImmuneReceptor class
"""

import unittest
from epytope.Core.AntigenImmuneReceptor import AntigenImmuneReceptor, BCellReceptor, TCellReceptor
from epytope.Core.ImmuneReceptorChain import ImmuneReceptorChain


class TestImmuneReceptorChainClass(unittest.TestCase):
    def setUp(self):
        alpha = ImmuneReceptorChain(chain_id="57", chain_type="TRA", v_gene="TRAV26-1", d_gene=None,
                                         j_gene="TRAJ37", cdr3="IVVRSSNTGKLI", cdr1="TISGNEY", cdr2="GLKNN")
        beta = ImmuneReceptorChain(chain_id="57", chain_type="TRB", v_gene="TRBV14", d_gene=None,
                                         j_gene="TRBJ2-3", cdr3="ASSQDRDTQY", cdr1="SGHDN", cdr2="FVKESK")
        self.tcr = AntigenImmuneReceptor(receptor_id=alpha.chain_id, chains=[alpha, beta], cell_type="CD8",
                                         tissue="PBMC")

        heavy_chain = ImmuneReceptorChain(chain_id="58", chain_type="IGH", v_gene="", d_gene=None,
                                         j_gene="", cdr3="IVVRSSNTGKLI")
        light_chain = ImmuneReceptorChain(chain_id="58", chain_type="IGL", v_gene="", d_gene=None,
                                          j_gene="", cdr3="IVVRSSNTGKLI")
        self.bcr = AntigenImmuneReceptor(receptor_id=alpha.chain_id, chains=[heavy_chain, light_chain], cell_type="",
                                         t_cell=False)

    def test1_chain_construction_novariants(self):
        self.assertEqual(self.tcr.receptor_id, "57", 'incorrect Id')
        self.assertEqual(self.tcr.cell_type, "CD8", 'incorrect chain type')
        self.assertEqual(self.tcr.tissue, "PBMC", 'incorrect tissue')
        self.assertIsInstance(self.tcr.chains[0], ImmuneReceptorChain, "incorrect chain type")
        self.assertIsInstance(self.tcr.chains[1], ImmuneReceptorChain, "incorrect chain type")
        self.assertIsInstance(self.tcr, TCellReceptor, "incorrect receptor type")
        self.assertIsInstance(self.bcr, BCellReceptor, "incorrect receptor type")


if __name__ == '__main__':
    unittest.main()