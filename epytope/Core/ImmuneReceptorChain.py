# This code is part of the epytope distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
.. module:: Core.ImmuneReceptorChain
   :synopsis: Contains the Immune Receptor Chain class of a B respectively T cell receptor
   :Note: All internal indices start at 0!
.. moduleauthor:: albahah, drost
"""


from Bio.Seq import Seq
from epytope.Core.Base import MetadataLogger


class ImmuneReceptorChain(MetadataLogger):
    """
        :class:`~epytope.Core.ImmuneReceptorChain.ImmuneReceptorChain` corresponding to exactly one polypeptide chain of
            a B respectively T cell receptor
    """
    def __init__(self, chain_type, cdr3, v_gene=None, d_gene=None, j_gene=None):
        """
        :param str chain_type: String representing the type of the polypeptide chain,
            ["TRA", "TRB", "TRG", "TRD", "IGK", "IGH", "IGL"]
        :param str v_gene: variable gene in the nomenclature 'TRBV4-1'
        :param str d_gene: diversity gene in the nomenclature 'TRBD1'
        :param str j_gene: joining gene in the nomenclature 'TRBJ1-1'
        :param str cdr3: Complementary Determining Region (CDR3) protein sequence in IUPACProtein alphabet
        """
        # Init parent type:
        MetadataLogger.__init__(self)
        self.chain_type = chain_type
        self.v_gene = v_gene
        self.d_gene = d_gene
        self.j_gene = j_gene
        self.cdr3 = Seq(cdr3.upper())

    def __repr__(self):
        lines = [f"IMMUNE RECEPTOR CHAIN: {self.chain_type}"]

        receptor_information = {
            "CDR3": self.cdr3,
            "V_gene": self.v_gene,
            "D_gene": self.d_gene,
            "J_gene": self.j_gene
        }
        for name, value in receptor_information.items():
            if value:
                lines.append(f"{name}: {value}")
        return "\n".join(lines)

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))
