# This code is part of the epytope distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
.. module:: Core.ImmuneReceptorChain
   :synopsis: Contains the Immune Receptor Chain class of a B respectively T cell receptor
   :Note: All internal indices start at 0!
.. moduleauthor:: albahah
"""


from Bio.Seq import Seq
from epytope.Core.Base import MetadataLogger


class ImmuneReceptorChain(MetadataLogger):
    """
        :class:`~epytope.Core.ImmuneReceptorChain.ImmuneReceptorChain` corresponding to exactly one polypeptide chain of
            a B respectively T cell receptor
    """
    def __init__(self, chain_type: str, v_gene: str, d_gene: str, j_gene: str, cdr3: str, cdr1: str = None,
                 cdr2: str = None, chain_id: str = None, nuc_seq: str = None):
        """
        :param str chain_type: String representing the type of the polypeptide chain,
            ["TRA", "TRB", "TRG", "TRD", "IGK", "IGH", "IGL"]
        :param str v_gene: String representing the V segment
        :param str d_gene: String representing the D segment
        :param str j_gene: String representing the J segment
        :param str cdr1: String of an IUPACProtein alphabet, representing the CDR1 amino acid sequence
        :param str cdr2: String of an IUPACProtein alphabet, representing the CDR2 amino acid sequence
        :param str cdr3: String of an IUPACProtein alphabet, representing the CDR3 amino acid sequence
        :param str chain_id: it can be sequence id or the id of the subject from whom the data has been obtained
        :param str nuc_seq: a string representing the nucleotide sequence of the chain
        """
        # Init parent type:
        MetadataLogger.__init__(self)
        for seq in [cdr1, cdr2, cdr3]:
            if seq:
                if self.invalid(seq):
                    raise ValueError(f"{seq} contains non amino acid letters")
        if nuc_seq and not isinstance(nuc_seq, str):
            raise ValueError(f"{nuc_seq} must be of type str")
        if nuc_seq and not set(nuc_seq.upper()) == {"A", 'T', 'C', 'G'}:
            raise ValueError(f"{nuc_seq} must consist of only the following bases [A, T, G, C]")
        self.chain_type = chain_type
        self.v_gene = v_gene
        self.d_gene = d_gene
        self.j_gene = j_gene
        self.cdr1 = cdr1
        self.cdr2 = cdr2
        self.cdr3 = Seq(cdr3.upper())
        self.chain_id = chain_id
        self.nuc_seq = nuc_seq.upper() if nuc_seq else None

    def __repr__(self):
        lines = [self.chain_type]
        if self.v_gene:
            lines.append(f"V_gene: {self.v_gene}")
        if self.d_gene:
            lines.append(f"D_gene: {self.d_gene}")
        if self.j_gene:
            lines.append(f"J_gene: {self.j_gene}")
        if self.cdr1:
            lines.append(f"CDR1: {self.cdr1}")
        if self.cdr2:
            lines.append(f"CDR2: {self.cdr2}")
        if self.cdr3:
            lines.append(f"CDR3: {self.cdr3}")
        if self.nuc_seq:
            lines.append(f"Nucleotide sequence: {self.nuc_seq}")
        return "\n".join(lines)

    def invalid(self, seq: str) -> bool:
        """
        checks if the passed sequence is an invalid protein sequence
        :param str seq: a String representing the protein sequence
        :return: Returns true if the passed sequence is not a protein sequence
        :rtype: bool
        """
        aas = set("ARNDCEQGHILKMFPSTWYV")
        if seq:
            return any([aa not in aas for aa in seq])
        return True

