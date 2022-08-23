# This code is part of the epytope distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
.. module:: Core.TCREpitope
   :synopsis: Contains the TCREpitope class
   :Note: All internal indices start at 0!
.. moduleauthor:: schubert
"""

from epytope.Core.Peptide import Peptide


class TCREpitope(Peptide):
    """
    This class encapsulates a :class:`~epytope.Core.TCREpitope.TCREpitope`, belonging to one or several
    :class:`~epytope.Core.Peptide.Peptide`.
    .. note:: For accessing and manipulating the sequence see also :mod:`Bio.Seq.Seq` (from Biopython)
    """

    def __init__(self, seq, protein_pos=None, mhc: str = None, species: str = None, epitope_id: str = None):
        """
        :param str seq: Sequence of the peptide in one letter amino acid code
        :param protein_pos: Dict of transcript_IDs to position of origin in protein
        :type protein_pos: dict(:class:`~epytope.Core.Protein.Protein`,list(int))`
        :param str mhc: major histo compatibility complex to which this chain binds
        :param str species: a string representing the species of the epitope
        :param str epitope_id: a string representing the epitope ID
        """
        if self.invalid(seq):
            raise ValueError(f"{seq} is an invalid protein sequence")
        super().__init__(seq, protein_pos)
        self.mhc = mhc
        self.species = species
        self.epitope_id = epitope_id

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
