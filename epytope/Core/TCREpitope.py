# This code is part of the epytope distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
.. module:: Core.Epitope
   :synopsis: Contains the TCREpitope class
   :Note: All internal indices start at 0!
.. moduleauthor:: albahah, drost
"""

from epytope.Core.Base import MetadataLogger


class Epitope(MetadataLogger):
    """
    This class encapsulates a :class:`~epytope.Core.Epitope.Epitope`, belonging to one or several
    :class:`~epytope.Core.Peptide.Peptide`.
    .. note:: For accessing and manipulating the sequence see also :mod:`Bio.Seq.Seq` (from Biopython)
    """

    def __init__(self, peptide, alleles=None, organism=None):
        """
        :param peptide: peptide presenting the amino acid sequence
        :type peptide: :class:`~epytope.Core.Peptide.Peptide`
        :param alleles: Major Histocompatibility Complexes (MHCs) which bound the peptide
        :type alleles: list(:class:`~epytope.Core.Allele.Allele`)
        :param str organism: origin of the peptide sequence
        """
        MetadataLogger.__init__(self)
        self.peptide = peptide
        self.alleles = alleles
        self.organism = organism

    def __repr__(self):
        lines = [f"EPITOPE:\n {self.peptide}"]
        if self.alleles is not None:
            for allele in self.alleles:
                lines.append(f"bound by ALLELE: {allele}")
        return '\n'.join(lines)

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))
