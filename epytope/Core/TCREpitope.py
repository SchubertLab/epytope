# This code is part of the epytope distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
.. module:: Core.TCREpitope
   :synopsis: Contains the TCREpitope class
   :Note: All internal indices start at 0!
.. moduleauthor:: albahah, drost
"""

from epytope.Core.Base import MetadataLogger


class TCREpitope(MetadataLogger):
    """
    This class encapsulates a :class:`~epytope.Core.TCREpitope.TCREpitope`, belonging to one or several
    :class:`~epytope.Core.Peptide.Peptide`.
    .. note:: For accessing and manipulating the sequence see also :mod:`Bio.Seq.Seq` (from Biopython)
    """

    def __init__(self, peptide, allele=None, organism=None):
        """
        :param peptide: peptide presenting the amino acid sequence
        :type peptide: :class:`~epytope.Core.Peptide.Peptide`
        :param allele: Major Histocompatibility Complex (MHCs) which bound the peptide
        :type allele: :class:`~epytope.Core.Allele.Allele`
        :param str organism: origin of the peptide sequence
        """
        MetadataLogger.__init__(self)
        self.peptide = peptide
        self.allele = allele if allele != "" else None
        self.organism = organism

    def __repr__(self):
        lines = [f"TCR EPITOPE:\n PEPTIDE {self.peptide}"]
        if self.allele is not None:
            line = f" bound by ALLELE: {self.allele}"
            lines.append(line)
        return "\n".join(lines)

    def __str__(self):
        if self.allele is None:
            return self.peptide
        return f"{self.peptide}, {self.allele}"

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))
