# This code is part of the epytope distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
.. module:: TCRSpecificityPrediction
   :synopsis: This module contains all classes for external TCR specificity prediction methods.
.. moduleauthor:: schubert
"""
import logging
import pandas
import pandas as pd
from epytope.Core.Base import ATCRSpecificityPrediction, AExternal
from epytope.Core.AntigenImmuneReceptor import AntigenImmuneReceptor
from epytope.Core.TCREpitope import TCREpitope


class AExternalTCRSpecificityPrediction(ATCRSpecificityPrediction, AExternal):
    """
        Abstract base class for external TCR specificity prediction methods.
        Implements predict functionality.
    """

    def invalid(self, seq: str) -> bool:
        """
        Overwrites ATCRSpecificityPrediction.invalid

        checks if the passed sequence is an invalid protein sequence
        :param str seq: a String representing the protein sequence
        :return: Returns true if the passed sequence is not a protein sequence
        :rtype: bool
        """
        aas = set("ARNDCEQGHILKMFPSTWYV")
        if seq:
            return any([aa not in aas for aa in seq])
        return True

    def canonical_CDR(self, seq: str) -> bool:
        """
        Overwrites ATCRSpecificityPrediction.canonical_CDR

        check if the CDR sequence starts with a Cysteine and ends with a Phenylalanine
        :param str seq: a string representing the CDR sequence
        :return: true if the CDR sequence is canonical
        :rtype: bool
        """
        if seq:
            return seq.startswith("C") and seq.endswith("F")
        return False

    def trimming_cdr3(self, seq: str) -> str:
        """
        Overwrites ATCRSpecificityPrediction.trimming_cdr3

        check if the CDR sequence starts with a Cysteine and ends with a Phenylalanine respectively tryptophan, if this
        is the case remove cysteine and Phenylalanine respectively tryptophan from the start and the end respectively
        :param str seq: a string representing the CDR sequence
        :return: seq[1:-1] if it starts with C and ends with F/W
        :rtype: str
        """
        if seq:
            if seq.startswith("C"):
                seq = seq[1:]
            if seq.endswith("F") or seq.endswith("W"):
                seq = seq[:-1]
        return seq

    def valid_epitope(self, seq: str) -> bool:
        """
        check if the epitope has a specific length and if the sequence is a protein sequence
        :param str seq: a string representing the epitope
        :return: true if the epitope is valid
        :rtype: bool
        """

        return True

    def predict(self, peptides, TCRs, repository: str, all: bool, trained_on: str=None):
        """
        Overwrites ATCRSpecificityPrediction.predict

        Predicts binding probability between a T-cell receptor CDR3 protein sequence and a peptide
        If alleles is not given, predictions for all valid alleles of the predictor is performed. If, however,
        a list of alleles is given, predictions for the valid allele subset is performed.
        :param peptides: The TCREpitope objects for which predictions should be performed
        :type peptides: :class:`~epytope.Core.TCREpitope.TCREpitope` or list(:class:`~epytope.Core.TCREpitope.TCREpitope`)
        :param TCRs: T cell receptor objects
        :type  :class:'~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor' or
        list(:class:'~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor')
        :param str repository: a path to a local github repository of the desired predictor
        :param bool all: if true each TCR object will be joined with each peptide to perform the prediction, otherwise
        the prediction will be preformed in the same order of the passed peptides and TCRs objects
        :param str trained_on: specifying the dataset the model trained on
        :return: Returns a nested dictionary
        {0:{'cdr3.alpha': '...', 'v.alpha': '...', ..., "antigen.epitope": '...'}, 1:{...},...}
        , which contains TCRs with the corresponding epitopes, for which the prediction will
        be performed in the next step with user predefined prediction model
        :rtype: :class:`pandas.DataFrame`
        """
        if isinstance(peptides, TCREpitope):
            peptides = [peptides]
        else:
            if isinstance(peptides, list):
                if len(peptides) == 0:
                    raise ValueError("At least one TCREpitope object should be passed")
        if isinstance(TCRs, AntigenImmuneReceptor):
            TCRs = [TCRs]
        else:
            if isinstance(TCRs, list):
                if len(TCRs) == 0:
                    raise ValueError("At least one AntigenImmuneReceptor object should be passed")
            else:
                raise ValueError(f"TCRs should an AntigenImmuneReceptor object or a list of AntigenImmuneReceptor "
                                 f"objects")
        data = {}
        if all:
            i = 0
            for tcr in TCRs:
                if not isinstance(tcr, AntigenImmuneReceptor):
                    raise ValueError(f"{tcr} is not of type AntigenImmuneReceptor")
                for pep in peptides:
                    if not isinstance(pep, TCREpitope):
                        raise ValueError(f"{pep} is not of type TCREpitope")
                    if pep == "":
                        raise ValueError(f"The peptide sequence should be a sequence of amino acids, not an empty string")
                    # predict binding specificity for each TCR in TCRs to each passed epitope
                    tcr.tcr["Peptide"] = str(pep)
                    tcr.tcr["MHC"] = pep.mhc
                    tcr.tcr["Antigen.species"] = pep.species
                    data[i] = dict(tcr.tcr)
                    i += 1
        else:
            # predict binding specificity in the same passed order
            if len(peptides) < len(TCRs):
                raise ValueError(f"len(TCRs) = {len(TCRs)} != {len(peptides)} = len(peptides). "
                                 f"TCRs(AntigenImmuneReceptor) objects should be as many as the peptides(TCREpitope) "
                                 f"objects, otherwise set the all parameter to True")
            for i, tcr in enumerate(TCRs):
                if not isinstance(tcr, AntigenImmuneReceptor):
                    raise ValueError(f"{tcr} is not of type AntigenImmuneReceptor")
                pep = peptides[i]
                if not isinstance(pep, TCREpitope):
                    raise ValueError(f"{pep} is not of type TCREpitope")
                tcr.tcr["Peptide"] = str(pep)
                tcr.tcr["MHC"] = pep.mhc
                tcr.tcr["Antigen.species"] = pep.species
                data[i] = tcr.tcr

        return pd.DataFrame(data).transpose()









