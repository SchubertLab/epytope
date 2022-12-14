# coding=utf-8
# This code is part of the epytope distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
.. module:: EpitopePrediction.PSSM
   :synopsis: This module contains all classes for PSSM-based epitope prediction.
.. moduleauthor:: schubert

"""
import itertools
import warnings
import math
import subprocess
import csv
import tempfile
import pandas
import logging

from epytope.Core.Allele import Allele
from epytope.Core.Peptide import Peptide
from epytope.Core.Result import EpitopePredictionResult
from epytope.Core.Base import AEpitopePrediction

from mhcflurry import Class1AffinityPredictor
from mhcnuggets.src.predict import predict


class APSSMEpitopePrediction(AEpitopePrediction):
    """
        Abstract base class for PSSM predictions.
        Implements predict functionality
    """

    def predict(self, peptides, alleles=None, **kwargs):
        """
        Returns predictions for given peptides an :class:`~epytope.Core.Allele.Allele`. If no
        :class:`~epytope.Core.Allele.Allele` are given, predictions for all available models are made.

        :param peptides: A single :class:`~epytope.Core.Peptide.Peptide` or a list of :class:`~epytope.Core.Peptide.Peptide`
        :type peptides: list(:class:`~epytope.Core.Peptide.Peptide`) or :class:`~epytope.Core.Peptide.Peptide`
        :param alleles: A list of :class:`~epytope.Core.Allele.Allele`
        :type alleles: list(:class:`~epytope.Core.Allele.Allele`) or class:`~epytope.Core.Allele.Allele`
        :param kwargs: optional parameter (not used yet)
        :return: Returns a :class:`~epytope.Core.Result.EpitopePredictionResult` object with the prediction results
        :rtype: :class:`~epytope.Core.Result.EpitopePredictionResult`
        """

        def __load_allele_model(allele, length):
            allele_model = "%s_%i" % (allele, length)
            return getattr(
                __import__("epytope.Data.pssms." + self.name + ".mat." + allele_model, fromlist=[allele_model]),
                allele_model)

        if isinstance(peptides, Peptide):
            pep_seqs = {str(peptides): peptides}
        else:
            pep_seqs = {}
            for p in peptides:
                if not isinstance(p, Peptide):
                    raise ValueError("Input is not of type Protein or Peptide")
                pep_seqs[str(p)] = p

        if alleles is None:
            al = [Allele(a) for a in self.supportedAlleles]
            alleles_string = {conv_a: a for conv_a, a in zip(self.convert_alleles(al), al)}
        else:
            if isinstance(alleles, Allele):
                alleles = [alleles]
            if any(not isinstance(p, Allele) for p in alleles):
                raise ValueError("Input is not of type Allele")
            alleles_string = {conv_a: a for conv_a, a in zip(self.convert_alleles(alleles), alleles)}

        result = {}
        pep_groups = list(pep_seqs.keys())
        pep_groups.sort(key=len)
        for length, peps in itertools.groupby(pep_groups, key=len):
            peps = list(peps)
            # dynamicaly import prediction PSSMS for alleles and predict
            if self.supportedLength is not None and length not in self.supportedLength:
                warnings.warn("Peptide length of %i is not supported by %s" % (length, self.name))
                continue
            
            for a in alleles_string.keys():
                try:
                    pssm = __load_allele_model(a, length)
                except ImportError:
                    warnings.warn("No model found for %s with length %i" % (alleles_string[a], length))
                    continue

                if alleles_string[a] not in result:
                    result[alleles_string[a]] = {'Score': {}}
                
                for p in peps:
                    score = sum(pssm[i].get(p[i], 0.0) for i in range(length)) + pssm.get(-1, {}).get("con", 0)
                    result[alleles_string[a]]['Score'][pep_seqs[p]] = score

        if not result:
            raise ValueError("No predictions could be made with "
                             + self.name + " for given input. Check your epitope length and HLA allele combination.")
        
        df_result = EpitopePredictionResult.from_dict(result, pep_seqs.values(), self.name)
        
        return df_result


class Syfpeithi(APSSMEpitopePrediction):
    """
    Represents the Syfpeithi PSSM predictor.

    .. note::

        Rammensee, H. G., Bachmann, J., Emmerich, N. P. N., Bachor, O. A., & Stevanovic, S. (1999).
        SYFPEITHI: database for MHC ligands and peptide motifs. Immunogenetics, 50(3-4), 213-219.
    """
    __alleles = frozenset(
        ['HLA-A*01:01','HLA-A*02:01','HLA-A*03:01','HLA-A*11:01','HLA-A*24:02','HLA-A*26:01','HLA-A*31:01','HLA-A*32:01','HLA-A*33:01','HLA-A*66:01','HLA-A*68:01','HLA-B*07:02','HLA-B*08:01','HLA-B*13:02','HLA-B*14:02','HLA-B*15:01','HLA-B*15:16','HLA-B*18:01','HLA-B*27:05','HLA-B*27:09','HLA-B*35:01','HLA-B*37:01','HLA-B*38:01','HLA-B*39:01','HLA-B*39:02','HLA-B*40:01','HLA-B*40:02','HLA-B*41:01','HLA-B*44:02','HLA-B*45:01','HLA-B*47:01','HLA-B*49:01','HLA-B*50:01','HLA-B*51:01','HLA-B*53:01','HLA-B*57:01','HLA-B*58:01','HLA-B*58:02','HLA-C*01:02','HLA-C*02:02','HLA-C*03:03','HLA-C*03:04','HLA-C*04:01','HLA-C*05:01','HLA-C*06:01','HLA-C*06:02','HLA-C*07:01','HLA-C*07:02','HLA-C*08:02','HLA-C*12:03','HLA-C*14:02','HLA-C*15:02','HLA-C*16:01','HLA-C*17:01','HLA-DRB1*01:01','HLA-DRB1*03:01','HLA-DRB1*07:01','HLA-DRB1*11:01','HLA-DRB1*15:01','HLA-G*01:01','H-2-Db','H-2-Kb','H-2-Kd','H-2-Ld'])
    __supported_length = frozenset([8, 9, 10, 11, 12, 13])
    __name = "syfpeithi"
    __version = "1.0"

    @property
    def version(self):
        """The version of the predictor"""
        return self.__version

    @property
    def name(self):
        """The name of the predictor"""
        return self.__name

    @property
    def supportedAlleles(self):
        """
        A list of supported :class:`~epytope.Core.Allele.Allele`
        """
        return self.__alleles

    @property
    def supportedLength(self):
        """
        A list of supported :class:`~epytope.Core.Peptide.Peptide` lengths
        """
        return self.__supported_length

    def convert_alleles(self, alleles):
        """
        Converts :class:`~epytope.Core.Allele.Allele` into the internal :class:`~epytope.Core.Allele.Allele`
        representation of the predictor and returns a string representation

        :param alleles: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is needed
        :type alleles: list(:class:`~epytope.Core.Allele.Allele`)
        :return: Returns a string representation of the input :class:`~epytope.Core.Allele.Allele`
        :rtype: list(str)
        """
        return ["%s_%s%s" % (a.locus, a.supertype, a.subtype) for a in alleles]


class BIMAS(APSSMEpitopePrediction):
    """
    Represents the BIMAS PSSM predictor.

    .. note::

        Parker, K.C., Bednarek, M.A. and Coligan, J.E. Scheme for ranking potential HLA-A2 binding peptides based on
        independent binding of individual peptide side-chains. The Journal of Immunology 1994;152(1):163-175.
    """

    __alleles = frozenset(['HLA-B*04:01', 'HLA-A*31:01', 'HLA-B*58:01', 'HLA-C*06:02', 'HLA-A*03:01', 'HLA-B*35:01',
                           'HLA-B*35:01', 'HLA-B*15:01', 'HLA-A*02:05', 'HLA-B*27:05', 'HLA-B*27:05', 'HLA-A*33:02',
                           'HLA-B*39:01', 'HLA-B*38:01', 'HLA-B*40:', 'HLA-A*24:02', 'HLA-B*51:01', 'HLA-B*07:02',
                           'HLA-B*08:01', 'HLA-B*51:02', 'HLA-B*40:06', 'HLA-B*40:06', 'HLA-B*51:02', 'HLA-B*37:01',
                           'HLA-A*11:01', 'HLA-B*08:01', 'HLA-B*44:03', 'HLA-A*68:01', 'HLA-B*51:03', 'HLA-B*52:01',
                           'HLA-A*02:01', 'HLA-A*01:01', 'HLA-C*07:02', 'HLA-C*03:01', 'HLA-B*40:01', 'HLA-B*51:01',
                           'HLA-B*39:02', 'HLA-B*52:01', 'HLA-C*04:01', 'HLA-B*27:02', 'HLA-B*39:01'])
    __supported_length = frozenset([8, 9])
    __name = "bimas"
    __version = "1.0"

    @property
    def version(self):
        """The version of the predictor"""
        return self.__version

    @property
    def name(self):
        """The name of the predictor"""
        return self.__name

    @property
    def supportedAlleles(self):
        """
        A list of supported :class:`~epytope.Core.Allele.Allele`
        """
        return self.__alleles

    @property
    def supportedLength(self):
        """
        A list of supported :class:`~epytope.Core.Peptide.Peptide` lengths
        """
        return self.__supported_length

    def convert_alleles(self, alleles):
        """
        Converts :class:`~epytope.Core.Allele.Allele` into the internal :class:`~epytope.Core.Allele.Allele`
        representation of the predictor and returns a string representation

        :param alleles: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is needed
        :type alleles: list(:class:`~epytope.Core.Allele.Allele`)
        :return: Returns a string representation of the input :class:`~epytope.Core.Allele.Allele`
        :rtype: list(str)
        """
        return ["%s_%s%s" % (a.locus, a.supertype, a.subtype) for a in alleles]

    def predict(self, peptides, alleles=None, **kwargs):
        """
        Returns predictions for given peptides an :class:`~epytope.Core.Allele.Allele`. If no
        :class:`~epytope.Core.Allele.Allele` are given, predictions for all available models are made.

        :param peptides: A single :class:`~epytope.Core.Peptide.Peptide` or a list of :class:`~epytope.Core.Peptide.Peptide`
        :type peptides: list(:class:`~epytope.Core.Peptide.Peptide`) or :class:`~epytope.Core.Peptide.Peptide`
        :param alleles: A list of :class:`~epytope.Core.Allele.Allele`
        :type alleles: list(:class:`~epytope.Core.Allele.Allele`) or class:`~epytope.Core.Allele.Allele`
        :param kwargs: optional parameter (not used yet)
        :return: Returns a :class:`~epytope.Core.Result.EpitopePredictionResult` object with the prediction results
        :rtype: :class:`~epytope.Core.Result.EpitopePredictionResult`
        """
        return EpitopePredictionResult(
            super(BIMAS, self).predict(peptides, alleles=alleles,
                                       **kwargs).applymap(lambda x: math.pow(math.e, x)))


class Epidemix(APSSMEpitopePrediction):
    """
    Represents the Epidemix PSSM predictor.

    .. note::

        Feldhahn, M., et al. FRED-a framework for T-cell epitope detection. Bioinformatics 2009;25(20):2758-2759.
    """
    __alleles = frozenset(
        ['HLA-B*27', 'HLA-A*11:01', 'HLA-B*27:05', 'HLA-B*07', 'HLA-B*27', 'HLA-A*01', 'HLA-B*44', 'HLA-A*03',
         'HLA-A*25', 'HLA-B*37:01', 'HLA-A*02:01', 'HLA-A*02:01', 'HLA-B*18:01', 'HLA-B*18:01', 'HLA-A*03',
         'HLA-A*24', 'HLA-A*25', 'HLA-A*02:01', 'HLA-A*11:01', 'HLA-A*24:02', 'HLA-B*08', 'HLA-B*08',
         'HLA-B*51:01', 'HLA-B*51:01'])
    __supported_length = frozenset([9, 10, 8, 11])
    __name = "epidemix"
    __version = "1.0"

    @property
    def version(self):
        """The version of the predictor"""
        return self.__version

    @property
    def name(self):
        """The name of the predictor"""
        return self.__name

    @property
    def supportedAlleles(self):
        """
        A list of supported :class:`~epytope.Core.Allele.Allele`
        """
        return self.__alleles

    @property
    def supportedLength(self):
        """
        A list of supported :class:`~epytope.Core.Peptide.Peptide` lengths
        """
        return self.__supported_length

    def convert_alleles(self, alleles):
        """
        Converts :class:`~epytope.Core.Allele.Allele` into the internal :class:`~epytope.Core.Allele.Allele`
        representation of the predictor and returns a string representation

        :param alleles: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is needed
        :type alleles: list(:class:`~epytope.Core.Allele.Allele`)
        :return: Returns a string representation of the input :class:`~epytope.Core.Allele.Allele`
        :rtype: list(str)
        """
        return ["%s_%s%s" % (a.locus, a.supertype, a.subtype) for a in alleles]


class Hammer(APSSMEpitopePrediction):
    """
    Represents the virtual pockets approach by Sturniolo et al.

    .. note::

        Sturniolo, T., et al. Generation of tissue-specific and promiscuous HLA ligand databases using DNA microarrays
        and virtual HLA class II matrices. Nature biotechnology 1999;17(6):555-561.
    """
    __alleles = frozenset(
        ['HLA-DRB1*07:03', 'HLA-DRB1*07:01', 'HLA-DRB1*11:28', 'HLA-DRB1*11:21', 'HLA-DRB1*11:20', 'HLA-DRB1*04:26',
         'HLA-DRB1*04:23', 'HLA-DRB1*04:21', 'HLA-DRB5*01:05:', 'HLA-DRB1*08:17', 'HLA-DRB1*13:05', 'HLA-DRB1*13:04',
         'HLA-DRB1*13:07', 'HLA-DRB1*13:01', 'HLA-DRB1*13:02', 'HLA-DRB1*08:04', 'HLA-DRB1*08:06', 'HLA-DRB1*08:01',
         'HLA-DRB1*01:01', 'HLA-DRB1*01:02', 'HLA-DRB1*08:02', 'HLA-DRB1*13:11', 'HLA-DRB1*03:11', 'HLA-DRB1*11:07',
         'HLA-DRB1*11:06', 'HLA-DRB1*11:04', 'HLA-DRB1*11:02', 'HLA-DRB1*11:01', 'HLA-DRB1*04:08', 'HLA-DRB1*04:01',
         'HLA-DRB1*04:02', 'HLA-DRB1*04:05', 'HLA-DRB1*04:04', 'HLA-DRB1*13:23', 'HLA-DRB1*13:22', 'HLA-DRB1*13:21',
         'HLA-DRB1*13:27', 'HLA-DRB1*08:13', 'HLA-DRB1*13:28', 'HLA-DRB1*03:06', 'HLA-DRB1*03:07', 'HLA-DRB1*03:05',
         'HLA-DRB1*11:14', 'HLA-DRB1*03:01', 'HLA-DRB1*15:02', 'HLA-DRB1*15:01', 'HLA-DRB1*15:06', 'HLA-DRB1*03:08',
         'HLA-DRB1*03:09', 'HLA-DRB1*04:10', 'HLA-DRB5*01:01'])
    __supported_length = frozenset([9])
    __name = "hammer"
    __version = "1.0"

    @property
    def version(self):
        """The version of the predictor"""
        return self.__version

    @property
    def name(self):
        """The name of the predictor"""
        return self.__name

    @property
    def supportedAlleles(self):
        """
        A list of supported :class:`~epytope.Core.Allele.Allele`
        """
        return self.__alleles

    @property
    def supportedLength(self):
        """
        A list of supported :class:`~epytope.Core.Peptide.Peptide` lengths
        """
        return self.__supported_length

    def convert_alleles(self, alleles):
        """
        Converts :class:`~epytope.Core.Allele.Allele` into the internal :class:`~epytope.Core.Allele.Allele`
        representation of the predictor and returns a string representation

        :param alleles: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is needed
        :type alleles: list(:class:`~epytope.Core.Allele.Allele`)
        :return: Returns a string representation of the input :class:`~epytope.Core.Allele.Allele`
        :rtype: list(str)
        """
        return ["%s_%s%s" % (a.locus, a.supertype, a.subtype) for a in alleles]


class SMM(APSSMEpitopePrediction):
    """
    Implements IEDBs SMM PSSM method.

    .. note::

        Peters B, Sette A. 2005. Generating quantitative models describing the sequence specificity of
        biological processes with the stabilized matrix method. BMC Bioinformatics 6:132.
    """

    __alleles = frozenset(
        ['HLA-B*27:20', 'HLA-B*83:01', 'HLA-A*32:15', 'HLA-B*15:17', 'HLA-B*40:13', 'HLA-A*24:02', 'HLA-A*24:03',
         'HLA-B*53:01', 'HLA-B*15:01', 'HLA-B*27:05', 'HLA-B*42:01', 'HLA-B*39:01', 'HLA-B*38:01', 'HLA-A*23:01',
         'HLA-A*25:01', 'HLA-C*04:01', 'HLA-A*29:02', 'HLA-A*02:06', 'HLA-A*02:01', 'HLA-A*02:02', 'HLA-A*02:03',
         'HLA-A*26:02', 'HLA-A*26:03', 'HLA-A*26:01', 'HLA-C*03:03', 'E*01:01', 'E*01:03', 'HLA-B*58:01', 'HLA-A*31:01',
         'HLA-C*06:02', 'HLA-B*07:02', 'HLA-A*66:01', 'HLA-B*57:01', 'HLA-A*68:01', 'HLA-A*68:02', 'HLA-C*14:02',
         'HLA-B*35:01', 'HLA-B*15:09', 'HLA-B*35:03', 'HLA-A*80:01', 'HLA-B*15:03', 'HLA-B*15:02', 'HLA-A*32:01',
         'HLA-A*02:50', 'HLA-A*32:07', 'HLA-B*58:02', 'HLA-A*69:01', 'HLA-A*68:23', 'HLA-A*11:01', 'HLA-A*03:01',
         'HLA-B*73:01', 'HLA-B*40:01', 'HLA-B*44:03', 'HLA-B*46:01', 'HLA-B*40:02', 'HLA-C*12:03', 'HLA-B*44:02',
         'HLA-A*30:01', 'HLA-A*02:19', 'HLA-A*30:02', 'HLA-A*02:17', 'HLA-A*02:16', 'HLA-B*51:01', 'HLA-B*45:01',
         'HLA-A*02:12', 'HLA-A*02:11', 'HLA-B*54:01', 'HLA-B*08:01', 'HLA-B*18:01', 'HLA-B*08:03', 'HLA-B*08:02',
         'HLA-C*05:01', 'HLA-C*15:02', 'HLA-A*33:01', 'HLA-B*14:02', 'HLA-C*07:01', 'HLA-B*48:01', 'HLA-B*15:42',
         'HLA-C*07:02', 'HLA-A*01:01', 'HLA-C*08:02'])
    __supported_length = frozenset([8, 9, 10, 11])
    __name = "smm"
    __version = "1.0"

    @property
    def version(self):
        """The version of the predictor"""
        return self.__version

    @property
    def name(self):
        """The name of the predictor"""
        return self.__name

    @property
    def supportedAlleles(self):
        """
        A list of supported :class:`~epytope.Core.Allele.Allele`
        """
        return self.__alleles

    @property
    def supportedLength(self):
        """
        A list of supported :class:`~epytope.Core.Peptide.Peptide` lengths
        """
        return self.__supported_length

    def convert_alleles(self, alleles):
        """
        Converts :class:`~epytope.Core.Allele.Allele` into the internal :class:`~epytope.Core.Allele.Allele`
        representation of the predictor and returns a string representation

        :param alleles: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is needed
        :type alleles: list(:class:`~epytope.Core.Allele.Allele`)
        :return: Returns a string representation of the input :class:`~epytope.Core.Allele.Allele`
        :rtype: list(str)
        """
        return ["%s_%s_%s" % (a.locus, a.supertype, a.subtype) for a in alleles]

    def predict(self, peptides, alleles=None, **kwargs):
        """
        Returns predictions for given peptides an :class:`~epytope.Core.Allele.Allele`. If no
        :class:`~epytope.Core.Allele.Allele` are given, predictions for all available models are made.

        :param peptides: A single :class:`~epytope.Core.Peptide.Peptide` or a list of :class:`~epytope.Core.Peptide.Peptide`
        :type peptides: list(:class:`~epytope.Core.Peptide.Peptide`) or :class:`~epytope.Core.Peptide.Peptide`
        :param alleles: A list of :class:`~epytope.Core.Allele.Allele`
        :type alleles: list(:class:`~epytope.Core.Allele.Allele`) or class:`~epytope.Core.Allele.Allele`
        :param kwargs: optional parameter (not used yet)
        :return: Returns a :class:`~epytope.Core.Result.EpitopePredictionResult` object with the prediction results
        :rtype: :class:`~epytope.Core.Result.EpitopePredictionResult`
        """
        return EpitopePredictionResult(
            super(SMM, self).predict(peptides, alleles=alleles, **kwargs).applymap(lambda x: math.pow(10, x)))


class SMMPMBEC(APSSMEpitopePrediction):
    """
    Implements IEDBs SMMPMBEC PSSM method.

    .. note::

        Kim, Y., Sidney, J., Pinilla, C., Sette, A., & Peters, B. (2009). Derivation of an amino acid similarity matrix
        for peptide: MHC binding and its application as a Bayesian prior. BMC Bioinformatics, 10(1), 394.
    """

    __alleles = frozenset(
        ['HLA-A*01:01', 'HLA-A*02:01', 'HLA-A*02:02', 'HLA-A*02:03', 'HLA-A*02:06', 'HLA-A*02:11', 'HLA-A*02:12',
         'HLA-A*02:16', 'HLA-A*02:17', 'HLA-A*02:19', 'HLA-A*02:50', 'HLA-A*03:01', 'HLA-A*11:01', 'HLA-A*23:01',
         'HLA-A*24:02', 'HLA-A*24:03', 'HLA-A*25:01', 'HLA-A*26:01', 'HLA-A*26:02', 'HLA-A*26:03', 'HLA-A*29:02',
         'HLA-A*30:01', 'HLA-A*30:02', 'HLA-A*31:01', 'HLA-A*32:01', 'HLA-A*32:07', 'HLA-A*32:15', 'HLA-A*33:01',
         'HLA-A*66:01', 'HLA-A*68:01', 'HLA-A*68:02', 'HLA-A*68:23', 'HLA-A*69:01', 'HLA-A*80:01', 'HLA-B*07:02',
         'HLA-B*08:01', 'HLA-B*08:02', 'HLA-B*08:03', 'HLA-B*14:02', 'HLA-B*15:01', 'HLA-B*15:02', 'HLA-B*15:03',
         'HLA-B*15:09', 'HLA-B*15:17', 'HLA-B*15:42', 'HLA-B*18:01', 'HLA-B*27:03', 'HLA-B*27:05', 'HLA-B*27:20',
         'HLA-B*35:01', 'HLA-B*35:03', 'HLA-B*38:01', 'HLA-B*39:01', 'HLA-B*40:01', 'HLA-B*40:02', 'HLA-B*40:13',
         'HLA-B*42:01', 'HLA-B*44:02', 'HLA-B*44:03', 'HLA-B*45:01', 'HLA-B*45:06', 'HLA-B*46:01', 'HLA-B*48:01',
         'HLA-B*51:01', 'HLA-B*53:01', 'HLA-B*54:01', 'HLA-B*57:01', 'HLA-B*58:01', 'HLA-B*58:02', 'HLA-B*73:01',
         'HLA-B*83:01', 'HLA-C*03:03', 'HLA-C*04:01', 'HLA-C*05:01', 'HLA-C*06:02', 'HLA-C*07:01', 'HLA-C*07:02',
         'HLA-C*08:02', 'HLA-C*12:03', 'HLA-C*14:02', 'HLA-C*15:02', 'E*01:01', 'E*01:03'])
    __supported_length = frozenset([8, 9, 10, 11])
    __name = "smmpmbec"
    __version = "1.0"

    @property
    def version(self):
        """The version of the predictor"""
        return self.__version

    @property
    def name(self):
        """The name of the predictor"""
        return self.__name

    @property
    def supportedAlleles(self):
        """
        A list of supported :class:`~epytope.Core.Allele.Allele`
        """
        return self.__alleles

    @property
    def supportedLength(self):
        """
        A list of supported :class:`~epytope.Core.Peptide.Peptide` lengths
        """
        return self.__supported_length

    def convert_alleles(self, alleles):
        """
        Converts :class:`~epytope.Core.Allele.Allele` into the internal :class:`~epytope.Core.Allele.Allele`
        representation of the predictor and returns a string representation

        :param alleles: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is needed
        :type alleles: list(:class:`~epytope.Core.Allele.Allele`)
        :return: Returns a string representation of the input :class:`~epytope.Core.Allele.Allele`
        :rtype: list(str)
        """
        return ["%s_%s_%s" % (a.locus, a.supertype, a.subtype) for a in alleles]

    def predict(self, peptides, alleles=None, **kwargs):
        """
        Returns predictions for given peptides an :class:`~epytope.Core.Allele.Allele`. If no
        :class:`~epytope.Core.Allele.Allele` are given, predictions for all available models are made.

        :param peptides: A single :class:`~epytope.Core.Peptide.Peptide` or a list of :class:`~epytope.Core.Peptide.Peptide`
        :type peptides: list(:class:`~epytope.Core.Peptide.Peptide`) or :class:`~epytope.Core.Peptide.Peptide`
        :param alleles: A list of :class:`~epytope.Core.Allele.Allele`
        :type alleles: list(:class:`~epytope.Core.Allele.Allele`) or class:`~epytope.Core.Allele.Allele`
        :param kwargs: optional parameter (not used yet)
        :return: Returns a :class:`~epytope.Core.Result.EpitopePredictionResult` object with the prediction results
        :rtype: :class:`~epytope.Core.Result.EpitopePredictionResult`
        """
        return EpitopePredictionResult(
            super(SMMPMBEC, self).predict(peptides, alleles=alleles, **kwargs).applymap(lambda x: math.pow(10, x)))


class ARB(APSSMEpitopePrediction):
    """
    Implements IEDBs ARB method.

    .. note::

        Bui HH, Sidney J, Peters B, Sathiamurthy M, Sinichi A, Purton KA, Mothe BR, Chisari FV, Watkins DI, Sette A.
        2005. Automated generation and evaluation of specific MHC binding predictive tools: ARB matrix applications.
        Immunogenetics 57:304-314.
    """

    __alleles = frozenset(
        ['HLA-A*01:01', 'HLA-A*02:01', 'HLA-A*02:02', 'HLA-A*02:03', 'HLA-A*02:06', 'HLA-A*02:11', 'HLA-A*02:12',
         'HLA-A*02:16', 'HLA-A*02:19', 'HLA-A*02:50', 'HLA-A*03:01', 'HLA-A*11:01', 'HLA-A*23:01', 'HLA-A*24:02',
         'HLA-A*24:03', 'HLA-A*25:01', 'HLA-A*26:01', 'HLA-A*26:02', 'HLA-A*26:03', 'HLA-A*29:02', 'HLA-A*30:01',
         'HLA-A*30:02', 'HLA-A*31:01', 'HLA-A*32:01', 'HLA-A*33:01', 'HLA-A*68:01', 'HLA-A*68:02', 'HLA-A*69:01',
         'HLA-A*80:01', 'HLA-B*07:02', 'HLA-B*08:01', 'HLA-B*08:02', 'HLA-B*08:03', 'HLA-B*15:01', 'HLA-B*15:02',
         'HLA-B*15:03', 'HLA-B*15:09', 'HLA-B*15:17', 'HLA-B*18:01', 'HLA-B*27:03', 'HLA-B*27:05', 'HLA-B*35:01',
         'HLA-B*38:01', 'HLA-B*39:01', 'HLA-B*40:01', 'HLA-B*40:02', 'HLA-B*44:02', 'HLA-B*44:03', 'HLA-B*45:01',
         'HLA-B*46:01', 'HLA-B*48:01', 'HLA-B*51:01', 'HLA-B*53:01', 'HLA-B*54:01', 'HLA-B*57:01', 'HLA-B*58:01',
         'HLA-B*73:01'])
    __supported_length = frozenset([8, 9, 10, 11])
    __name = "arb"
    __version = "1.0"

    @property
    def version(self):
        """The version of the predictor"""
        return self.__version

    @property
    def name(self):
        """The name of the predictor"""
        return self.__name

    @property
    def supportedAlleles(self):
        """
        A list of supported :class:`~epytope.Core.Allele.Allele`
        """
        return self.__alleles

    @property
    def supportedLength(self):
        """
        A list of supported :class:`~epytope.Core.Peptide.Peptide` lengths
        """
        return self.__supported_length

    def convert_alleles(self, alleles):
        """
        Converts :class:`~epytope.Core.Allele.Allele` into the internal :class:`~epytope.Core.Allele.Allele`
        representation of the predictor and returns a string representation

        :param alleles: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is needed
        :type alleles: list(:class:`~epytope.Core.Allele.Allele`)
        :return: Returns a string representation of the input :class:`~epytope.Core.Allele.Allele`
        :rtype: list(str)
        """
        return ["%s_%s%s" % (a.locus, a.supertype, a.subtype) for a in alleles]

    def predict(self, peptides, alleles=None, **kwargs):
        """
        Returns predictions for given peptides an :class:`~epytope.Core.Allele.Allele`. If no
        :class:`~epytope.Core.Allele.Allele` are given, predictions for all available models are made.

        :param peptides: A single :class:`~epytope.Core.Peptide.Peptide` or a list of :class:`~epytope.Core.Peptide.Peptide`
        :type peptides: list(:class:`~epytope.Core.Peptide.Peptide`) or :class:`~epytope.Core.Peptide.Peptide`
        :param alleles: A list of :class:`~epytope.Core.Allele.Allele`
        :type alleles: list(:class:`~epytope.Core.Allele.Allele`) or class:`~epytope.Core.Allele.Allele`
        :param kwargs: optional parameter (not used yet)
        :return: Returns a :class:`~epytope.Core.Result.EpitopePredictionResult` object with the prediction results
        :rtype: :class:`~epytope.Core.Result.EpitopePredictionResult`
        """

        def __load_allele_model(allele, length):
            allele_model = "%s_%i" % (allele, length)
            return getattr(
                __import__("epytope.Data.pssms." + self.name + ".mat." + allele_model, fromlist=[allele_model]),
                allele_model)

        if isinstance(peptides, Peptide):
            pep_seqs = {str(peptides): peptides}
        else:
            pep_seqs = {}
            for p in peptides:
                if not isinstance(p, Peptide):
                    raise ValueError("Input is not of type Protein or Peptide")
                pep_seqs[str(p)] = p

        if alleles is None:
            al = [Allele(a) for a in self.supportedAlleles]
            alleles_string = {conv_a: a for conv_a, a in zip(self.convert_alleles(al), al)}
        else:
            if isinstance(alleles, Allele):
                alleles = [alleles]
            if any(not isinstance(p, Allele) for p in alleles):
                raise ValueError("Input is not of type Allele")
            alleles_string = {conv_a: a for conv_a, a in zip(self.convert_alleles(alleles), alleles)}

        scores = {}
        for length, peps in itertools.groupby(pep_seqs.keys(), key=lambda x: len(x)):
            peps = list(peps)
            # dynamicaly import prediction PSSMS for alleles and predict
            if length not in self.supportedLength:
                warnings.warn("Peptide length of %i is not supported by %s" % (length, self.name))
                continue

            for a in alleles_string.keys():
                try:
                    pssm = __load_allele_model(a, length)
                except ImportError:
                    warnings.warn("No model found for %s with length %i" % (alleles_string[a], length))
                    continue

                scores[alleles_string[a]] = {}
                ##here is the prediction and result object missing##
                for p in peps:
                    score = sum(pssm[i].get(p[i], 0.0) for i in range(length)) + pssm.get(-1, {}).get("con", 0)
                    score /= -length
                    score -= pssm[-1]["intercept"]
                    score /= pssm[-1]["slope"]
                    score = math.pow(10, score)
                    if score < 0.0001:
                        score = 0.0001
                    elif score > 1e6:
                        score = 1e6
                    scores[alleles_string[a]][pep_seqs[p]] = score

        if not scores:
            raise ValueError("No predictions could be made with " + self.name + " for given input. Check your"
                                                                                "epitope length and HLA allele combination.")
        
        result = {allele: {"Score":(list(scores.values())[j])} for j, allele in enumerate(alleles)}

        df_result = EpitopePredictionResult.from_dict(result, peps, self.name)
        return df_result


class ComblibSidney2008(APSSMEpitopePrediction):
    """
    Implements IEDBs Comblib_Sidney2008 PSSM method.

    .. note::

        Sidney J, Assarsson E, Moore C, Ngo S, Pinilla C, Sette A, Peters B. 2008. Quantitative peptide binding motifs
        for 19 human and mouse MHC class I molecules derived using positional scanning combinatorial peptide libraries.
        Immunome Res 4:2.
    """

    __alleles = frozenset(
        ['HLA-B*35:01', 'HLA-B*51:01', 'HLA-B*54:01', 'HLA-B*58:02', 'HLA-A*02:01', 'HLA-A*68:02', 'HLA-B*27:05',
         'HLA-B*08:01', 'HLA-B*07:02', 'HLA-A*32:01', 'HLA-B*53:01', 'HLA-A*30:01', 'HLA-B*15:03', 'HLA-B*15:01',
         'HLA-B*58:01'])
    __supported_length = frozenset([9])
    __name = "comblibsidney"
    __version = "1.0"

    @property
    def version(self):
        """The version of the predictor"""
        return self.__version

    @property
    def name(self):
        """The name of the predictor"""
        return self.__name

    @property
    def supportedAlleles(self):
        """
        A list of supported :class:`~epytope.Core.Allele.Allele`
        """
        return self.__alleles

    @property
    def supportedLength(self):
        """
        A list of supported :class:`~epytope.Core.Peptide.Peptide` lengths
        """
        return self.__supported_length

    def convert_alleles(self, alleles):
        """
        Converts :class:`~epytope.Core.Allele.Allele` into the internal :class:`~epytope.Core.Allele.Allele`
        representation of the predictor and returns a string representation

        :param alleles: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is needed
        :type alleles: list(:class:`~epytope.Core.Allele.Allele`)
        :return: Returns a string representation of the input :class:`~epytope.Core.Allele.Allele`
        :rtype: list(str)
        """
        return ["%s_%s%s" % (a.locus, a.supertype, a.subtype) for a in alleles]

    def predict(self, peptides, alleles=None, **kwargs):
        """
        Returns predictions for given peptides an :class:`~epytope.Core.Allele.Allele`. If no
        :class:`~epytope.Core.Allele.Allele` are given, predictions for all available models are made.

        :param peptides: A single :class:`~epytope.Core.Peptide.Peptide` or a list of :class:`~epytope.Core.Peptide.Peptide`
        :type peptides: list(:class:`~epytope.Core.Peptide.Peptide`) or :class:`~epytope.Core.Peptide.Peptide`
        :param alleles: A list of :class:`~epytope.Core.Allele.Allele`
        :type alleles: list(:class:`~epytope.Core.Allele.Allele`) or class:`~epytope.Core.Allele.Allele`
        :param kwargs: optional parameter (not used yet)
        :return: Returns a :class:`~epytope.Core.Result.EpitopePredictionResult` object with the prediction results
        :rtype: :class:`~epytope.Core.Result.EpitopePredictionResult`
        """
        return EpitopePredictionResult(
            super(ComblibSidney2008, self).predict(peptides,
                                                   alleles=alleles,
                                                   **kwargs).applymap(lambda x: math.pow(10, x)))


class TEPITOPEpan(APSSMEpitopePrediction):
    """
    Implements TEPITOPEpan.

    .. note::

        TEPITOPEpan: Extending TEPITOPE for Peptide Binding Prediction Covering over 700 HLA-HLA-DR Molecules
        Zhang L, Chen Y, Wong H-S, Zhou S, Mamitsuka H, et al. (2012) TEPITOPEpan: Extending TEPITOPE
        for Peptide Binding Prediction Covering over 700 HLA-HLA-DR Molecules. PLoS ONE 7(2): e30483.
    """

    __alleles = frozenset(
        ['HLA-DRB1*01:01', 'HLA-DRB1*01:02', 'HLA-DRB1*01:03', 'HLA-DRB1*01:04', 'HLA-DRB1*01:05', 'HLA-DRB1*01:06',
         'HLA-DRB1*01:07', 'HLA-DRB1*01:08', 'HLA-DRB1*01:09', 'HLA-DRB1*01:10', 'HLA-DRB1*01:11', 'HLA-DRB1*01:12',
         'HLA-DRB1*01:13', 'HLA-DRB1*01:14', 'HLA-DRB1*01:15', 'HLA-DRB1*01:16', 'HLA-DRB1*01:17', 'HLA-DRB1*01:18',
         'HLA-DRB1*01:19', 'HLA-DRB1*01:20', 'HLA-DRB1*01:21', 'HLA-DRB1*01:22', 'HLA-DRB1*01:23', 'HLA-DRB1*01:24',
         'HLA-DRB1*01:25', 'HLA-DRB1*01:26', 'HLA-DRB1*01:27', 'HLA-DRB1*01:28', 'HLA-DRB1*01:29', 'HLA-DRB1*01:30',
         'HLA-DRB1*01:31', 'HLA-DRB1*01:32', 'HLA-DRB1*01:34', 'HLA-DRB1*01:35', 'HLA-DRB1*01:36', 'HLA-DRB1*03:01',
         'HLA-DRB1*03:02', 'HLA-DRB1*03:03', 'HLA-DRB1*03:04', 'HLA-DRB1*03:05', 'HLA-DRB1*03:06', 'HLA-DRB1*03:07',
         'HLA-DRB1*03:08', 'HLA-DRB1*03:09', 'HLA-DRB1*03:10', 'HLA-DRB1*03:11', 'HLA-DRB1*03:12', 'HLA-DRB1*03:13',
         'HLA-DRB1*03:14', 'HLA-DRB1*03:15', 'HLA-DRB1*03:16', 'HLA-DRB1*03:17', 'HLA-DRB1*03:18', 'HLA-DRB1*03:19',
         'HLA-DRB1*03:20', 'HLA-DRB1*03:21', 'HLA-DRB1*03:22', 'HLA-DRB1*03:23', 'HLA-DRB1*03:24', 'HLA-DRB1*03:25',
         'HLA-DRB1*03:26', 'HLA-DRB1*03:27', 'HLA-DRB1*03:28', 'HLA-DRB1*03:29', 'HLA-DRB1*03:30', 'HLA-DRB1*03:31',
         'HLA-DRB1*03:32', 'HLA-DRB1*03:33', 'HLA-DRB1*03:34', 'HLA-DRB1*03:35', 'HLA-DRB1*03:36', 'HLA-DRB1*03:37',
         'HLA-DRB1*03:38', 'HLA-DRB1*03:39', 'HLA-DRB1*03:40', 'HLA-DRB1*03:41', 'HLA-DRB1*03:42', 'HLA-DRB1*03:43',
         'HLA-DRB1*03:44', 'HLA-DRB1*03:45', 'HLA-DRB1*03:46', 'HLA-DRB1*03:47', 'HLA-DRB1*03:48', 'HLA-DRB1*03:49',
         'HLA-DRB1*03:50', 'HLA-DRB1*03:51', 'HLA-DRB1*03:52', 'HLA-DRB1*03:53', 'HLA-DRB1*03:54', 'HLA-DRB1*03:55',
         'HLA-DRB1*03:56', 'HLA-DRB1*03:57', 'HLA-DRB1*03:58', 'HLA-DRB1*03:59', 'HLA-DRB1*03:60', 'HLA-DRB1*03:61',
         'HLA-DRB1*03:62', 'HLA-DRB1*03:63', 'HLA-DRB1*03:64', 'HLA-DRB1*04:01', 'HLA-DRB1*04:02', 'HLA-DRB1*04:03',
         'HLA-DRB1*04:04', 'HLA-DRB1*04:05', 'HLA-DRB1*04:06', 'HLA-DRB1*04:07', 'HLA-DRB1*04:08', 'HLA-DRB1*04:09',
         'HLA-DRB1*04:10', 'HLA-DRB1*04:11', 'HLA-DRB1*04:12', 'HLA-DRB1*04:13', 'HLA-DRB1*04:14', 'HLA-DRB1*04:15',
         'HLA-DRB1*04:16', 'HLA-DRB1*04:17', 'HLA-DRB1*04:18', 'HLA-DRB1*04:19', 'HLA-DRB1*04:20', 'HLA-DRB1*04:21',
         'HLA-DRB1*04:22', 'HLA-DRB1*04:23', 'HLA-DRB1*04:24', 'HLA-DRB1*04:25', 'HLA-DRB1*04:26', 'HLA-DRB1*04:27',
         'HLA-DRB1*04:28', 'HLA-DRB1*04:29', 'HLA-DRB1*04:30', 'HLA-DRB1*04:31', 'HLA-DRB1*04:32', 'HLA-DRB1*04:33',
         'HLA-DRB1*04:34', 'HLA-DRB1*04:35', 'HLA-DRB1*04:36', 'HLA-DRB1*04:37', 'HLA-DRB1*04:38', 'HLA-DRB1*04:39',
         'HLA-DRB1*04:40', 'HLA-DRB1*04:41', 'HLA-DRB1*04:42', 'HLA-DRB1*04:43', 'HLA-DRB1*04:44', 'HLA-DRB1*04:45',
         'HLA-DRB1*04:46', 'HLA-DRB1*04:47', 'HLA-DRB1*04:48', 'HLA-DRB1*04:49', 'HLA-DRB1*04:50', 'HLA-DRB1*04:51',
         'HLA-DRB1*04:52', 'HLA-DRB1*04:53', 'HLA-DRB1*04:54', 'HLA-DRB1*04:55', 'HLA-DRB1*04:56', 'HLA-DRB1*04:57',
         'HLA-DRB1*04:58', 'HLA-DRB1*04:59', 'HLA-DRB1*04:60', 'HLA-DRB1*04:61', 'HLA-DRB1*04:62', 'HLA-DRB1*04:63',
         'HLA-DRB1*04:64', 'HLA-DRB1*04:65', 'HLA-DRB1*04:66', 'HLA-DRB1*04:67', 'HLA-DRB1*04:68', 'HLA-DRB1*04:69',
         'HLA-DRB1*04:70', 'HLA-DRB1*04:71', 'HLA-DRB1*04:72', 'HLA-DRB1*04:73', 'HLA-DRB1*04:74', 'HLA-DRB1*04:75',
         'HLA-DRB1*04:76', 'HLA-DRB1*04:77', 'HLA-DRB1*04:78', 'HLA-DRB1*04:79', 'HLA-DRB1*04:80', 'HLA-DRB1*04:82',
         'HLA-DRB1*04:83', 'HLA-DRB1*04:84', 'HLA-DRB1*04:85', 'HLA-DRB1*04:86', 'HLA-DRB1*04:87', 'HLA-DRB1*04:88',
         'HLA-DRB1*04:89', 'HLA-DRB1*04:90', 'HLA-DRB1*04:91', 'HLA-DRB1*04:92', 'HLA-DRB1*04:93', 'HLA-DRB1*04:95',
         'HLA-DRB1*04:96', 'HLA-DRB1*04:97', 'HLA-DRB1*04:98', 'HLA-DRB1*07:01', 'HLA-DRB1*07:03', 'HLA-DRB1*07:04',
         'HLA-DRB1*07:05', 'HLA-DRB1*07:06', 'HLA-DRB1*07:07', 'HLA-DRB1*07:08', 'HLA-DRB1*07:09', 'HLA-DRB1*07:11',
         'HLA-DRB1*07:12', 'HLA-DRB1*07:13', 'HLA-DRB1*07:14', 'HLA-DRB1*07:15', 'HLA-DRB1*07:16', 'HLA-DRB1*07:17',
         'HLA-DRB1*07:18', 'HLA-DRB1*07:19', 'HLA-DRB1*07:20', 'HLA-DRB1*07:21', 'HLA-DRB1*08:01', 'HLA-DRB1*08:02',
         'HLA-DRB1*08:03', 'HLA-DRB1*08:04', 'HLA-DRB1*08:05', 'HLA-DRB1*08:06', 'HLA-DRB1*08:07', 'HLA-DRB1*08:08',
         'HLA-DRB1*08:09', 'HLA-DRB1*08:10', 'HLA-DRB1*08:11', 'HLA-DRB1*08:12', 'HLA-DRB1*08:13', 'HLA-DRB1*08:14',
         'HLA-DRB1*08:15', 'HLA-DRB1*08:16', 'HLA-DRB1*08:17', 'HLA-DRB1*08:18', 'HLA-DRB1*08:19', 'HLA-DRB1*08:20',
         'HLA-DRB1*08:21', 'HLA-DRB1*08:22', 'HLA-DRB1*08:23', 'HLA-DRB1*08:24', 'HLA-DRB1*08:25', 'HLA-DRB1*08:26',
         'HLA-DRB1*08:27', 'HLA-DRB1*08:28', 'HLA-DRB1*08:29', 'HLA-DRB1*08:30', 'HLA-DRB1*08:31', 'HLA-DRB1*08:32',
         'HLA-DRB1*08:33', 'HLA-DRB1*08:34', 'HLA-DRB1*08:35', 'HLA-DRB1*08:36', 'HLA-DRB1*08:37', 'HLA-DRB1*08:38',
         'HLA-DRB1*08:39', 'HLA-DRB1*08:40', 'HLA-DRB1*08:41', 'HLA-DRB1*08:42', 'HLA-DRB1*08:43', 'HLA-DRB1*08:44',
         'HLA-DRB1*08:45', 'HLA-DRB1*09:01', 'HLA-DRB1*09:02', 'HLA-DRB1*09:03', 'HLA-DRB1*09:04', 'HLA-DRB1*09:05',
         'HLA-DRB1*09:06', 'HLA-DRB1*09:07', 'HLA-DRB1*09:08', 'HLA-DRB1*09:09', 'HLA-DRB1*09:10', 'HLA-DRB1*09:11',
         'HLA-DRB1*09:12', 'HLA-DRB1*10:01', 'HLA-DRB1*10:02', 'HLA-DRB1*10:03', 'HLA-DRB1*11:01', 'HLA-DRB1*11:02',
         'HLA-DRB1*11:03', 'HLA-DRB1*11:04', 'HLA-DRB1*11:05', 'HLA-DRB1*11:06', 'HLA-DRB1*11:07', 'HLA-DRB1*11:08',
         'HLA-DRB1*11:09', 'HLA-DRB1*11:10', 'HLA-DRB1*11:11', 'HLA-DRB1*11:12', 'HLA-DRB1*11:13', 'HLA-DRB1*11:14',
         'HLA-DRB1*11:15', 'HLA-DRB1*11:16', 'HLA-DRB1*11:17', 'HLA-DRB1*11:18', 'HLA-DRB1*11:19', 'HLA-DRB1*11:20',
         'HLA-DRB1*11:21', 'HLA-DRB1*11:22', 'HLA-DRB1*11:23', 'HLA-DRB1*11:24', 'HLA-DRB1*11:25', 'HLA-DRB1*11:26',
         'HLA-DRB1*11:27', 'HLA-DRB1*11:28', 'HLA-DRB1*11:29', 'HLA-DRB1*11:30', 'HLA-DRB1*11:31', 'HLA-DRB1*11:32',
         'HLA-DRB1*11:33', 'HLA-DRB1*11:34', 'HLA-DRB1*11:35', 'HLA-DRB1*11:36', 'HLA-DRB1*11:37', 'HLA-DRB1*11:38',
         'HLA-DRB1*11:39', 'HLA-DRB1*11:40', 'HLA-DRB1*11:41', 'HLA-DRB1*11:42', 'HLA-DRB1*11:43', 'HLA-DRB1*11:44',
         'HLA-DRB1*11:45', 'HLA-DRB1*11:46', 'HLA-DRB1*11:47', 'HLA-DRB1*11:48', 'HLA-DRB1*11:49', 'HLA-DRB1*11:50',
         'HLA-DRB1*11:51', 'HLA-DRB1*11:52', 'HLA-DRB1*11:53', 'HLA-DRB1*11:54', 'HLA-DRB1*11:55', 'HLA-DRB1*11:56',
         'HLA-DRB1*11:57', 'HLA-DRB1*11:58', 'HLA-DRB1*11:59', 'HLA-DRB1*11:60', 'HLA-DRB1*11:61', 'HLA-DRB1*11:62',
         'HLA-DRB1*11:63', 'HLA-DRB1*11:64', 'HLA-DRB1*11:65', 'HLA-DRB1*11:66', 'HLA-DRB1*11:67', 'HLA-DRB1*11:68',
         'HLA-DRB1*11:69', 'HLA-DRB1*11:70', 'HLA-DRB1*11:72', 'HLA-DRB1*11:73', 'HLA-DRB1*11:74', 'HLA-DRB1*11:75',
         'HLA-DRB1*11:76', 'HLA-DRB1*11:77', 'HLA-DRB1*11:78', 'HLA-DRB1*11:79', 'HLA-DRB1*11:80', 'HLA-DRB1*11:81',
         'HLA-DRB1*11:82', 'HLA-DRB1*11:83', 'HLA-DRB1*11:84', 'HLA-DRB1*11:85', 'HLA-DRB1*11:86', 'HLA-DRB1*11:87',
         'HLA-DRB1*11:88', 'HLA-DRB1*11:89', 'HLA-DRB1*11:90', 'HLA-DRB1*11:91', 'HLA-DRB1*11:92', 'HLA-DRB1*11:93',
         'HLA-DRB1*11:94', 'HLA-DRB1*11:95', 'HLA-DRB1*11:96', 'HLA-DRB1*11:97', 'HLA-DRB1*11:98', 'HLA-DRB1*11:99',
         'HLA-DRB1*12:01', 'HLA-DRB1*12:02', 'HLA-DRB1*12:03', 'HLA-DRB1*12:04', 'HLA-DRB1*12:05', 'HLA-DRB1*12:06',
         'HLA-DRB1*12:07', 'HLA-DRB1*12:08', 'HLA-DRB1*12:09', 'HLA-DRB1*12:10', 'HLA-DRB1*12:11', 'HLA-DRB1*12:12',
         'HLA-DRB1*12:13', 'HLA-DRB1*12:14', 'HLA-DRB1*12:15', 'HLA-DRB1*12:16', 'HLA-DRB1*12:17', 'HLA-DRB1*12:18',
         'HLA-DRB1*12:19', 'HLA-DRB1*12:20', 'HLA-DRB1*12:21', 'HLA-DRB1*12:22', 'HLA-DRB1*12:23', 'HLA-DRB1*12:25',
         'HLA-DRB1*12:26', 'HLA-DRB1*12:27', 'HLA-DRB1*13:01', 'HLA-DRB1*13:02', 'HLA-DRB1*13:03', 'HLA-DRB1*13:04',
         'HLA-DRB1*13:05', 'HLA-DRB1*13:06', 'HLA-DRB1*13:07', 'HLA-DRB1*13:08', 'HLA-DRB1*13:09', 'HLA-DRB1*13:10',
         'HLA-DRB1*13:11', 'HLA-DRB1*13:12', 'HLA-DRB1*13:13', 'HLA-DRB1*13:14', 'HLA-DRB1*13:15', 'HLA-DRB1*13:16',
         'HLA-DRB1*13:17', 'HLA-DRB1*13:18', 'HLA-DRB1*13:19', 'HLA-DRB1*13:20', 'HLA-DRB1*13:21', 'HLA-DRB1*13:22',
         'HLA-DRB1*13:23', 'HLA-DRB1*13:24', 'HLA-DRB1*13:25', 'HLA-DRB1*13:26', 'HLA-DRB1*13:27', 'HLA-DRB1*13:28',
         'HLA-DRB1*13:29', 'HLA-DRB1*13:30', 'HLA-DRB1*13:31', 'HLA-DRB1*13:32', 'HLA-DRB1*13:33', 'HLA-DRB1*13:34',
         'HLA-DRB1*13:35', 'HLA-DRB1*13:36', 'HLA-DRB1*13:37', 'HLA-DRB1*13:38', 'HLA-DRB1*13:39', 'HLA-DRB1*13:40',
         'HLA-DRB1*13:41', 'HLA-DRB1*13:42', 'HLA-DRB1*13:43', 'HLA-DRB1*13:44', 'HLA-DRB1*13:45', 'HLA-DRB1*13:46',
         'HLA-DRB1*13:47', 'HLA-DRB1*13:48', 'HLA-DRB1*13:49', 'HLA-DRB1*13:50', 'HLA-DRB1*13:51', 'HLA-DRB1*13:52',
         'HLA-DRB1*13:53', 'HLA-DRB1*13:54', 'HLA-DRB1*13:55', 'HLA-DRB1*13:56', 'HLA-DRB1*13:57', 'HLA-DRB1*13:58',
         'HLA-DRB1*13:59', 'HLA-DRB1*13:60', 'HLA-DRB1*13:61', 'HLA-DRB1*13:62', 'HLA-DRB1*13:63', 'HLA-DRB1*13:64',
         'HLA-DRB1*13:65', 'HLA-DRB1*13:66', 'HLA-DRB1*13:67', 'HLA-DRB1*13:68', 'HLA-DRB1*13:69', 'HLA-DRB1*13:70',
         'HLA-DRB1*13:71', 'HLA-DRB1*13:72', 'HLA-DRB1*13:73', 'HLA-DRB1*13:74', 'HLA-DRB1*13:75', 'HLA-DRB1*13:76',
         'HLA-DRB1*13:77', 'HLA-DRB1*13:78', 'HLA-DRB1*13:79', 'HLA-DRB1*13:80', 'HLA-DRB1*13:81', 'HLA-DRB1*13:82',
         'HLA-DRB1*13:83', 'HLA-DRB1*13:84', 'HLA-DRB1*13:85', 'HLA-DRB1*13:86', 'HLA-DRB1*13:87', 'HLA-DRB1*13:88',
         'HLA-DRB1*13:89', 'HLA-DRB1*13:90', 'HLA-DRB1*13:91', 'HLA-DRB1*13:92', 'HLA-DRB1*13:93', 'HLA-DRB1*13:94',
         'HLA-DRB1*13:95', 'HLA-DRB1*13:96', 'HLA-DRB1*13:97', 'HLA-DRB1*13:98', 'HLA-DRB1*13:99', 'HLA-DRB1*14:01',
         'HLA-DRB1*14:02', 'HLA-DRB1*14:03', 'HLA-DRB1*14:04', 'HLA-DRB1*14:05', 'HLA-DRB1*14:06', 'HLA-DRB1*14:07',
         'HLA-DRB1*14:08', 'HLA-DRB1*14:09', 'HLA-DRB1*14:10', 'HLA-DRB1*14:11', 'HLA-DRB1*14:12', 'HLA-DRB1*14:13',
         'HLA-DRB1*14:14', 'HLA-DRB1*14:15', 'HLA-DRB1*14:16', 'HLA-DRB1*14:17', 'HLA-DRB1*14:18', 'HLA-DRB1*14:19',
         'HLA-DRB1*14:20', 'HLA-DRB1*14:21', 'HLA-DRB1*14:22', 'HLA-DRB1*14:23', 'HLA-DRB1*14:24', 'HLA-DRB1*14:25',
         'HLA-DRB1*14:26', 'HLA-DRB1*14:27', 'HLA-DRB1*14:28', 'HLA-DRB1*14:29', 'HLA-DRB1*14:30', 'HLA-DRB1*14:31',
         'HLA-DRB1*14:32', 'HLA-DRB1*14:33', 'HLA-DRB1*14:34', 'HLA-DRB1*14:35', 'HLA-DRB1*14:36', 'HLA-DRB1*14:37',
         'HLA-DRB1*14:38', 'HLA-DRB1*14:39', 'HLA-DRB1*14:40', 'HLA-DRB1*14:41', 'HLA-DRB1*14:42', 'HLA-DRB1*14:43',
         'HLA-DRB1*14:44', 'HLA-DRB1*14:45', 'HLA-DRB1*14:46', 'HLA-DRB1*14:47', 'HLA-DRB1*14:48', 'HLA-DRB1*14:49',
         'HLA-DRB1*14:50', 'HLA-DRB1*14:51', 'HLA-DRB1*14:52', 'HLA-DRB1*14:53', 'HLA-DRB1*14:54', 'HLA-DRB1*14:55',
         'HLA-DRB1*14:56', 'HLA-DRB1*14:57', 'HLA-DRB1*14:58', 'HLA-DRB1*14:59', 'HLA-DRB1*14:60', 'HLA-DRB1*14:61',
         'HLA-DRB1*14:62', 'HLA-DRB1*14:63', 'HLA-DRB1*14:64', 'HLA-DRB1*14:65', 'HLA-DRB1*14:67', 'HLA-DRB1*14:68',
         'HLA-DRB1*14:69', 'HLA-DRB1*14:70', 'HLA-DRB1*14:71', 'HLA-DRB1*14:72', 'HLA-DRB1*14:73', 'HLA-DRB1*14:74',
         'HLA-DRB1*14:75', 'HLA-DRB1*14:76', 'HLA-DRB1*14:77', 'HLA-DRB1*14:78', 'HLA-DRB1*14:79', 'HLA-DRB1*14:80',
         'HLA-DRB1*14:81', 'HLA-DRB1*14:82', 'HLA-DRB1*14:83', 'HLA-DRB1*14:84', 'HLA-DRB1*14:85', 'HLA-DRB1*14:86',
         'HLA-DRB1*14:87', 'HLA-DRB1*14:88', 'HLA-DRB1*14:89', 'HLA-DRB1*14:90', 'HLA-DRB1*14:91', 'HLA-DRB1*14:93',
         'HLA-DRB1*14:94', 'HLA-DRB1*14:95', 'HLA-DRB1*14:96', 'HLA-DRB1*14:97', 'HLA-DRB1*14:98', 'HLA-DRB1*14:99',
         'HLA-DRB1*15:01', 'HLA-DRB1*15:02', 'HLA-DRB1*15:03', 'HLA-DRB1*15:04', 'HLA-DRB1*15:05', 'HLA-DRB1*15:06',
         'HLA-DRB1*15:07', 'HLA-DRB1*15:08', 'HLA-DRB1*15:09', 'HLA-DRB1*15:10', 'HLA-DRB1*15:11', 'HLA-DRB1*15:12',
         'HLA-DRB1*15:13', 'HLA-DRB1*15:14', 'HLA-DRB1*15:15', 'HLA-DRB1*15:16', 'HLA-DRB1*15:18', 'HLA-DRB1*15:19',
         'HLA-DRB1*15:20', 'HLA-DRB1*15:21', 'HLA-DRB1*15:22', 'HLA-DRB1*15:23', 'HLA-DRB1*15:24', 'HLA-DRB1*15:25',
         'HLA-DRB1*15:26', 'HLA-DRB1*15:27', 'HLA-DRB1*15:28', 'HLA-DRB1*15:29', 'HLA-DRB1*15:30', 'HLA-DRB1*15:31',
         'HLA-DRB1*15:32', 'HLA-DRB1*15:33', 'HLA-DRB1*15:34', 'HLA-DRB1*15:35', 'HLA-DRB1*15:36', 'HLA-DRB1*15:37',
         'HLA-DRB1*15:38', 'HLA-DRB1*15:39', 'HLA-DRB1*15:40', 'HLA-DRB1*15:41', 'HLA-DRB1*15:42', 'HLA-DRB1*15:43',
         'HLA-DRB1*15:44', 'HLA-DRB1*15:45', 'HLA-DRB1*15:46', 'HLA-DRB1*15:47', 'HLA-DRB1*15:48', 'HLA-DRB1*15:49',
         'HLA-DRB1*15:51', 'HLA-DRB1*15:52', 'HLA-DRB1*15:53', 'HLA-DRB1*15:54', 'HLA-DRB1*15:55', 'HLA-DRB1*15:56',
         'HLA-DRB1*15:57', 'HLA-DRB1*16:01', 'HLA-DRB1*16:02', 'HLA-DRB1*16:03', 'HLA-DRB1*16:04', 'HLA-DRB1*16:05',
         'HLA-DRB1*16:07', 'HLA-DRB1*16:08', 'HLA-DRB1*16:09', 'HLA-DRB1*16:10', 'HLA-DRB1*16:11', 'HLA-DRB1*16:12',
         'HLA-DRB1*16:14', 'HLA-DRB1*16:15', 'HLA-DRB1*16:16', 'HLA-DRB1*16:17', 'HLA-DRB1*16:18', 'HLA-DRB3*01:01',
         'HLA-DRB3*01:02', 'HLA-DRB3*01:03', 'HLA-DRB3*01:04', 'HLA-DRB3*01:05', 'HLA-DRB3*01:06', 'HLA-DRB3*01:07',
         'HLA-DRB3*01:08', 'HLA-DRB3*01:09', 'HLA-DRB3*01:10', 'HLA-DRB3*01:11', 'HLA-DRB3*01:12', 'HLA-DRB3*01:13',
         'HLA-DRB3*01:14', 'HLA-DRB3*01:15', 'HLA-DRB3*02:01', 'HLA-DRB3*02:02', 'HLA-DRB3*02:03', 'HLA-DRB3*02:04',
         'HLA-DRB3*02:05', 'HLA-DRB3*02:06', 'HLA-DRB3*02:07', 'HLA-DRB3*02:08', 'HLA-DRB3*02:09', 'HLA-DRB3*02:10',
         'HLA-DRB3*02:11', 'HLA-DRB3*02:12', 'HLA-DRB3*02:13', 'HLA-DRB3*02:14', 'HLA-DRB3*02:15', 'HLA-DRB3*02:16',
         'HLA-DRB3*02:17', 'HLA-DRB3*02:18', 'HLA-DRB3*02:19', 'HLA-DRB3*02:20', 'HLA-DRB3*02:21', 'HLA-DRB3*02:22',
         'HLA-DRB3*02:23', 'HLA-DRB3*02:24', 'HLA-DRB3*02:25', 'HLA-DRB3*02:26', 'HLA-DRB3*02:27', 'HLA-DRB3*02:28',
         'HLA-DRB3*03:01', 'HLA-DRB3*03:02', 'HLA-DRB3*03:03', 'HLA-DRB4*01:01', 'HLA-DRB4*01:03', 'HLA-DRB4*01:04',
         'HLA-DRB4*01:05', 'HLA-DRB4*01:06', 'HLA-DRB4*01:07', 'HLA-DRB4*01:08', 'HLA-DRB5*01:01', 'HLA-DRB5*01:02',
         'HLA-DRB5*01:04', 'HLA-DRB5*01:05', 'HLA-DRB5*01:06', 'HLA-DRB5*01:07', 'HLA-DRB5*01:08', 'HLA-DRB5*01:09',
         'HLA-DRB5*01:11', 'HLA-DRB5*01:12', 'HLA-DRB5*01:13', 'HLA-DRB5*01:14', 'HLA-DRB5*02:02', 'HLA-DRB5*02:03',
         'HLA-DRB5*02:04', 'HLA-DRB5*02:05'])
    __supported_length = frozenset([9])
    __name = "tepitopepan"
    __version = "1.0"

    @property
    def version(self):
        """The version of the predictor"""
        return self.__version

    @property
    def name(self):
        """The name of the predictor"""
        return self.__name

    @property
    def supportedAlleles(self):
        """
        A list of supported :class:`~epytope.Core.Allele.Allele`
        """
        return self.__alleles

    @property
    def supportedLength(self):
        """
        A list of supported :class:`~epytope.Core.Peptide.Peptide` lengths
        """
        return self.__supported_length

    def convert_alleles(self, alleles):
        """
        Converts :class:`~epytope.Core.Allele.Allele` into the internal :class:`~epytope.Core.Allele.Allele`
        representation of the predictor and returns a string representation

        :param alleles: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is needed
        :type alleles: list(:class:`~epytope.Core.Allele.Allele`)
        :return: Returns a string representation of the input :class:`~epytope.Core.Allele.Allele`
        :rtype: list(str)
        """
        return ["%s_%s%s" % (a.locus, a.supertype, a.subtype) for a in alleles]


class CalisImm(APSSMEpitopePrediction):
    """
    Implements the Immunogenicity propensity score proposed by Calis et al.

    ..note:

        Calis, Jorg JA, et al.(2013). Properties of MHC class I presented peptides that enhance immunogenicity.
        PLoS Comput Biol 9.10 e1003266.

    """

    __alleles = frozenset(
        ['HLA-B*40:02', 'HLA-B*07:02', 'HLA-B*08:01', 'HLA-B*44:03', 'HLA-B*44:02', 'HLA-A*68:01', 'HLA-A*24:02',
         'HLA-B*40:01', 'HLA-A*31:01', 'HLA-B*39:01', 'HLA-A*01:01', 'HLA-B*58:01',
         'HLA-B*57:01', 'HLA-A*30:02', 'HLA-A*30:01', 'HLA-B*35:01', 'HLA-B*51:01', 'HLA-A*32:01', 'HLA-B*53:01',
         'HLA-A*26:01', 'HLA-A*03:01', 'HLA-B*15:02', 'HLA-B*15:01', 'HLA-B*45:01',
         'HLA-B*54:01', 'HLA-B*18:01', 'HLA-A*68:02', 'HLA-A*69:01', 'HLA-A*02:11', 'HLA-A*11:01', 'HLA-A*23:01',
         'HLA-A*33:01', 'HLA-B*46:01', 'HLA-A*02:06', 'HLA-A*02:01', 'HLA-A*02:02',
         'HLA-A*02:03', 'HLA-A*29:02', 'HLA-B*27:05'])
    __supported_length = frozenset([9, 10, 11])
    __name = "calisimm"
    __version = "1.0"

    __log_enrichment = {"A": 0.127, "C": -0.175, "D": 0.072, "E": 0.325, "F": 0.380, "G": 0.11, "H": 0.105, "I": 0.432,
                        "K": -0.7, "L": -0.036, "M": -0.57, "N": -0.021, "P": -0.036, "Q": -0.376, "R": 0.168,
                        "S": -0.537, "T": 0.126, "V": 0.134, "W": 0.719, "Y": -0.012}
    __importance = [0., 0., 0.1, 0.31, 0.3, 0.29, 0.26, 0.18, 0.]

    @property
    def version(self):
        """The version of the predictor"""
        return self.__version

    @property
    def name(self):
        """The name of the predictor"""
        return self.__name

    @property
    def supportedAlleles(self):
        """
        A list of supported :class:`~epytope.Core.Allele.Allele`
        """
        return self.__alleles

    @property
    def supportedLength(self):
        """
        A list of supported :class:`~epytope.Core.Peptide.Peptide` lengths
        """
        return self.__supported_length

    def predict(self, peptides, alleles=None, **kwargs):
        """
        Returns predictions for given peptides an :class:`~epytope.Core.Allele.Allele`. If no
        :class:`~epytope.Core.Allele.Allele` are given, predictions for all available models are made.

        :param peptides: A single :class:`~epytope.Core.Peptide.Peptide` or a list of :class:`~epytope.Core.Peptide.Peptide`
        :type peptides: list(:class:`~epytope.Core.Peptide.Peptide`) or :class:`~epytope.Core.Peptide.Peptide`
        :param kwargs: optional parameter (not used yet)
        :return: Returns a :class:`pandas.DataFrame` object with the prediction results
        :rtype: :class:`pandas.DataFrame`
        """

        def __load_allele_model(allele, length):
            allele_model = "%s" % allele
            return getattr(
                __import__("epytope.Data.pssms." + self.name + ".mat." + allele_model, fromlist=[allele_model]),
                allele_model)

        if isinstance(peptides, Peptide):
            pep_seqs = {str(peptides): peptides}
        else:
            pep_seqs = {}
            for p in peptides:
                if not isinstance(p, Peptide):
                    raise ValueError("Input is not of type Protein or Peptide")
                pep_seqs[str(p)] = p

        if alleles is None:
            al = [Allele(a) for a in self.supportedAlleles]
            alleles_string = {conv_a: a for conv_a, a in zip(self.convert_alleles(al), al)}
        else:
            if isinstance(alleles, Allele):
                alleles = [alleles]
            if any(not isinstance(p, Allele) for p in alleles):
                raise ValueError("Input is not of type Allele")
            alleles_string = {conv_a: a for conv_a, a in zip(self.convert_alleles(alleles), alleles)}

        scores = {}
        pep_groups = list(pep_seqs.keys())
        pep_groups.sort(key=len)
        for length, peps in itertools.groupby(pep_groups, key=len):

            if self.supportedLength is not None and length not in self.supportedLength:
                warnings.warn("Peptide length of %i is not supported by %s" % (length, self.name))
                continue

            peps = list(peps)
            for a, allele in alleles_string.items():

                if alleles_string[a] not in scores:
                    scores[allele] = {}

                # load matrix
                try:
                    pssm = __load_allele_model(a, length)
                except ImportError:
                    pssm = []

                importance = self.__importance if length <= 9 else \
                    self.__importance[:5] + ((length - 9) * [0.30]) + self.__importance[5:]

                for p in peps:
                    score = sum(self.__log_enrichment.get(p[i], 0.0) * importance[i]
                                for i in range(length) if i not in pssm)
                    scores[allele][pep_seqs[p]] = score

        if not scores:
            raise ValueError("No predictions could be made with " + self.name + " for given input. Check your"
                                                                                "epitope length and HLA allele combination.")

        result = {allele: {"Score":(list(scores.values())[j])} for j, allele in enumerate(alleles)}

        df_result = EpitopePredictionResult.from_dict(result, peps, self.name)
        return df_result

    def convert_alleles(self, alleles):
        return [x.name.replace("*", "").replace(":", "") for x in alleles]
