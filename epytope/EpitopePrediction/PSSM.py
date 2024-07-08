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
import pandas

from epytope.Core.Allele import Allele
from epytope.Core.Peptide import Peptide
from epytope.Core.Result import EpitopePredictionResult
from epytope.Core.Base import AEpitopePrediction


class APSSMEpitopePrediction(AEpitopePrediction):
    """
        Abstract base class for PSSM predictions.
        Implements predict functionality
    """
    @classmethod
    def convert_alleles(cls, alleles):
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

        result = {}
        pep_groups = list(pep_seqs.keys())
        pep_groups.sort(key=len)
        for length, peps in itertools.groupby(pep_groups, key=len):
            peps = list(peps)
            # dynamically import prediction PSSMs for alleles and predict
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
    __name = "syfpeithi"
    __version = "1.0"
    __supported_length = frozenset([8, 9, 10, 11, 12, 13])
    __allele_import_name = f"{__name}_{__version}".replace('.', '_')
    __alleles = getattr(__import__("epytope.Data.supportedAlleles.pssm." + __allele_import_name,
                                   fromlist=[__allele_import_name])
                        , __allele_import_name)

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


class BIMAS(APSSMEpitopePrediction):
    """
    Represents the BIMAS PSSM predictor.

    .. note::

        Parker, K.C., Bednarek, M.A. and Coligan, J.E. Scheme for ranking potential HLA-A2 binding peptides based on
        independent binding of individual peptide side-chains. The Journal of Immunology 1994;152(1):163-175.
    """
    __name = "bimas"
    __version = "1.0"
    __supported_length = frozenset([8, 9])
    __allele_import_name = f"{__name}_{__version}".replace('.', '_')
    __alleles = getattr(__import__("epytope.Data.supportedAlleles.pssm." + __allele_import_name,
                                   fromlist=[__allele_import_name])
                        , __allele_import_name)

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
    __name = "epidemix"
    __version = "1.0"
    __supported_length = frozenset([9, 10, 8, 11])
    __allele_import_name = f"{__name}_{__version}".replace('.', '_')
    __alleles = getattr(__import__("epytope.Data.supportedAlleles.pssm." + __allele_import_name,
                                   fromlist=[__allele_import_name])
                        , __allele_import_name)

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


class Hammer(APSSMEpitopePrediction):
    """
    Represents the virtual pockets approach by Sturniolo et al.

    .. note::

        Sturniolo, T., et al. Generation of tissue-specific and promiscuous HLA ligand databases using DNA microarrays
        and virtual HLA class II matrices. Nature biotechnology 1999;17(6):555-561.
    """
    __name = "hammer"
    __version = "1.0"
    __supported_length = frozenset([9])
    __allele_import_name = f"{__name}_{__version}".replace('.', '_')
    __alleles = getattr(__import__("epytope.Data.supportedAlleles.pssm." + __allele_import_name,
                                   fromlist=[__allele_import_name])
                        , __allele_import_name)

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


class SMM(APSSMEpitopePrediction):
    """
    Implements IEDBs SMM PSSM method.

    .. note::

        Peters B, Sette A. 2005. Generating quantitative models describing the sequence specificity of
        biological processes with the stabilized matrix method. BMC Bioinformatics 6:132.
    """
    __name = "smm"
    __version = "1.0"
    __supported_length = frozenset([8, 9, 10, 11])
    __allele_import_name = f"{__name}_{__version}".replace('.', '_')
    __alleles = getattr(__import__("epytope.Data.supportedAlleles.pssm." + __allele_import_name,
                                   fromlist=[__allele_import_name])
                        , __allele_import_name)

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
    __name = "smmpmbec"
    __version = "1.0"
    __supported_length = frozenset([8, 9, 10, 11])
    __allele_import_name = f"{__name}_{__version}".replace('.', '_')
    __alleles = getattr(__import__("epytope.Data.supportedAlleles.pssm." + __allele_import_name,
                                   fromlist=[__allele_import_name])
                        , __allele_import_name)

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
    __name = "arb"
    __version = "1.0"
    __supported_length = frozenset([8, 9, 10, 11])
    __allele_import_name = f"{__name}_{__version}".replace('.', '_')
    __alleles = getattr(__import__("epytope.Data.supportedAlleles.pssm." + __allele_import_name,
                                   fromlist=[__allele_import_name])
                        , __allele_import_name)


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
    __name = "comblibsidney"
    __version = "1.0"
    __supported_length = frozenset([9])
    __allele_import_name = f"{__name}_{__version}".replace('.', '_')
    __alleles = getattr(__import__("epytope.Data.supportedAlleles.pssm." + __allele_import_name,
                                   fromlist=[__allele_import_name])
                        , __allele_import_name)

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
    __name = "tepitopepan"
    __version = "1.0"
    __supported_length = frozenset([9])
    __allele_import_name = f"{__name}_{__version}".replace('.', '_')
    __alleles = getattr(__import__("epytope.Data.supportedAlleles.pssm." + __allele_import_name,
                                   fromlist=[__allele_import_name])
                        , __allele_import_name)

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


class CalisImm(APSSMEpitopePrediction):
    """
    Implements the Immunogenicity propensity score proposed by Calis et al.

    ..note:

        Calis, Jorg JA, et al.(2013). Properties of MHC class I presented peptides that enhance immunogenicity.
        PLoS Comput Biol 9.10 e1003266.

    """

    __name = "calisimm"
    __version = "1.0"
    __supported_length = frozenset([9, 10, 11])
    __allele_import_name = __name + '_' + __version.replace('.', '_')
    __alleles = getattr(__import__("epytope.Data.supportedAlleles.pssm." + __allele_import_name,
                                   fromlist=[__allele_import_name])
                        , __allele_import_name)

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
