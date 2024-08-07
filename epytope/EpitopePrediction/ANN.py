# coding=utf-8
# This code is part of the epytope distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
.. module:: EpitopePrediction.ANN
   :synopsis: This module contains all classes for ANN-based epitope prediction.
.. moduleauthor:: heumos, krakau

"""
import itertools
import logging
import subprocess
import csv
import sys
import tempfile
import io
import math

from abc import abstractmethod
from enum import IntEnum

import pandas
from collections import defaultdict

from mhcflurry import Class1AffinityPredictor
from mhcnuggets.src.predict import predict as mhcnuggets_predict
from epytope.Core import EpitopePredictionResult
from epytope.Core.Base import AEpitopePrediction
from epytope.Core.Allele import Allele, CombinedAllele, MouseAllele
from epytope.IO.Utils import capture_stdout

import inspect


class BadSignatureException(Exception):
    pass


class SignatureCheckerMeta(type):
    def __new__(cls, name, baseClasses, d):
        # For each method in d, check to see if any base class already
        # defined a method with that name. If so, make sure the
        # signatures are the same.
        for methodName in d:
            f = d[methodName]
            for baseClass in baseClasses:
                try:
                    fBase = getattr(baseClass, methodName).__func__
<<<<<<< HEAD
                    if not inspect.getargspec(f) == inspect.getargspec(fBase):
=======
                    if not inspect.getfullargspec(f) == inspect.getfullargspec(fBase):
>>>>>>> repo-b2/main
                        raise BadSignatureException(str(methodName))
                except AttributeError:
                    # This method was not defined in this base class,
                    # So just go to the next base class.
                    continue

        return type(name, baseClasses, d)


class AANNEpitopePrediction(AEpitopePrediction):
    """
        Abstract base class for ANN predictions.
        Implements predict functionality
    """

    @abstractmethod
    def predict(self, peptides, alleles=None, binary=False, **kwargs):
        """
        All ANN based predictors have to implement their custom predict method.
        Furthermore, all of them have to use the metaclass SignatureCheckerMeta to check for any contract violations.
        They have to adhere to the following contract

        :param peptides: A single :class:`~epytope.Core.Peptide.Peptide` or a list of :class:`~epytope.Core.Peptide.Peptide`
        :type peptides: list(:class:`~epytope.Core.Peptide.Peptide`) or :class:`~epytope.Core.Peptide.Peptide`
        :param kwargs: optional parameter (not used yet)
        :return: Returns a nested dictionary {allele1: {scoreType1: {pep1: score1, pep2:..}, scoreType2: {..}, ..}, allele2:..}
        :rtype: :class:`pandas.DataFrame`
        """

try:
    class MHCNuggetsPredictor_class1_2_0(AANNEpitopePrediction):
        """
        Implements MHCNuggets Class I

        .. note::
            Evaluation of machine learning methods to predict peptide binding to MHC Class I proteins
            Rohit Bhattacharya, Ashok Sivakumar, Collin Tokheim, Violeta Beleva Guthrie, Valsamo Anagnostou,
            Victor E. Velculescu, Rachel Karchin (2017) bioRxiv
        """
<<<<<<< HEAD
        __alleles = frozenset(
            ["HLA-A*01:01", "HLA-A*02:01", "HLA-A*02:02", "HLA-A*02:03", "HLA-A*02:04", "HLA-A*02:05", "HLA-A*02:06",
             "HLA-A*02:07", "HLA-A*02:08", "HLA-A*02:09", "HLA-A*02:10", "HLA-A*02:11", "HLA-A*02:12", "HLA-A*02:14",
             "HLA-A*02:16", "HLA-A*02:17", "HLA-A*02:19", "HLA-A*02:50", "HLA-A*03:01", "HLA-A*03:02", "HLA-A*03:19",
             "HLA-A*11:01", "HLA-A*11:02", "HLA-A*23:01", "HLA-A*24:01", "HLA-A*24:02", "HLA-A*24:03", "HLA-A*25:01",
             "HLA-A*26:01", "HLA-A*26:02", "HLA-A*26:03", "HLA-A*29:01", "HLA-A*29:02", "HLA-A*30:01", "HLA-A*30:02",
             "HLA-A*30:03", "HLA-A*30:04", "HLA-A*31:01", "HLA-A*32:01", "HLA-A*32:07", "HLA-A*32:15", "HLA-A*33:01",
             "HLA-A*33:03", "HLA-A*66:01", "HLA-A*68:01", "HLA-A*68:02", "HLA-A*68:23", "HLA-A*69:01", "HLA-A*74:01",
             "HLA-A*80:01", "HLA-B*07:01", "HLA-B*07:02", "HLA-B*08:01", "HLA-B*08:02", "HLA-B*08:03", "HLA-B*12:01",
             "HLA-B*13:02", "HLA-B*14:01", "HLA-B*14:02", "HLA-B*15:01", "HLA-B*15:02", "HLA-B*15:03", "HLA-B*15:08",
             "HLA-B*15:09", "HLA-B*15:10", "HLA-B*15:13", "HLA-B*15:16", "HLA-B*15:17", "HLA-B*15:42", "HLA-B*18:01",
             "HLA-B*27:01", "HLA-B*27:02", "HLA-B*27:03", "HLA-B*27:04", "HLA-B*27:05", "HLA-B*27:06", "HLA-B*27:09",
             "HLA-B*27:10", "HLA-B*27:20", "HLA-B*35:01", "HLA-B*35:02", "HLA-B*35:03", "HLA-B*35:08", "HLA-B*37:01",
             "HLA-B*38:01", "HLA-B*39:01", "HLA-B*39:06", "HLA-B*39:09", "HLA-B*39:10", "HLA-B*40:01", "HLA-B*40:02",
             "HLA-B*40:13", "HLA-B*41:03", "HLA-B*41:04", "HLA-B*42:01", "HLA-B*42:02", "HLA-B*44:01", "HLA-B*44:02",
             "HLA-B*44:03", "HLA-B*44:05", "HLA-B*45:01", "HLA-B*45:06", "HLA-B*46:01", "HLA-B*48:01", "HLA-B*51:01",
             "HLA-B*51:02", "HLA-B*52:01", "HLA-B*53:01", "HLA-B*54:01", "HLA-B*55:01", "HLA-B*55:02", "HLA-B*56:01",
             "HLA-B*57:01", "HLA-B*57:02", "HLA-B*57:03", "HLA-B*58:01", "HLA-B*58:02", "HLA-B*60:01", "HLA-B*61:01",
             "HLA-B*62:01", "HLA-B*73:01", "HLA-B*81:01", "HLA-B*83:01", "H-2-Db", "H-2-Dd", "H-2-Dk",
             "H-2-Kb", "H-2-Kd", "H-2-Kk", "H-2-Ld", "H-2-Lq"])
        __supported_length = frozenset([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        __name = "mhcnuggets-class-1"
        __version = "2.0"

        # the interface defines three class properties
=======
        __name = "mhcnuggets-class-1"
        __version = "2.0"
        __supported_length = frozenset([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        __allele_import_name = f"{__name}_{__version}".replace('-', '_').replace('.', '_')
        __alleles = getattr(__import__("epytope.Data.supportedAlleles.ann." + __allele_import_name,
                                       fromlist=[__allele_import_name])
                            , __allele_import_name)

>>>>>>> repo-b2/main
        @property
        def name(self):
            # returns the name of the predictor
            return self.__name

        @property
        def supportedAlleles(self):
            # returns the supported alleles as strings (without the HLA prefix)
            return self.__alleles

        @property
        def supportedLength(self):
            # returns the supported epitope lengths as iterable
            return self.__supported_length

        @property
        def version(self):
            # returns the version of the predictor
            return self.__version

        def _represent(self, allele):
            """
            Internal function transforming an allele object into its representative string
            :param allele: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is
                            needed
            :type alleles: :class:`~epytope.Core.Allele.Allele`
            :return: str
            """
            if isinstance(allele, MouseAllele):
                return "%s-%s%s%s" % (allele.organism, allele.locus, allele.supertype.upper(), allele.subtype)
            else:
                return "%s-%s%s%s" % (allele.organism, allele.locus, allele.supertype, allele.subtype)

        # Converts epytopes internal allele representation into the format required by mhcnuggets
        def convert_alleles(self, alleles):
            """
            Converts :class:`~epytope.Core.Allele.Allele` into the internal :class:`~epytope.Core.Allele.Allele` representation
            of the predictor and returns a string representation

            :param alleles: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is
                            needed
            :type alleles: :class:`~epytope.Core.Allele.Allele`
            :return: Returns a string representation of the input :class:`~epytope.Core.Allele.Allele`
            :rtype: list(str)
            """
            return [self._represent(a) for a in alleles]

        # Converts the internal mhcnuggets-class-1 HLA representation back into a epytope representation
        def revert_allele_repr(self, name):
            if name.startswith("H-2-"):
                return MouseAllele(name)
            else:
                name = name[:5] + '*' + name[5:7] + ':' + name[7:]
                return Allele(name)

        # predicts the binding affinity for a set of peptides and alleles
        def predict(self, peptides, alleles=None, binary=False, **kwargs):

            # test whether one peptide or a list
            if not isinstance(peptides, list):
                peptides = [peptides]

            # if no alleles are specified do predictions for all supported alleles
            if alleles is None:
                alleles = self.supportedAlleles
            else:
                # filter for supported alleles
                alleles = [a for a in alleles if a in self.supportedAlleles]
            # Create a dictionary with Allele Obj as key and the respective allele predictor representation as value
            alleles_repr = {allele: self._represent(allele) for allele in alleles}

            # prepare results dictionary
            scores = defaultdict(defaultdict)

            # keep input peptide objects for later use
            peptide_objects = {}
            for peptide in peptides:
                peptide_objects[str(peptide)] = peptide

            # group peptides by length
            pep_groups = list(peptide_objects.keys())
            pep_groups.sort(key=len)
            for length, peps in itertools.groupby(pep_groups, key=len):
                if length not in self.supportedLength:
                    logging.warning("Peptide length must be at least %i or at most %i for %s but is %i" % (min(self.supportedLength), max(self.supportedLength),
                                                                                        self.name, length))
                    continue
                peps = list(peps)

                # write peptides temporarily, new line separated
                tmp_input_file = tempfile.NamedTemporaryFile().name
                with open(tmp_input_file, 'w') as file:
                    for peptide in peps:
                        file.write(peptide + "\n")

                # predict binding affinities
                for a in alleles:
                    # workaround for mhcnuggets file i/o buffer bug
                    mhcnuggets_output = io.StringIO()
                    with capture_stdout(mhcnuggets_output):
                        mhcnuggets_predict(class_='I',
                                            peptides_path=tmp_input_file,
                                            mhc=alleles_repr[a])

                    # read predicted binding affinities back
                    mhcnuggets_output.seek(0)
                    reader = csv.reader(mhcnuggets_output, delimiter=' ', quotechar='|')
                    # skip log statements from mhcnuggets and header
                    for row in reader:
                        if row[0] == 'peptide,ic50':
                            break
                        logging.info(' '.join(row))

                    # assign binding affinities
                    for row in reader:
                        content = row[0].split(',')
                        # get original peptide object
                        peptide = content[0]
                        binding_affinity = float(content[ScoreIndex.MHCNUGGETS_CLASS1_2_0])
                        if binary:
                            if binding_affinity <= 500:
                                scores[a][peptide] = 1.0
                            else:
                                scores[a][peptide] = 0.0
                        else:
                            # convert ic50 to raw prediction score
                            scores[a][peptide] = 1- math.log(binding_affinity, 50000)

            if not scores:
                raise ValueError("No predictions could be made with " + self.name +
                                " for given input. Check your epitope length and HLA allele combination.")
            
            # Create dictionary with hierarchy: {'Allele1': {'Score': {'Pep1': AffScore1, 'Pep2': AffScore2,..}, 'Allele2':...}
            result = {alleles: {"Score":(list(scores.values())[j])} for j, alleles in enumerate(alleles)}

            # create EpitopePredictionResult object. This is a multi-indexed DataFrame
            # with Allele, Method and Score type as multi-columns and peptides as rows
            df_result = EpitopePredictionResult.from_dict(result, peptide_objects.values(), self.name)

            return df_result

except BadSignatureException:
    logging.warning("Class MHCNuggetsPredictor_class1_2_0 cannot be constructed, because of a bad method signature (predict)")


try:
    class MHCNuggetsPredictor_class1_2_3_2(MHCNuggetsPredictor_class1_2_0):
        """
        Implements MHCNuggets Class I

        .. note::
            Evaluation of machine learning methods to predict peptide binding to MHC Class I proteins
            Rohit Bhattacharya, Ashok Sivakumar, Collin Tokheim, Violeta Beleva Guthrie, Valsamo Anagnostou,
            Victor E. Velculescu, Rachel Karchin (2017) bioRxiv
        """
<<<<<<< HEAD
        __alleles = frozenset(["HLA-A*01:01", "HLA-A*02:01", "HLA-A*02:02", "HLA-A*02:03", "HLA-A*02:05", "HLA-A*02:06", "HLA-A*02:07",
                               "HLA-A*02:11", "HLA-A*02:12", "HLA-A*02:16", "HLA-A*02:17", "HLA-A*02:19", "HLA-A*02:50", "HLA-A*03:01",
                               "HLA-A*03:19", "HLA-A*11:01", "HLA-A*23:01", "HLA-A*24:01", "HLA-A*24:02", "HLA-A*24:03", "HLA-A*25:01",
                               "HLA-A*26:01", "HLA-A*26:02", "HLA-A*26:03", "HLA-A*29:02", "HLA-A*30:01", "HLA-A*30:02", "HLA-A*31:01",
                               "HLA-A*32:01", "HLA-A*32:07", "HLA-A*32:15", "HLA-A*33:01", "HLA-A*66:01", "HLA-A*68:01", "HLA-A*68:02",
                               "HLA-A*68:23", "HLA-A*69:01", "HLA-A*80:01", "HLA-B*07:01", "HLA-B*07:02", "HLA-B*08:01", "HLA-B*08:02",
                               "HLA-B*08:03", "HLA-B*14:01", "HLA-B*14:02", "HLA-B*15:01", "HLA-B*15:02", "HLA-B*15:03", "HLA-B*15:09",
                               "HLA-B*15:16", "HLA-B*15:17", "HLA-B*15:42", "HLA-B*18:01", "HLA-B*27:01", "HLA-B*27:02", "HLA-B*27:03",
                               "HLA-B*27:04", "HLA-B*27:05", "HLA-B*27:06", "HLA-B*27:09", "HLA-B*27:20", "HLA-B*35:01", "HLA-B*35:03",
                               "HLA-B*35:08", "HLA-B*37:01", "HLA-B*38:01", "HLA-B*39:01", "HLA-B*39:06", "HLA-B*40:01", "HLA-B*40:02",
                               "HLA-B*40:13", "HLA-B*42:01", "HLA-B*44:01", "HLA-B*44:02", "HLA-B*44:03", "HLA-B*45:01", "HLA-B*45:06",
                               "HLA-B*46:01", "HLA-B*48:01", "HLA-B*51:01", "HLA-B*53:01", "HLA-B*54:01", "HLA-B*57:01", "HLA-B*57:03",
                               "HLA-B*58:01", "HLA-B*58:02", "HLA-B*62:01", "HLA-B*73:01", "HLA-B*81:01", "HLA-B*83:01", "HLA-C*03:03",
                               "HLA-C*03:04", "HLA-C*04:01", "HLA-C*05:01", "HLA-C*06:02", "HLA-C*07:01", "HLA-C*07:02", "HLA-C*08:01",
                               "HLA-C*08:02", "HLA-C*12:03", "HLA-C*14:02", "HLA-C*15:02", "HLA-E*01:03", "H-2-Db", "H-2-Dd", "H-2-Kb",
                               "H-2-Kd", "H-2-Kk", "H-2-Ld"])
        __supported_length = frozenset([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        __name = "mhcnuggets-class-1"
        __version = "2.3.2"

        # the interface defines three class properties
        @property
        def name(self):
            # returns the name of the predictor
=======
        __name = "mhcnuggets-class-1"
        __version = "2.3.2"
        __supported_length = frozenset([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        __allele_import_name = f"{__name}_{__version}".replace('-', '_').replace('.', '_')
        __alleles = getattr(__import__("epytope.Data.supportedAlleles.ann." + __allele_import_name,
                                       fromlist=[__allele_import_name])
                            , __allele_import_name)

        @property
        def name(self):
>>>>>>> repo-b2/main
            return self.__name

        @property
        def supportedAlleles(self):
<<<<<<< HEAD
            # returns the supported alleles as strings (without the HLA prefix)
=======
>>>>>>> repo-b2/main
            return self.__alleles

        @property
        def supportedLength(self):
<<<<<<< HEAD
            # returns the supported epitope lengths as iterable
=======
>>>>>>> repo-b2/main
            return self.__supported_length

        @property
        def version(self):
<<<<<<< HEAD
            # returns the version of the predictor
=======
>>>>>>> repo-b2/main
            return self.__version

        def _represent(self, allele):
            """
            Internal function transforming an allele object into its representative string
            :param allele: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is
                            needed
            :type alleles: :class:`~epytope.Core.Allele.Allele`
            :return: str
            """
            if isinstance(allele, MouseAllele):
                return "%s-%s%s%s" % (allele.organism, allele.locus, allele.supertype, allele.subtype)
            else:
                return "%s-%s%s:%s" % (allele.organism, allele.locus, allele.supertype, allele.subtype)

<<<<<<< HEAD
=======
        def convert_alleles(self, alleles):
            """
            Converts :class:`~epytope.Core.Allele.Allele` into the internal :class:`~epytope.Core.Allele.Allele` representation
            of the predictor and returns a string representation

            :param alleles: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is
                            needed
            :type alleles: :class:`~epytope.Core.Allele.Allele`
            :return: Returns a string representation of the input :class:`~epytope.Core.Allele.Allele`
            :rtype: list(str)
            """
            return [self._represent(a) for a in alleles]

>>>>>>> repo-b2/main
        # Converts the internal mhcnuggets-class-1 HLA representation back into a epytope representation
        def revert_allele_repr(self, name):
            if name.startswith("H-2-"):
                return MouseAllele(name)
            else:
                name = name[:5] + '*' + name[5:]
                return Allele(name)

        # predict(): same workaround for mhcnuggets file i/o buffer bug needed here
except BadSignatureException:
    logging.warning("Class MHCNuggetsPredictor_class1_2_3_2 cannot be constructed, because of a bad method signature (predict)")


try:
    class MHCNuggetsPredictor_class2_2_0(AANNEpitopePrediction):
        """
        Implements MHCNuggets Class II

        .. note::
            Evaluation of machine learning methods to predict peptide binding to MHC Class I proteins
            Rohit Bhattacharya, Ashok Sivakumar, Collin Tokheim, Violeta Beleva Guthrie, Valsamo Anagnostou,
            Victor E. Velculescu, Rachel Karchin (2017) bioRxiv
        """
<<<<<<< HEAD
        __alleles = frozenset(["HLA-DPA1*01:03-DPB1*02:01", "HLA-DPA1*01:03-DPB1*03:01", "HLA-DPA1*01:03-DPB1*04:01",
                               "HLA-DPA1*01:03-DPB1*04:02", "HLA-DPA1*02:01-DPB1*01:01", "HLA-DPA1*02:01-DPB1*05:01",
                               "HLA-DPA1*02:02-DPB1*05:01", "HLA-DPA1*03:01-DPB1*04:02", "HLA-DPB1*01:01",
                               "HLA-DPB1*02:01", "HLA-DPB1*03:01", "HLA-DPB1*04:01", "HLA-DPB1*04:02", "HLA-DPB1*05:01",
                               "HLA-DPB1*09:01", "HLA-DPB1*11:01", "HLA-DPB1*14:01", "HLA-DPB1*20:01", "HLA-DQA1*01:01",
                               "HLA-DQA1*01:01-DQB1*05:01", "HLA-DQA1*01:01-DQB1*05:03", "HLA-DQA1*01:02",
                               "HLA-DQA1*01:02-DQB1*05:01", "HLA-DQA1*01:02-DQB1*05:02", "HLA-DQA1*01:02-DQB1*06:02",
                               "HLA-DQA1*01:02-DQB1*06:04", "HLA-DQA1*01:03-DQB1*03:02", "HLA-DQA1*01:03-DQB1*06:01",
                               "HLA-DQA1*01:03-DQB1*06:03", "HLA-DQA1*01:04-DQB1*05:03", "HLA-DQA1*02:01-DQB1*02:01",
                               "HLA-DQA1*02:01-DQB1*02:02", "HLA-DQA1*02:01-DQB1*03:01", "HLA-DQA1*02:01-DQB1*03:03",
                               "HLA-DQA1*02:01-DQB1*04:02", "HLA-DQA1*03:01", "HLA-DQA1*03:01-DQB1*02:01",
                               "HLA-DQA1*03:01-DQB1*03:01", "HLA-DQA1*03:01-DQB1*03:02", "HLA-DQA1*03:01-DQB1*04:01",
                               "HLA-DQA1*03:02-DQB1*03:01", "HLA-DQA1*03:02-DQB1*03:03", "HLA-DQA1*03:02-DQB1*04:01",
                               "HLA-DQA1*03:03-DQB1*04:02", "HLA-DQA1*04:01-DQB1*04:02", "HLA-DQA1*05:01",
                               "HLA-DQA1*05:01-DQB1*02:01", "HLA-DQA1*05:01-DQB1*03:01", "HLA-DQA1*05:01-DQB1*03:02",
                               "HLA-DQA1*05:01-DQB1*03:03", "HLA-DQA1*05:01-DQB1*04:02", "HLA-DQA1*05:05-DQB1*03:01",
                               "HLA-DQA1*06:01-DQB1*04:02", "HLA-DQB1*02:01", "HLA-DQB1*02:02", "HLA-DQB1*03:01",
                               "HLA-DQB1*03:02", "HLA-DQB1*03:19", "HLA-DQB1*04:02", "HLA-DQB1*05:01", "HLA-DQB1*05:02",
                               "HLA-DQB1*05:03", "HLA-DQB1*06:02", "HLA-DQB1*06:03", "HLA-DQB1*06:04",
                               "HLA-DRA0*10:1-DRB1*01:01", "HLA-DRA0*10:1-DRB1*03:01", "HLA-DRA0*10:1-DRB1*04:01",
                               "HLA-DRA0*10:1-DRB1*04:04", "HLA-DRA0*10:1-DRB1*07:01", "HLA-DRA0*10:1-DRB1*08:01",
                               "HLA-DRA0*10:1-DRB1*09:01", "HLA-DRA0*10:1-DRB1*11:01", "HLA-DRA0*10:1-DRB1*13:01",
                               "HLA-DRA0*10:1-DRB1*14:54", "HLA-DRA0*10:1-DRB1*15:01", "HLA-DRA0*10:1-DRB3*01:01",
                               "HLA-DRA0*10:1-DRB3*02:02", "HLA-DRA0*10:1-DRB3*03:01", "HLA-DRA0*10:1-DRB4*01:03",
                               "HLA-DRA0*10:1-DRB5*01:01", "HLA-DRB1*01:01", "HLA-DRB1*01:02", "HLA-DRB1*01:03",
                               "HLA-DRB1*03:01", "HLA-DRB1*03:02", "HLA-DRB1*03:03", "HLA-DRB1*03:04", "HLA-DRB1*03:05",
                               "HLA-DRB1*04:01", "HLA-DRB1*04:02", "HLA-DRB1*04:03", "HLA-DRB1*04:04", "HLA-DRB1*04:05",
                               "HLA-DRB1*04:06", "HLA-DRB1*04:07", "HLA-DRB1*04:11", "HLA-DRB1*07:01", "HLA-DRB1*08:01",
                               "HLA-DRB1*08:02", "HLA-DRB1*08:03", "HLA-DRB1*08:04", "HLA-DRB1*09:01", "HLA-DRB1*10:01",
                               "HLA-DRB1*11:01", "HLA-DRB1*11:02", "HLA-DRB1*11:03", "HLA-DRB1*11:04", "HLA-DRB1*12:01",
                               "HLA-DRB1*12:02", "HLA-DRB1*13:01", "HLA-DRB1*13:02", "HLA-DRB1*13:03", "HLA-DRB1*13:04",
                               "HLA-DRB1*13:05", "HLA-DRB1*14:01", "HLA-DRB1*14:02", "HLA-DRB1*15:01", "HLA-DRB1*15:02",
                               "HLA-DRB1*15:03", "HLA-DRB1*16:01", "HLA-DRB1*16:02", "HLA-DRB3*01:01", "HLA-DRB3*02:02",
                               "HLA-DRB3*03:01", "HLA-DRB4*01:01", "HLA-DRB4*01:03", "HLA-DRB5*01:01", "HLA-DRB5*01:02",
                               "H-2-Iab", "H-2-Iad", "H-2-Iak", "H-2-Iap", "H-2-Iaq",
                               "H-2-Iar", "H-2-Ias", "H-2-Iau", "H-2-Ieb", "H-2-Ied",
                               "H-2-Iek", "H-2-Iep", "H-2-Ier"])
        __supported_length = frozenset([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
        __name = "mhcnuggets-class-2"
        __version = "2.0"

        # the interface defines three class properties
        @property
        def name(self):
            # returns the name of the predictor
=======
        __name = "mhcnuggets-class-2"
        __version = "2.0"
        __supported_length = frozenset([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
        __allele_import_name = f"{__name}_{__version}".replace('-', '_').replace('.', '_')
        __alleles = getattr(__import__("epytope.Data.supportedAlleles.ann." + __allele_import_name,
                                       fromlist=[__allele_import_name])
                            , __allele_import_name)

        @property
        def name(self):
>>>>>>> repo-b2/main
            return self.__name

        @property
        def supportedAlleles(self):
<<<<<<< HEAD
            # returns the supported alleles as strings (without the HLA prefix)
=======
>>>>>>> repo-b2/main
            return self.__alleles

        @property
        def supportedLength(self):
<<<<<<< HEAD
            # returns the supported epitope lengths as iterable
=======
>>>>>>> repo-b2/main
            return self.__supported_length

        @property
        def version(self):
<<<<<<< HEAD
            # returns the version of the predictor
=======
>>>>>>> repo-b2/main
            return self.__version

        def _represent(self, allele):
            """
            Internal function transforming an allele object into its representative string
            :param allele: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is
                            needed
            :type alleles: :class:`~epytope.Core.Allele.Allele`
            :return: str
            """
            if isinstance(allele, MouseAllele):
                # expects H-2-XXx
                return "%s-%s%s%s" % (allele.organism, allele.locus, allele.supertype.upper(), allele.subtype)
            elif isinstance(allele, CombinedAllele):
                return "%s-%s%s%s-%s%s%s" % (allele.organism, allele.alpha_locus, allele.alpha_supertype, allele.alpha_subtype,
                                        allele.beta_locus, allele.beta_supertype, allele.beta_subtype)
            else:
                return "%s-%s%s%s" % (allele.organism, allele.locus, allele.supertype, allele.subtype)

        # Converts epytopes internal allele representation into the format required by mhcnuggets
        def convert_alleles(self, alleles):
            """
            Converts :class:`~epytope.Core.Allele.Allele` into the internal :class:`~epytope.Core.Allele.Allele` representation
            of the predictor and returns a string representation

            :param alleles: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is
                            needed
            :type alleles: :class:`~epytope.Core.Allele.Allele`
            :return: Returns a string representation of the input :class:`~epytope.Core.Allele.Allele`
            :rtype: list(str)
            """
            return [self._represent(a) for a in alleles]

        # Converts the internal mhcnuggets-class-2 representation back into a epytope representation
        def revert_allele_repr(self, name):
            if name.startswith("H-2-"):
                return MouseAllele(name)
            else:
                # since we need to support single and double mhc2 alleles
                name_split = name.split('-')
                if len(name_split) > 2:
                    return CombinedAllele(name_split[0] + '-' +
                                          name_split[1][:4] + '*' + name_split[1][4:6] + ':' + name_split[1][6:] + '-' +
                                          name_split[2][:4] + '*' + name_split[2][4:6] + ':' + name_split[2][6:])
                else:
                    return Allele(name_split[0] + '-' +
                                  name_split[1][:4] + '*' + name_split[1][4:6] + ':' + name_split[1][6:])

        # predicts the binding affinity for a set of peptides and alleles
        def predict(self, peptides, alleles=None, binary=False, **kwargs):

            # test whether one peptide or a list
            if not isinstance(peptides, list):
                peptides = [peptides]

            # if no alleles are specified do predictions for all supported alleles
            if alleles is None:
                alleles = self.supportedAlleles
            else:
                # filter for supported alleles
                alleles = [a for a in alleles if a in self.supportedAlleles]

            # Create a dictionary with Allele Obj as key and the respective allele predictor representation as value
            alleles_repr = {allele: self._represent(allele) for allele in alleles}

            # prepare results dictionary
            scores = defaultdict(defaultdict)

            # keep input peptide objects for later use
            peptide_objects = {}
            for peptide in peptides:
                peptide_objects[str(peptide)] = peptide

            # group peptides by length
            pep_groups = list(peptide_objects.keys())
            pep_groups.sort(key=len)
            for length, peps in itertools.groupby(pep_groups, key=len):
                if length not in self.supportedLength:
                    logging.warning("Peptide length must be at least %i or at most %i for %s but is %i" % (min(self.supportedLength), max(self.supportedLength),
                                                                                        self.name, length))
                    continue
                peps = list(peps)

                # write peptides temporarily, new line separated
                tmp_input_file = tempfile.NamedTemporaryFile().name
                with open(tmp_input_file, 'w') as file:
                    for peptide in peps:
                        file.write(peptide + "\n")

                # predict bindings
                for a in alleles:
                    # workaround for mhcnuggets file i/o buffer bug
                    mhcnuggets_output = io.StringIO()
                    with capture_stdout(mhcnuggets_output):
                        mhcnuggets_predict(class_='II',
                                        peptides_path=tmp_input_file,
                                        mhc=alleles_repr[a])

                    # read predicted binding affinities back
                    mhcnuggets_output.seek(0)
                    reader = csv.reader(mhcnuggets_output, delimiter=' ', quotechar='|')
                    # skip log statements from mhcnuggets and header
                    for row in reader:
                        if row[0] == 'peptide,ic50':
                            break
                        logging.warning(' '.join(row))

                    for row in reader:
                        content = row[0].split(',')
                        # get original peptide object
                        peptide = peptide_objects[content[0]]
                        binding_affinity = float(content[ScoreIndex.MHCNUGGETS_CLASS2_2_0])
                        if binary:
                            if binding_affinity <= 500:
                                scores[a][peptide] = 1.0
                            else:
                                scores[a][peptide] = 0.0
                        else:
                            # convert ic50 to raw prediction score
                            scores[a][peptide] = 1- math.log(binding_affinity, 50000)

            if not scores:
                raise ValueError("No predictions could be made with " + self.name +
                                " for given input. Check your epitope length and HLA allele combination.")

            # Create dictionary with hierarchy: {'Allele1': {'Score': {'Pep1': AffScore1, 'Pep2': AffScore2,..}, 'Allele2':...}
            result = {allele: {"Score":(list(scores.values())[j])} for j, allele in enumerate(alleles)}

            # create EpitopePredictionResult object. This is a multi-indexed DataFrame
            # with Allele, Method and Score type as multi-columns and peptides as rows
            df_result = EpitopePredictionResult.from_dict(result, peptide_objects.values(), self.name)
            return df_result

except BadSignatureException:
    logging.warning("Class MHCNuggetsPredictor_class2_2_0 cannot be constructed, because of a bad method signature (predict)")

try:
    class MHCNuggetsPredictor_class2_2_3_2(MHCNuggetsPredictor_class2_2_0):
        """
        Implements MHCNuggets Class II

        .. note::
            Evaluation of machine learning methods to predict peptide binding to MHC Class I proteins
            Rohit Bhattacharya, Ashok Sivakumar, Collin Tokheim, Violeta Beleva Guthrie, Valsamo Anagnostou,
            Victor E. Velculescu, Rachel Karchin (2017) bioRxiv
        """
<<<<<<< HEAD
        __alleles = frozenset(["HLA-DPA1*01:03-DPB1*02:01", "HLA-DPA1*01:03-DPB1*03:01", "HLA-DPA1*01:03-DPB1*04:01",
                               "HLA-DPA1*01:03-DPB1*04:02", "HLA-DPA1*02:01-DPB1*01:01", "HLA-DPA1*02:01-DPB1*05:01",
                               "HLA-DPA1*02:02-DPB1*05:01", "HLA-DPA1*03:01-DPB1*04:02", "HLA-DPB1*01:01",
                               "HLA-DPB1*02:01", "HLA-DPB1*03:01", "HLA-DPB1*04:01",
                               "HLA-DPB1*04:02", "HLA-DPB1*05:01", "HLA-DPB1*09:01",
                               "HLA-DPB1*11:01", "HLA-DPB1*14:01", "HLA-DPB1*20:01",
                               "HLA-DQA1*01:01-DQB1*05:01", "HLA-DQA1*01:02-DQB1*05:01", "HLA-DQA1*01:02-DQB1*05:02",
                               "HLA-DQA1*01:02-DQB1*06:02", "HLA-DQA1*01:02-DQB1*06:04", "HLA-DQA1*01:03-DQB1*06:01",
                               "HLA-DQA1*01:03-DQB1*06:03", "HLA-DQA1*01:04-DQB1*05:03", "HLA-DQA1*02:01-DQB1*02:01",
                               "HLA-DQA1*02:01-DQB1*02:02", "HLA-DQA1*02:01-DQB1*03:01", "HLA-DQA1*02:01-DQB1*03:03",
                               "HLA-DQA1*02:01-DQB1*04:02", "HLA-DQA1*03:01-DQB1*02:01", "HLA-DQA1*03:01-DQB1*03:01",
                               "HLA-DQA1*03:01-DQB1*03:02", "HLA-DQA1*03:01-DQB1*04:01", "HLA-DQA1*03:02-DQB1*03:01",
                               "HLA-DQA1*03:02-DQB1*04:01", "HLA-DQA1*03:03-DQB1*04:02", "HLA-DQA1*04:01-DQB1*04:02",
                               "HLA-DQA1*05:01-DQB1*02:01", "HLA-DQA1*05:01-DQB1*03:01", "HLA-DQA1*05:01-DQB1*03:02",
                               "HLA-DQA1*05:01-DQB1*03:03", "HLA-DQA1*05:01-DQB1*04:02", "HLA-DQA1*05:05-DQB1*03:01",
                               "HLA-DQA1*06:01-DQB1*04:02", "HLA-DQB1*02:01", "HLA-DQB1*02:02",
                               "HLA-DQB1*03:01", "HLA-DQB1*03:02", "HLA-DQB1*03:19",
                               "HLA-DQB1*04:02", "HLA-DQB1*05:01", "HLA-DQB1*05:02",
                               "HLA-DQB1*05:03", "HLA-DQB1*06:02", "HLA-DRB1*01:01",
                               "HLA-DRB1*01:02", "HLA-DRB1*01:03", "HLA-DRB1*03:01",
                               "HLA-DRB1*03:02", "HLA-DRB1*03:03", "HLA-DRB1*03:05",
                               "HLA-DRB1*04:01", "HLA-DRB1*04:02", "HLA-DRB1*04:03",
                               "HLA-DRB1*04:04", "HLA-DRB1*04:05", "HLA-DRB1*04:06",
                               "HLA-DRB1*04:07", "HLA-DRB1*04:11", "HLA-DRB1*07:01",
                               "HLA-DRB1*08:01", "HLA-DRB1*08:02", "HLA-DRB1*08:03",
                               "HLA-DRB1*09:01", "HLA-DRB1*10:01", "HLA-DRB1*11:01",
                               "HLA-DRB1*11:02", "HLA-DRB1*11:03", "HLA-DRB1*11:04",
                               "HLA-DRB1*12:01", "HLA-DRB1*12:02", "HLA-DRB1*13:01",
                               "HLA-DRB1*13:02", "HLA-DRB1*13:03", "HLA-DRB1*13:04",
                               "HLA-DRB1*14:01", "HLA-DRB1*14:02", "HLA-DRB1*15:01",
                               "HLA-DRB1*15:02", "HLA-DRB1*15:03", "HLA-DRB1*16:01",
                               "HLA-DRB1*16:02", "HLA-DRB3*01:01", "HLA-DRB3*02:02",
                               "HLA-DRB3*03:01", "HLA-DRB4*01:01", "HLA-DRB4*01:03",
                               "HLA-DRB5*01:01", "HLA-DRB5*01:02", "H-2-Iab",
                               "H-2-Iad", "H-2-Iak", "H-2-Iap",
                               "H-2-Iaq", "H-2-Iar", "H-2-Ias",
                               "H-2-Iau", "H-2-Ieb", "H-2-Ied",
                               "H-2-Iek", "H-2-Iep", "H-2-Ier"])
        __supported_length = frozenset([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
        __name = "mhcnuggets-class-2"
        __version = "2.3.2"

        # the interface defines three class properties
        @property
        def name(self):
            # returns the name of the predictor
=======
        __name = "mhcnuggets-class-2"
        __version = "2.3.2"
        __supported_length = frozenset(
            [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
        __allele_import_name = f"{__name}_{__version}".replace('-', '_').replace('.', '_')
        __alleles = getattr(__import__("epytope.Data.supportedAlleles.ann." + __allele_import_name,
                                       fromlist=[__allele_import_name])
                            , __allele_import_name)

        @property
        def name(self):
>>>>>>> repo-b2/main
            return self.__name

        @property
        def supportedAlleles(self):
<<<<<<< HEAD
            # returns the supported alleles as strings (without the HLA prefix)
=======
>>>>>>> repo-b2/main
            return self.__alleles

        @property
        def supportedLength(self):
<<<<<<< HEAD
            # returns the supported epitope lengths as iterable
=======
>>>>>>> repo-b2/main
            return self.__supported_length

        @property
        def version(self):
<<<<<<< HEAD
            # returns the version of the predictor
=======
>>>>>>> repo-b2/main
            return self.__version

        def _represent(self, allele):
            """
            Internal function transforming an allele object into its representative string
            :param allele: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is
                            needed
            :type alleles: :class:`~epytope.Core.Allele.Allele`
            :return: str
            """
            if isinstance(allele, MouseAllele):
                # expects H-2-XXx
                return "%s-%s%s%s" % (allele.organism, allele.locus, allele.supertype.upper(), allele.subtype)
            elif isinstance(allele, CombinedAllele):
                return "%s-%s%s:%s-%s%s:%s" % (allele.organism, allele.alpha_locus, allele.alpha_supertype, allele.alpha_subtype,
                                        allele.beta_locus, allele.beta_supertype, allele.beta_subtype)
            else:
                return "%s-%s%s:%s" % (allele.organism, allele.locus, allele.supertype, allele.subtype)

<<<<<<< HEAD
=======
        def convert_alleles(self, alleles):
            """
            Converts :class:`~epytope.Core.Allele.Allele` into the internal :class:`~epytope.Core.Allele.Allele` representation
            of the predictor and returns a string representation

            :param alleles: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is
                            needed
            :type alleles: :class:`~epytope.Core.Allele.Allele`
            :return: Returns a string representation of the input :class:`~epytope.Core.Allele.Allele`
            :rtype: list(str)
            """
            return [self._represent(a) for a in alleles]

>>>>>>> repo-b2/main
        # Converts the internal mhcnuggets-class-2 representation back into a epytope representation
        def revert_allele_repr(self, name):
            if name.startswith("H-2-"):
                return MouseAllele(name)
            else:
                # since we need to support single and double mhc2 alleles
                name_split = name.split('-')
                if len(name_split) > 2:
                    return CombinedAllele(name_split[0] + '-' +
                            name_split[1][:4] + '*' + name_split[1][4:]  + '-' +
                            name_split[2][:4] + '*' + name_split[2][4:])
                else:
                    return Allele(name_split[0] + '-' + name_split[1][:4] + '*' + name_split[1][4:])

        # predict(): same workaround for mhcnuggets file i/o buffer bug needed here
except BadSignatureException:
    logging.warning("Class MHCNuggetsPredictor_class2_2_3_2 cannot be constructed, because of a bad method signature (predict)")


try:
    class MHCFlurryPredictor_1_2_2(AANNEpitopePrediction):
        """
        Implements MHCFlurry

        .. note::
            T. J. O’Donnell, A. Rubinsteyn, M. Bonsack, A. B. Riemer, U. Laserson, and J. Hammerbacher,
             "MHCflurry: Open-Source Class I MHC Binding Affinity Prediction," Cell Systems, 2018.
              Available at: https://www.cell.com/cell-systems/fulltext/S2405-4712(18)30232-1.
        """
<<<<<<< HEAD
        __alleles = frozenset(
            ["HLA-A*01:01", "HLA-A*02:01", "HLA-A*02:02", "HLA-A*02:03", "HLA-A*02:05", "HLA-A*02:06", "HLA-A*02:07",
             "HLA-A*02:11", "HLA-A*02:12", "HLA-A*02:16", "HLA-A*02:17", "HLA-A*02:19", "HLA-A*02:50", "HLA-A*03:01",
             "HLA-A*11:01", "HLA-A*23:01", "HLA-A*24:02", "HLA-A*24:03", "HLA-A*25:01", "HLA-A*26:01", "HLA-A*26:02",
             "HLA-A*26:03", "HLA-A*29:02", "HLA-A*30:01", 'HLA-A*30:02', "HLA-A*31:01", "HLA-A*32:01", "HLA-A*33:01",
             "HLA-A*66:01", "HLA-A*68:01", "HLA-A*68:02", "HLA-A*68:23", "HLA-A*69:01", "HLA-A*80:01", "HLA-B*07:01",
             "HLA-B*07:02", "HLA-B*08:01", "HLA-B*08:02", "HLA-B*08:03", "HLA-B*14:02", "HLA-B*15:01", "HLA-B*15:02",
             "HLA-B*15:03", "HLA-B*15:09", "HLA-B*15:17", "HLA-B*18:01", "HLA-B*27:02", "HLA-B*27:03", "HLA-B*27:04",
             "HLA-B*27:05", "HLA-B*27:06", "HLA-B*35:01", "HLA-B*35:03", "HLA-B*37:01", "HLA-B*38:01", "HLA-B*39:01",
             "HLA-B*39:06", "HLA-B*40:01", "HLA-B*40:02", "HLA-B*42:01", "HLA-B*44:02", "HLA-B*44:03", "HLA-B*45:01",
             "HLA-B*46:01", "HLA-B*48:01", "HLA-B*51:01", "HLA-B*53:01", "HLA-B*54:01", "HLA-B*57:01", "HLA-B*58:01",
             "HLA-B*83:01", "HLA-C*03:03", "HLA-C*04:01", "HLA-C*05:01", "HLA-C*06:02", "HLA-C*07:02", "HLA-C*08:02",
             "HLA-C*12:03", "HLA-C*14:02", "HLA-C*15:02", "H-2-Db", "H-2-Dd", "H-2-Kb", "H-2-Kd",
             "H-2-Kk", "H-2-Ld"])
        __supported_length = frozenset([8, 9, 10, 11, 12, 13, 14, 15])
        __name = "mhcflurry"
        __version = "1.2.2"

        # the interface defines three class properties
        @property
        def name(self):
            # returns the name of the predictor
=======
        __name = "mhcflurry"
        __version = "1.2.2"
        __supported_length = frozenset([8, 9, 10, 11, 12, 13, 14, 15])
        __allele_import_name = f"{__name}_{__version}".replace('.', '_')
        __alleles = getattr(__import__("epytope.Data.supportedAlleles.ann." + __allele_import_name,
                                       fromlist=[__allele_import_name])
                            , __allele_import_name)

        @property
        def name(self):
>>>>>>> repo-b2/main
            return self.__name

        @property
        def supportedAlleles(self):
<<<<<<< HEAD
            # returns the supported alleles as strings (without the HLA prefix)
=======
>>>>>>> repo-b2/main
            return self.__alleles

        @property
        def supportedLength(self):
<<<<<<< HEAD
            # returns the supported epitope lengths as iterable
=======
>>>>>>> repo-b2/main
            return self.__supported_length

        @property
        def version(self):
<<<<<<< HEAD
            # returns the version of the predictor
=======
>>>>>>> repo-b2/main
            return self.__version

        def _represent(self, allele):
            """
            Internal function transforming an allele object into its representative string
            :param allele: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is
                            needed
            :type alleles: :class:`~epytope.Core.Allele.Allele`
            :return: str
            """
            if isinstance(allele, MouseAllele):
                return "%s-%s%s%s" % (allele.organism, allele.locus, allele.supertype, allele.subtype)
            else:
                return "%s-%s%s%s" % (allele.organism, allele.locus, allele.supertype, allele.subtype)

        # Converts epytopes internal allele representation into the format required by MHCFlurry
        def convert_alleles(self, alleles):
            """
            Converts :class:`~epytope.Core.Allele.Allele` into the internal :class:`~epytope.Core.Allele.Allele` representation
            of the predictor and returns a string representation

            :param alleles: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is
                            needed
            :type alleles: :class:`~epytope.Core.Allele.Allele`
            :return: Returns a string representation of the input :class:`~epytope.Core.Allele.Allele`
            :rtype: list(str)
            """
            return [self._represent(a) for a in alleles]

        # Converts the internal MHCFlurry representation back into a epytope representation
        def revert_allele_repr(self, name):
            if name.startswith("H-2-"):
                return MouseAllele(name)
            else:
                return Allele(name[:5] + '*' + name[5:7] + ':' + name[7:])

        # predicts the binding affinity for a set of peptides and alleles
        def predict(self, peptides, alleles=None, binary=False, **kwargs):

            # test whether one peptide or a list
            if not isinstance(peptides, list):
                peptides = [peptides]

            # if no alleles are specified do predictions for all supported alleles
            if alleles is None:
                alleles = self.supportedAlleles
            else:
                # filter for supported alleles
                alleles = [a for a in alleles if a in self.supportedAlleles]

            # Create a dictionary with Allele Obj as key and the respective allele predictor representation as value
            alleles_repr = {allele: self._represent(allele) for allele in alleles}

            # test mhcflurry models are available => download if not
            p = subprocess.call(['mhcflurry-downloads', 'path', 'models_class1'],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if p != 0:
                logging.warn("mhcflurry models must be downloaded, as they were not found locally.")
                cp = subprocess.run(['mhcflurry-downloads', 'fetch', 'models_class1'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                if cp.returncode != 0:
                    for line in cp.stdout.decode().splitlines():
                        logging.error(line)
                    raise RuntimeError("mhcflurry failed to download model file")

            # load model
            predictor = Class1AffinityPredictor.load()

            # prepare results dictionary
            scores = defaultdict(defaultdict)

            # keep input peptide objects for later use
            peptide_objects = {}
            for peptide in peptides:
                peptide_objects[str(peptide)] = peptide

            # group peptides by length
            pep_groups = list(peptide_objects.keys())
            pep_groups.sort(key=len)

            for length, peps in itertools.groupby(peptides, key=len):
                if length not in self.supportedLength:
                    logging.warning("Peptide length must be at least %i or at most %i for %s but is %i" % (min(self.supportedLength), max(self.supportedLength),
                                                                                        self.name, length))
                    continue
                peps = list(peps)

                # predict and assign binding affinities
                for a in alleles:
                    for p in peps:
                        binding_affinity = predictor.predict(allele=alleles_repr[a], peptides=[str(p)])[ScoreIndex.MHCFLURRY]
                        if binary:
                            if binding_affinity <= 500:
                                scores[a][p] = 1.0
                            else:
                                scores[a][p] = 0.0
                        else:
                            # convert ic50 to raw prediction score
                            scores[a][p] = 1- math.log(binding_affinity, 50000)

            if not scores:
                raise ValueError("No predictions could be made with " + self.name +
                                " for given input. Check your epitope length and HLA allele combination.")
                                
            # Create dictionary with hierarchy: {'Allele1': {'Score': {'Pep1': AffScore1, 'Pep2': AffScore2,..}, 'Allele2':...}
            result = {allele: {"Score":(list(scores.values())[j])} for j, allele in enumerate(alleles)}

            # create EpitopePredictionResult object. This is a multi-indexed DataFrame
            # with Allele, Method and Score type as multi-columns and peptides as rows
            df_result = EpitopePredictionResult.from_dict(result, peptide_objects.values(), self.name)
            return df_result

except BadSignatureException:
    logging.warning("Class MHCFlurryPredictor_1_2_2 cannot be constructed, because of a bad method signature (predict)")


try:
    class MHCFlurryPredictor_1_4_3(MHCFlurryPredictor_1_2_2):
        """
        Implements MHCFlurry

        .. note::
            T. J. O’Donnell, A. Rubinsteyn, M. Bonsack, A. B. Riemer, U. Laserson, and J. Hammerbacher,
             "MHCflurry: Open-Source Class I MHC Binding Affinity Prediction," Cell Systems, 2018.
              Available at: https://www.cell.com/cell-systems/fulltext/S2405-4712(18)30232-1.
        """
<<<<<<< HEAD
        # retrieved with `mhcflurry-predict --list-supported-alleles`
        __alleles = frozenset(["HLA-A*01:01", "HLA-A*02:01", "HLA-A*02:02", "HLA-A*02:03", "HLA-A*02:05",
                               "HLA-A*02:06", "HLA-A*02:07", "HLA-A*02:11", "HLA-A*02:12", "HLA-A*02:16",
                               "HLA-A*02:17", "HLA-A*02:19", "HLA-A*02:50", "HLA-A*03:01", "HLA-A*11:01",
                               "HLA-A*23:01", "HLA-A*24:02", "HLA-A*24:03", "HLA-A*25:01", "HLA-A*26:01",
                               "HLA-A*26:02", "HLA-A*26:03", "HLA-A*29:02", "HLA-A*30:01", "HLA-A*30:02",
                               "HLA-A*31:01", "HLA-A*32:01", "HLA-A*33:01", "HLA-A*66:01", "HLA-A*68:01",
                               "HLA-A*68:02", "HLA-A*68:23", "HLA-A*69:01", "HLA-A*80:01", "HLA-B*07:01",
                               "HLA-B*07:02", "HLA-B*08:01", "HLA-B*08:02", "HLA-B*08:03", "HLA-B*14:02",
                               "HLA-B*15:01", "HLA-B*15:02", "HLA-B*15:03", "HLA-B*15:09", "HLA-B*15:17",
                               "HLA-B*18:01", "HLA-B*27:02", "HLA-B*27:03", "HLA-B*27:04", "HLA-B*27:05",
                               "HLA-B*27:06", "HLA-B*35:01", "HLA-B*35:03", "HLA-B*37:01", "HLA-B*38:01",
                               "HLA-B*39:01", "HLA-B*39:06", "HLA-B*40:01", "HLA-B*40:02", "HLA-B*42:01",
                               "HLA-B*44:02", "HLA-B*44:03", "HLA-B*45:01", "HLA-B*46:01", "HLA-B*48:01",
                               "HLA-B*51:01", "HLA-B*53:01", "HLA-B*54:01", "HLA-B*57:01", "HLA-B*58:01",
                               "HLA-B*83:01", "HLA-C*03:03", "HLA-C*04:01", "HLA-C*05:01", "HLA-C*06:02",
                               "HLA-C*07:02", "HLA-C*08:02", "HLA-C*12:03", "HLA-C*14:02", "HLA-C*15:02",
                               "H-2-Db", "H-2-Dd", "H-2-Kb", "H-2-Kd", "H-2-Kk",
                               "H-2-Ld"])
        # retrieved with `mhcflurry-predict --list-supported-peptide-lengths`
        __supported_length = frozenset([8, 9, 10, 11, 12, 13, 14, 15])
        __name = "mhcflurry"
        __version = "1.4.3"

        # the interface defines three class properties
        @property
        def name(self):
            # returns the name of the predictor
=======
        __name = "mhcflurry"
        __version = "1.4.3"
        __supported_length = frozenset([8, 9, 10, 11, 12, 13, 14, 15])
        __allele_import_name = f"{__name}_{__version}".replace('.', '_')
        __alleles = getattr(__import__("epytope.Data.supportedAlleles.ann." + __allele_import_name,
                                       fromlist=[__allele_import_name])
                            , __allele_import_name)

        @property
        def name(self):
>>>>>>> repo-b2/main
            return self.__name

        @property
        def supportedAlleles(self):
<<<<<<< HEAD
            # returns the supported alleles as strings (without the HLA prefix)
=======
>>>>>>> repo-b2/main
            return self.__alleles

        @property
        def supportedLength(self):
<<<<<<< HEAD
            # returns the supported epitope lengths as iterable
=======
>>>>>>> repo-b2/main
            return self.__supported_length

        @property
        def version(self):
<<<<<<< HEAD
            # returns the version of the predictor
=======
>>>>>>> repo-b2/main
            return self.__version

        def _represent(self, allele):
            """
            Internal function transforming an allele object into its representative string
            :param allele: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is
                            needed
            :type alleles: :class:`~epytope.Core.Allele.Allele`
            :return: str
            """
            if isinstance(allele, MouseAllele):
                return "%s-%s%s%s" % (allele.organism, allele.locus, allele.supertype, allele.subtype)
            else:
                return "%s-%s*%s:%s" % (allele.organism, allele.locus, allele.supertype, allele.subtype)

<<<<<<< HEAD
=======
        def convert_alleles(self, alleles):
            """
            Converts :class:`~epytope.Core.Allele.Allele` into the internal :class:`~epytope.Core.Allele.Allele` representation
            of the predictor and returns a string representation

            :param alleles: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is
                            needed
            :type alleles: :class:`~epytope.Core.Allele.Allele`
            :return: Returns a string representation of the input :class:`~epytope.Core.Allele.Allele`
            :rtype: list(str)
            """
            return [self._represent(a) for a in alleles]

>>>>>>> repo-b2/main
        # Converts the internal MHCFlurry representation back into a epytope representation
        def revert_allele_repr(self, name):
            if name.startswith("H-2-"):
                return MouseAllele(name)
            else:
                return Allele(name)
except BadSignatureException:
    logging.warning("Class MHCFlurryPredictor_1_4_3 cannot be constructed, because of a bad method signature (predict)")



class ScoreIndex(IntEnum):
    """
    Specifies the score column index in the respective output format
    """
    MHCNUGGETS_CLASS1_2_0 = 1
    MHCNUGGETS_CLASS2_2_0 = 1
    MHCFLURRY = 0
