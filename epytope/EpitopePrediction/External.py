# This code is part of the epytope distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
.. module:: EpitopePrediction.ANN
   :synopsis: This module contains all classes for ANN-based epitope prediction methods.
.. moduleauthor:: schubert, walzer

"""
import abc

import itertools
import warnings
import logging
import pandas
import subprocess
import csv
import os
import math
import re

from collections import defaultdict
from enum import IntEnum

from epytope.Core.Allele import Allele, CombinedAllele, MouseAllele
from epytope.Core.Peptide import Peptide
from epytope.Core.Result import EpitopePredictionResult
from epytope.Core.Base import AEpitopePrediction, AExternal
from tempfile import NamedTemporaryFile, mkstemp

class AExternalEpitopePrediction(AEpitopePrediction, AExternal):
    """
        Abstract class representing an external prediction function. Implementations shall wrap external binaries by
        following the given abstraction.
    """

    @abc.abstractmethod
    def prepare_input(self, input, file):
        """
        Prepares input for external tools
        and writes them to _file in the specific format

        NO return value!

        :param: list(str) _input: The :class:`~epytope.Core.Peptide.Peptide` sequences to write into file
        :param File file: File-handler to input file for external tool
        """
        return NotImplementedError

    def predict(self, peptides, alleles=None, command=None, options=None, **kwargs):
        """
        Overwrites AEpitopePrediction.predict

        :param peptides: A list of or a single :class:`~epytope.Core.Peptide.Peptide` object
        :type peptides: list(:class:`~epytope.Core.Peptide.Peptide`) or :class:`~epytope.Core.Peptide.Peptide`
        :param alleles: A list of or a single :class:`~epytope.Core.Allele.Allele` object. If no
                        :class:`~epytope.Core.Allele.Allele` are provided, predictions are made for all
                        :class:`~epytope.Core.Allele.Allele` supported by the prediction method
        :type alleles: list(:class:`~epytope.Core.Allele.Allele`)/:class:`~epytope.Core.Allele.Allele`
        :param str command: The path to a alternative binary (can be used if binary is not globally executable)
        :param str options: A string of additional options directly past to the external tool.
        :keyword chunksize: denotes the chunksize in which the number of peptides are bulk processed
        :return: A :class:`~epytope.Core.Result.EpitopePredictionResult` object
        :rtype: :class:`~epytope.Core.Result.EpitopePredictionResult`
        """
        if not self.is_in_path() and command is None:
            raise RuntimeError("{name} {version} could not be found in PATH".format(name=self.name,
                                                                                    version=self.version))
        external_version = self.get_external_version(path=command)
        if self.version != external_version and external_version is not None:
            raise RuntimeError("Internal version {internal_version} does "
                               "not match external version {external_version}".format(internal_version=self.version,
                                                                                      external_version=external_version))

        if isinstance(peptides, Peptide):
            pep_seqs = {str(peptides): peptides}
        else:
            pep_seqs = {}
            for p in peptides:
                if not isinstance(p, Peptide):
                    raise ValueError("Input is not of type Protein or Peptide")
                pep_seqs[str(p)] = p

        chunksize = len(pep_seqs)
        if 'chunks' in kwargs:
            chunksize = kwargs['chunks']

        if alleles is None:
            alleles = [Allele(a) for a in self.supportedAlleles]
        else:
            if isinstance(alleles, Allele):
                alleles = [alleles]
            if any(not isinstance(p, Allele) for p in alleles):
                raise ValueError("Input is not of type Allele")

        # Create dictionary containing the predictors string representation and the Allele Obj representation of the allele
        alleles_string = {conv_a: a for conv_a, a in zip(self.convert_alleles(alleles), alleles)}
        
        # Create empty result dictionary to fill downstream
        result = {}

        # group alleles in blocks of 80 alleles (NetMHC can't deal with more)
        _MAX_ALLELES = 50

        # allow custom executable specification
        if command is not None:
            exe = self.command.split()[0]
            _command = self.command.replace(exe, command)
        else:
            _command = self.command

        allele_groups = []
        c_a = 0
        allele_group = []
        for a in alleles_string.keys():
            if c_a >= _MAX_ALLELES:
                c_a = 0
                allele_groups.append(allele_group)
                if str(alleles_string[a]) not in self.supportedAlleles:
                    logging.warning("Allele %s is not supported by %s" % (str(alleles_string[a]), self.name))
                    allele_group = []
                    continue
                allele_group = [a]
            else:
                if str(alleles_string[a]) not in self.supportedAlleles:
                    logging.warning("Allele %s is not supported by %s" % (str(alleles_string[a]), self.name))
                    continue
                allele_group.append(a)
                c_a += 1

        if len(allele_group) > 0:
            allele_groups.append(allele_group)
        # export peptides to peptide list

        pep_groups = list(pep_seqs.keys())
        pep_groups.sort(key=len)
        for length, peps in itertools.groupby(pep_groups, key=len):
            if length not in self.supportedLength:
                logging.warning("Peptide length must be at least %i or at most %i for %s but is %i" % (min(self.supportedLength), max(self.supportedLength),
                                                                                       self.name, length))
                continue
            peps = list(peps)
            
            for i in range(0, len(peps), chunksize):
                # Create a temporary file for subprocess to write to. The
                # handle is not needed on the python end, as only the path will
                # be passed to the subprocess.
                _, tmp_out_path = mkstemp()
                # Create a temporary file to be used for the peptide input
                tmp_file = NamedTemporaryFile(mode="r+", delete=False)
                self.prepare_input(peps[i:i+chunksize], tmp_file)
                tmp_file.close()

                # generate cmd command
                for allele_group in allele_groups:
                    try:
                        stdo = None
                        stde = None
                        cmd = _command.format(peptides=tmp_file.name, alleles=",".join(allele_group),
                                              options="" if options is None else options, out=tmp_out_path, length=str(length))
                        p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT)
                        stdo, stde = p.communicate()
                        stdr = p.returncode
                        if stdr > 0:
                            raise RuntimeError("Unsuccessful execution of " + cmd + " (EXIT!=0) with output:\n" + stdo.decode())
                        if os.path.getsize(tmp_out_path) == 0:
                            raise RuntimeError("Unsuccessful execution of " + cmd + " (empty output file) with output:\n" + stdo.decode())
                    except Exception as e:
                        raise RuntimeError(e)

                    # Obtain parsed output dataframe containing the peptide scores and/or ranks
                    res_tmp = self.parse_external_result(tmp_out_path)
                    for allele_string, scores in res_tmp.items():
                        allele = alleles_string[allele_string]
                        if allele not in result.keys():
                            result[allele] = {}
                        for scoretype, pep_scores in scores.items():
                            if scoretype not in result[allele].keys():
                                result[allele][scoretype] = {}
                            for pep, score in pep_scores.items():
                                result[allele][scoretype][pep_seqs[pep]] = score
                                
                os.remove(tmp_file.name)
                os.remove(tmp_out_path)
        
        if not result:
            raise ValueError("No predictions could be made with " + self.name +
                             " for given input. Check your epitope length and HLA allele combination.")
        
        df_result = EpitopePredictionResult.from_dict(result, list(pep_seqs.values()), self.name)

        return df_result


class NetMHC_3_4(AExternalEpitopePrediction):
    """
    Implements the NetMHC binding (in current form for netMHC3.4).

    .. note::

        NetMHC-3.0: accurate web accessible predictions of human, mouse and monkey MHC class I affinities for peptides
        of length 8-11. Lundegaard C, Lamberth K, Harndahl M, Buus S, Lund O, Nielsen M.
        Nucleic Acids Res. 1;36(Web Server issue):W509-12. 2008

        Accurate approximation method for prediction of class I MHC affinities for peptides of length 8, 10 and 11 using
        prediction tools trained on 9mers. Lundegaard C, Lund O, Nielsen M. Bioinformatics, 24(11):1397-98, 2008.

    """
    __name = "netmhc"
    __supported_length = frozenset([8, 9, 10, 11])
    __version = "3.4"
    __allele_import_name = f"{__name}_{__version}".replace('.', '_')
    __alleles = getattr(__import__("epytope.Data.supportedAlleles.external." + __allele_import_name,
                                   fromlist=[__allele_import_name])
                        , __allele_import_name)
    __command = "netMHC -p {peptides} -a {alleles} -x {out} {options}"

    @property
    def version(self):
        """The version of the predictor"""
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
            return "H-2-%s%s%s" % (allele.locus, allele.supertype, allele.subtype)
        else:
            return "HLA-%s%s:%s" % (allele.locus, allele.supertype, allele.subtype)

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

    @property
    def supportedAlleles(self):
        """
        A list of valid allele models
        """
        return self.__alleles

    @property
    def name(self):
        """The name of the predictor"""
        return self.__name

    @property
    def command(self):
        """
        Defines the commandline call for external tool
        """
        return self.__command

    @property
    def supportedLength(self):
        """
        A list of supported :class:`~epytope.Core.Peptide.Peptide` lengths
        """
        return self.__supported_length

    def parse_external_result(self, file):
        """
        Parses external results and returns the result containing the predictors string representation
        of alleles and peptides.

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results 
        :rtype: dict
        """
        result = defaultdict(defaultdict)
        f = csv.reader(open(file, "r"), delimiter='\t')
        next(f)
        next(f)
        alleles = [x.split()[0] for x in f.next()[3:]]
        for l in f:
            if not l:
                continue
            pep_seq = l[PeptideIndex.NETMHC_3_4]
            for ic_50, a in zip(l[ScoreIndex.NETMHC_3_0:], alleles):
                sc = 1.0 - math.log(float(ic_50), 50000)
                result[a][pep_seq] = sc if sc > 0.0 else 0.0
        return dict(result)

    def get_external_version(self, path=None):
        """
        Returns the external version of the tool by executing
        >{command} --version

        might be dependent on the method and has to be overwritten
        therefore it is declared abstract to enforce the user to
        overwrite the method. The function in the base class can be called
        with super()

        :param str path: Optional specification of executable path if deviant from :attr:`self.__command`
        :return: The external version of the tool or None if tool does not support versioning
        :rtype: dict
        """
        return super(NetMHC_3_4, self).get_external_version()

    def prepare_input(self, input, file):
        """
        Prepares input for external tools
        and writes them to file in the specific format

        NO return value!

        :param: list(str) input: The : sequences to write into _file
        :param File file: File-handler to input file for external tool
        """
        file.write("\n".join(input))


class NetMHC_3_0(NetMHC_3_4):
    """
    Implements the NetMHC binding (for netMHC3.0)::


    .. note::

        NetMHC-3.0: accurate web accessible predictions of human, mouse and monkey MHC class I affinities for peptides
        of length 8-11. Lundegaard C, Lamberth K, Harndahl M, Buus S, Lund O, Nielsen M.
        Nucleic Acids Res. 1;36(Web Server issue):W509-12. 2008

        Accurate approximation method for prediction of class I MHC affinities for peptides of length 8, 10 and 11
        using prediction tools trained on 9mers. Lundegaard C, Lund O, Nielsen M. Bioinformatics, 24(11):1397-98, 2008.
    """
    __name = "netmhc"
    __version = "3.0"
    __supported_length = frozenset([8, 9, 10, 11])
    __allele_import_name = f"{__name}_{__version}".replace('.', '_')
    __alleles = getattr(__import__("epytope.Data.supportedAlleles.external." + __allele_import_name,
                                   fromlist=[__allele_import_name])
                        , __allele_import_name)
    __command = "netMHC-3.0 -p {peptides} -a {alleles} -x {out} -l {length} {options}"

    @property
    def version(self):
        """The version of the predictor"""
        return self.__version

    @property
    def name(self):
        """The name of the predictor"""
        return self.__name

    @property
    def command(self):
        """
        Defines the commandline call for external tool
        """
        return self.__command

    @property
    def supportedAlleles(self):
        """
        A list of valid :class:`~epytope.Core.Allele.Allele` models
        """
        return self.__alleles

    @property
    def supportedLength(self):
        """
        A list of supported :class:`~epytope.Core.Peptide.Peptide` lengths
        """
        return self.__supported_length

    def _represent(self, allele):
        """
        Internal function transforming an allele object into its representative string
        :param allele: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is
                        needed
        :type alleles: :class:`~epytope.Core.Allele.Allele`
        :return: str
        """
        if isinstance(allele, MouseAllele):
            return "H-2-%s%s%s" % (allele.locus, allele.supertype, allele.subtype)
        else:
            return "HLA-%s%s%s" % (allele.locus, allele.supertype, allele.subtype)

    def parse_external_result(self, file):
        """
        Parses external results and returns the result containing the predictors string representation
        of alleles and peptides.

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results
        :rtype: dict
        """
        result = defaultdict(dict)
        with open(file, 'r') as f:
            next(f, None)  # skip first line with logging stuff
            next(f, None)  # skip first line with nothing
            csvr = csv.reader(f, delimiter='\t')
            alleles = [x.split()[0] for x in csvr.next()[3:]]
            for l in csvr:
                if not l:
                    continue
                pep_seq = l[PeptideIndex.NETMHC_3_0]
                for ic_50, a in zip(l[ScoreIndex.NETMHC_3_0:], alleles):
                    sc = 1.0 - math.log(float(ic_50), 50000)
                    result[a][pep_seq] = sc if sc > 0.0 else 0.0
        if 'Average' in result:
            result.pop('Average')
        return dict(result)


class NetMHC_4_0(NetMHC_3_4):
    """
    Implements the NetMHC 4.0 binding

    .. note::
        Andreatta M, Nielsen M. Gapped sequence alignment using artificial neural networks:
        application to the MHC class I system. Bioinformatics (2016) Feb 15;32(4):511-7
    """
    __name = 'netmhc'
    __version = "4.0"
    __allele_import_name = f"{__name}_{__version}".replace('.', '_')
    __alleles = getattr(__import__("epytope.Data.supportedAlleles.external." + __allele_import_name,
                                   fromlist=[__allele_import_name])
                        , __allele_import_name)
    __command = "netMHC -p {peptides} -a {alleles} -xls -xlsfile {out} {options}"

    @property
    def version(self):
        """The version of the predictor"""
        return self.__version

    @property
    def command(self):
        """
        Defines the commandline call for external tool
        """
        return self.__command

    def _represent(self, allele):
        """
        Internal function transforming an allele object into its representative string
        :param allele: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is
                        needed
        :type alleles: :class:`~epytope.Core.Allele.Allele`
        :return: str
        """
        if isinstance(allele, MouseAllele):
            return "H-2-%s%s%s" % (allele.locus, allele.supertype, allele.subtype)
        else:
            return "HLA-%s%s%s" % (allele.locus, allele.supertype, allele.subtype)

    def parse_external_result(self, file):
        """
        Parses external results and returns the result containing the predictors string representation
        of alleles and peptides.

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results
        :rtype: dict
        """
        scores = defaultdict(defaultdict)
        ranks = defaultdict(defaultdict)
        f = csv.reader(open(file, "r"), delimiter='\t')
        alleles = [x.split()[0] for x in [x for x in next(f) if x.strip() != ""]]
        next(f)
        for l in f:
            if not l:
                continue
            pep_seq = l[PeptideIndex.NETMHC_4_0]
            for i, a in enumerate(alleles):
                ic_50 = l[(i+1) * Offset.NETMHC_4_0]
                sc = 1.0 - math.log(float(ic_50), 50000)
                rank = l[(i+1)* Offset.NETMHC_4_0 + 1]
                scores[a][pep_seq] = sc if sc > 0.0 else 0.0
                ranks[a][pep_seq] = float(rank)
        
        result = {allele: {metric:(list(scores.values())[j] if metric == "Score" else list(ranks.values())[j]) for metric in ["Score", "Rank"]} for j, allele in enumerate(alleles)}
        return result

    def get_external_version(self, path=None):
        """
        Returns the external version of the tool by executing
        >{command} --version

        might be dependent on the method and has to be overwritten
        therefore it is declared abstract to enforce the user to
        overwrite the method. The function in the base class can be called
        with super()

        :param str path: Optional specification of executable path if deviant from :attr:`self.__command`
        :return: The external version of the tool or None if tool does not support versioning
        :rtype: str
        """
        # can not be determined netmhcpan does not support --version or similar
        return None


class NetMHCpan_2_4(AExternalEpitopePrediction):
    """
    Implements the NetMHC binding (in current form for netMHCpan 2.4).
    Supported  MHC alleles currently only restricted to HLA alleles.

    .. note::

        Nielsen, Morten, et al. "NetMHCpan, a method for quantitative predictions of peptide binding to any HLA-A and-B
        locus protein of known sequence." PloS one 2.8 (2007): e796.
    """
    __name = "netmhcpan"
    __version = "2.4"
    __supported_length = frozenset([8, 9, 10, 11])
    __allele_import_name = f"{__name}_{__version}".replace('.', '_')
    __alleles = getattr(__import__("epytope.Data.supportedAlleles.external." + __allele_import_name,
                                   fromlist=[__allele_import_name])
                        , __allele_import_name)
    __command = "netMHCpan-2.4 -p {peptides} -a {alleles} {options} -ic50 -xls -xlsfile {out}"

    @property
    def version(self):
        """The version of the predictor"""
        return self.__version

    @property
    def supportedAlleles(self):
        """
        A list of valid :class:`~epytope.Core.Allele.Allele` models
        """
        return self.__alleles

    @property
    def name(self):
        """The name of the predictor"""
        return self.__name

    @property
    def command(self):
        """
        Defines the commandline call for external tool
        """
        return self.__command

    @property
    def supportedLength(self):
        """
        A list of supported :class:`~epytope.Core.Peptide.Peptide` lengths
        """
        return self.__supported_length

    def _represent(self, allele):
        """
        Internal function transforming an allele object into its representative string
        :param allele: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is
                        needed
        :type alleles: :class:`~epytope.Core.Allele.Allele`
        :return: str
        """
        if isinstance(allele, MouseAllele):
            return "H-2-%s%s%s" % (allele.locus, allele.supertype, allele.subtype)
        else:
            return "HLA-%s%s:%s" % (allele.locus, allele.supertype, allele.subtype)

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

    def parse_external_result(self, file):
        """
        Parses external results and returns the result containing the predictors string representation
        of alleles and peptides.

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results
        :rtype: dict
        """
        f = csv.reader(open(file, "r"), delimiter = '\t')
        scores = defaultdict(defaultdict)
        alleles = [x for x in next(f) if "HLA" in x]
        # Rank is not supported in command line tool of NetMHCpan 2.4
        for row in f:
            pep_seq = row[PeptideIndex.NETMHCPAN_2_4]
            for i, a in enumerate(alleles):
                scores[a][pep_seq] = float(row[ScoreIndex.NETMHCPAN_2_4 + i])
        # Create dictionary with hierarchy: {'Allele1': {'Score': {'Pep1': Score1, 'Pep2': Score2,..}, 'Allele2':...}
        result = {allele: {"Score":(list(scores.values())[j])} for j, allele in enumerate(alleles)}

        return result

    def get_external_version(self, path=None):
        """
        Returns the external version of the tool by executing
        >{command} --version

        might be dependent on the method and has to be overwritten
        therefore it is declared abstract to enforce the user to
        overwrite the method. The function in the base class can be called
        with super()

        :param str path: Optional specification of executable path if deviant from :attr:`self.__command`
        :return: The external version of the tool or None if tool does not support versioning
        :rtype: str
        """
        # can not be determined netmhcpan does not support --version or similar
        return None

    def prepare_input(self, input, file):
        """
        Prepares input for external tools and writes them to file in the specific format

        NO return value!

        :param: list(str) input: The :class:`~epytope.Core.Peptide.Peptide` sequences to write into file
        :param File file: File-handler to input file for external tool
        """
        file.write("\n".join(input))


class NetMHCpan_2_8(AExternalEpitopePrediction):
    """
    Implements the NetMHC binding (in current form for netMHCpan 2.8).
    Supported  MHC alleles currently only restricted to HLA alleles.

    .. note::

        Nielsen, Morten, et al. "NetMHCpan, a method for quantitative predictions of peptide binding to any HLA-A and-B
        locus protein of known sequence." PloS one 2.8 (2007): e796.
    """
    __name = "netmhcpan"
    __version = "2.8"
    __supported_length = frozenset([8, 9, 10, 11, 12, 13, 14])
    __allele_import_name = f"{__name}_{__version}".replace('.', '_')
    __alleles = getattr(__import__("epytope.Data.supportedAlleles.external." + __allele_import_name,
                                   fromlist=[__allele_import_name])
                        , __allele_import_name)
    __command = "netMHCpan -p {peptides} -a {alleles} {options} -ic50 -xls -xlsfile {out}"

    @property
    def version(self):
        """The version of the predictor"""
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
            return "H-2-%s%s%s" % (allele.locus, allele.supertype, allele.subtype)
        else:
            return "HLA-%s%s:%s" % (allele.locus, allele.supertype, allele.subtype)

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

    @property
    def supportedAlleles(self):
        """A list of valid :class:`~epytope.Core.Allele.Allele` models"""
        return self.__alleles

    @property
    def name(self):
        """The name of the predictor"""
        return self.__name

    @property
    def command(self):
        """
        Defines the commandline call for external tool
        """
        return self.__command

    @property
    def supportedLength(self):
        """
        A list of supported :class:`~epytope.Core.Peptide.Peptide` lengths
        """
        return self.__supported_length

    def get_external_version(self, path=None):
        """
        Returns the external version of the tool by executing
        >{command} --version

        might be dependent on the method and has to be overwritten
        therefore it is declared abstract to enforce the user to
        overwrite the method. The function in the base class can be called
        with super()

        :param str path: Optional specification of executable path if deviant from :attr:`self.__command`
        :return: The external version of the tool or None if tool does not support versioning
        :rtype: str
        """
        return None

    def prepare_input(self, input, file):
        """
        Prepares input for external tools
        and writes them to file in the specific format

        No return value!

        :param: list(str) input: The :class:`~epytope.Core.Peptide.Peptide` sequences to write into file
        :param File file: File-handler to input file for external tool
        """
        file.write("\n".join(input))

    def parse_external_result(self, file):
        """
        Parses external results and returns the result containing the predictors string representation
        of alleles and peptides.

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results
        :rtype: dict
        """
        f = csv.reader(open(file, "r"), delimiter = '\t')
        scores = defaultdict(defaultdict)
        ranks = defaultdict(defaultdict)
        alleles = [x for x in next(f) if x != ""]
        next(f)
        for row in f:
            pep_seq = row[PeptideIndex.NETMHCPAN_2_8]
            for i, a in enumerate(alleles):
                if row[ScoreIndex.NETMHCPAN_2_8 + i * Offset.NETMHCPAN_2_8] != "1-log50k":     # Avoid header column, only access raw and rank scores
                    scores[a][pep_seq] = float(row[ScoreIndex.NETMHCPAN_2_8 + i * Offset.NETMHCPAN_2_8])
                    ranks[a][pep_seq] = float(row[RankIndex.NETMHCPAN_2_8 + i * Offset.NETMHCPAN_2_8])
        # Create dictionary with hierarchy: {'Allele1': {'Score': {'Pep1': Score1, 'Pep2': Score2,..}, 'Rank': {'Pep1': RankScore1, 'Pep2': RankScore2,..}}, 'Allele2':...}
        result = {allele: {metric:(list(scores.values())[j] if metric == "Score" else list(ranks.values())[j]) for metric in ["Score", "Rank"]} for j, allele in enumerate(alleles)}
        return result


class NetMHCpan_3_0(NetMHCpan_2_8):
    """
        Implements the NetMHC binding version 3.0
        Supported  MHC alleles currently only restricted to HLA alleles.

    .. note::

        Nielsen, M., & Andreatta, M. (2016).
        NetMHCpan-3.0; improved prediction of binding to MHC class I molecules integrating information from multiple
        receptor and peptide length datasets. Genome Medicine, 8(1), 1.
    """
    __name = "netmhcpan"
    __version = "3.0"
    __allele_import_name = f"{__name}_{__version}".replace('.', '_')
    __alleles = getattr(__import__("epytope.Data.supportedAlleles.external." + __allele_import_name,
                                   fromlist=[__allele_import_name])
                        , __allele_import_name)
    __command = "netMHCpan -p {peptides} -a {alleles} {options} -xls -xlsfile {out}"

    @property
    def version(self):
        return self.__version

    @property
    def command(self):
        return self.__command

    def parse_external_result(self, file):
        """
        Parses external results and returns the result

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results
        :rtype: dict
        """
        f = csv.reader(open(file, "r"), delimiter = '\t')
        scores = defaultdict(defaultdict)
        ranks = defaultdict(defaultdict)
        alleles = [x for x in next(f) if x != ""]
        for row in f:
            pep_seq = row[PeptideIndex.NETMHCPAN_3_0]
            for i, a in enumerate(alleles):
                if row[ScoreIndex.NETMHCPAN_3_0 + i * Offset.NETMHCPAN_3_0] != "1-log50k":     # Avoid header column, only access raw and rank scores
                    scores[a][pep_seq] = float(row[ScoreIndex.NETMHCPAN_3_0 + i * Offset.NETMHCPAN_3_0])
                    ranks[a][pep_seq] = float(row[RankIndex.NETMHCPAN_3_0 + i * Offset.NETMHCPAN_3_0])
                    
        # Create dictionary with hierarchy: {'Allele1': {'Score': {'Pep1': Score1, 'Pep2': Score2,..}, 'Rank': {'Pep1': RankScore1, 'Pep2': RankScore2,..}}, 'Allele2':...}
        result = {allele: {metric:(list(scores.values())[j] if metric == "Score" else list(ranks.values())[j]) for metric in ["Score", "Rank"]} for j, allele in enumerate(alleles)}
        return result


class NetMHCpan_4_0(NetMHCpan_3_0):
    """
        Implements the NetMHC binding version 4.0
        Supported  MHC alleles currently only restricted to HLA alleles.
    """
    __name = "netmhcpan"
    __version = "4.0"
    __allele_import_name = f"{__name}_{__version}".replace('.', '_')
    __alleles = getattr(__import__("epytope.Data.supportedAlleles.external." + __allele_import_name,
                                   fromlist=[__allele_import_name])
                        , __allele_import_name)
    __command = "netMHCpan -p {peptides} -a {alleles} {options} -xls -xlsfile {out}"

    @property
    def version(self):
        return self.__version

    @property
    def command(self):
        return self.__command

    def parse_external_result(self, file):
        """
        Parses external results and returns the result containing the predictors string representation
        of alleles and peptides.

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results
        :rtype: dict
        """
        f = csv.reader(open(file, "r"), delimiter = '\t')
        scores = defaultdict(defaultdict)
        ranks = defaultdict(defaultdict)
        alleles = [x for x in next(f) if x != ""]
        for row in f:
            pep_seq = row[PeptideIndex.NETMHCPAN_4_0]
            for i, a in enumerate(alleles):
                if row[ScoreIndex.NETMHCPAN_4_0 + i * Offset.NETMHCPAN_4_0] != "1-log50k":     # Avoid header column, only access raw and rank scores
                    scores[a][pep_seq] = float(row[ScoreIndex.NETMHCPAN_4_0 + i * Offset.NETMHCPAN_4_0])
                    ranks[a][pep_seq] = float(row[RankIndex.NETMHCPAN_4_0 + i * Offset.NETMHCPAN_4_0])
        # Create dictionary with hierarchy: {'Allele1': {'Score': {'Pep1': Score1, 'Pep2': Score2,..}, 'Rank': {'Pep1': RankScore1, 'Pep2': RankScore2,..}}, 'Allele2':...}
        result = {allele: {metric:(list(scores.values())[j] if metric == "Score" else list(ranks.values())[j]) for metric in ["Score", "Rank"]} for j, allele in enumerate(alleles)}
        return result



class NetMHCpan_4_1(NetMHCpan_4_0):
    """
        Implements the NetMHC binding version 4.1
        Supported  MHC alleles currently only restricted to HLA alleles.
    """
    __name = "netmhcpan"
    __version = "4.1"
    __allele_import_name = f"{__name}_{__version}".replace('.', '_')
    __alleles = getattr(__import__("epytope.Data.supportedAlleles.external." + __allele_import_name,
                                   fromlist=[__allele_import_name])
                        , __allele_import_name)
    __command = "netMHCpan -p {peptides} -a {alleles} {options} -xls -xlsfile {out}"
    @property
    def version(self):
        return self.__version

    @property
    def command(self):
        return self.__command

    def parse_external_result(self, file):
        """
        Parses external results and returns the result containing the predictors string representation
        of alleles and peptides.

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results
        :rtype: dict
        """
        f = csv.reader(open(file, "r"), delimiter = '\t')
        scores = defaultdict(defaultdict)
        ranks = defaultdict(defaultdict)
        alleles = [x for x in next(f) if x != ""]
        next(f, None) # Avoid header column, only access raw and rank scores
        for row in f:
            pep_seq = row[PeptideIndex.NETMHCPAN_4_1]
            for i, a in enumerate(alleles):
                scores[a][pep_seq] = float(row[ScoreIndex.NETMHCPAN_4_1 + i * Offset.NETMHCPAN_4_1])
                ranks[a][pep_seq] = float(row[RankIndex.NETMHCPAN_4_1 + i * Offset.NETMHCPAN_4_1])
        # Create dictionary with hierarchy: {'Allele1': {'Score': {'Pep1': Score1, 'Pep2': Score2,..}, 'Rank': {'Pep1': RankScore1, 'Pep2': RankScore2,..}}, 'Allele2':...}
        result = {allele: {metric:(list(scores.values())[j] if metric == "Score" else list(ranks.values())[j]) for metric in ["Score", "Rank"]} for j, allele in enumerate(alleles)}
        return result


class NetMHCstabpan_1_0(AExternalEpitopePrediction):
    """
    Implements a wrapper to NetMHCstabpan 1.0

    .. note:

    Pan-specific prediction of peptide-MHC-I complex stability; a correlate of T cell immunogenicity
    M Rasmussen, E Fenoy, M Nielsen, Buus S, Accepted JI June, 2016
    """
    __name = "netMHCstabpan"
    __length = frozenset([8, 9, 10, 11])
    __version = "1.0"
    __allele_import_name = f"{__name}_{__version}".replace('.', '_')
    __alleles = getattr(__import__("epytope.Data.supportedAlleles.external." + __allele_import_name,
                                   fromlist=[__allele_import_name])
                        , __allele_import_name)
    __command = "netMHCstabpan -p {peptides} -a {alleles} {options} -xls -xlsfile {out}"

    @property
    def command(self):
        return self.__command

    @property
    def name(self):
        return self.__name

    @property
    def version(self):
        return self.__version

    @property
    def supportedAlleles(self):
        return self.__alleles

    @property
    def supportedLength(self):
        return self.__length

    def convert_alleles(self, alleles):
        """
        Converts :class:`~epytope.Core.Allele.Allele` into the internal allele representation of the predictor
        and returns a string representation

        :param  alleles: The :class:`~epytope.Core.Allele.Allele` for which the
                         internal predictor representation is needed
        :type alleles: list(:class:`~epytope.Core.Allele.Allele`)
        :return: Returns a string representation of the input :class:`~epytope.Core.Allele.Allele`
        :rtype: list(str)
        """
        return ["HLA-%s%s:%s" % (a.locus, a.supertype, a.subtype) for a in alleles]

    def parse_external_result(self, file):
        """
        Parses external results and returns the result containing the predictors string representation
        of alleles and peptides.

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results
        :rtype: dict
        """
        result = defaultdict(dict)
        with open(file, "r") as f:
            f = csv.reader(f, delimiter='\t')
            alleles = [x for x in next(f) if x != ""]
            ranks = defaultdict(defaultdict)
            rank_pos = 5
            offset = 3
            header = next(f)
            if "Aff(nM)" in header:  # With option command line option '-ia', which includes prediction score in output file
                scores = defaultdict(defaultdict)
                for row in f:
                    pep_seq = row[PeptideIndex.NETMHCSTABPAN_1_0]
                    for i, a in enumerate(alleles):
                        scores[a][pep_seq] = float(row[ScoreIndex.NETMHCSTABPAN_1_0 + i * Offset.NETMHCSTABPAN_1_0_W_SCORE])
                        ranks[a][pep_seq] = float(row[RankIndex.NETMHCSTABPAN_1_0 + i * Offset.NETMHCSTABPAN_1_0_W_SCORE])
                        # Create dictionary with hierarchy: {'Allele1': {'Score': {'Pep1': Score1, 'Pep2': Score2,..}, 'Rank': {'Pep1': RankScore1, 'Pep2': RankScore2,..}}, 'Allele2':...}
                result = {allele: {metric:(list(scores.values())[j] if metric == "Score" else list(ranks.values())[j]) for metric in ["Score", "Rank"]} for j, allele in enumerate(alleles)}
            else:
                for row in f:
                    pep_seq = row[PeptideIndex.NETMHCSTABPAN_1_0]
                    for i, a in enumerate(alleles):
                        ranks[a][pep_seq] = float(row[RankIndex.NETMHCSTABPAN_1_0 + i * Offset.NETMHCSTABPAN_1_0_WO_SCORE])
                        # Create dictionary with hierarchy: {'Allele1':{'Rank': {'Pep1': RankScore1, 'Pep2': RankScore2,..}}, 'Allele2':...}
                result = {allele: {"Rank":list(ranks.values())[j]} for j, allele in enumerate(alleles)}

        return result

    def get_external_version(self, path=None):
        """
        Returns the external version of the tool by executing
        >{command} --version

        might be dependent on the method and has to be overwritten
        therefore it is declared abstract to enforce the user to
        overwrite the method. The function in the base class can be called
        with super()

        :param str path: Optional specification of executable path if deviant from :attr:`self.__command`
        :return: The external version of the tool or None if tool does not support versioning
        :rtype: str
        """
        # can not be determined netmhcpan does not support --version or similar
        return None

    def prepare_input(self, input, file):
        """
        Prepares input for external tools and writes them to file in the specific format

        NO return value!

        :param: list(str) input: The :class:`~epytope.Core.Peptide.Peptide` sequences to write into file
        :param File file: File-handler to input file for external tool
        """
        file.write("\n".join(input))



class NetMHCII_2_2(AExternalEpitopePrediction):
    """
    Implements a wrapper for NetMHCII

    .. note::

        Nielsen, M., & Lund, O. (2009). NN-align. An artificial neural network-based alignment algorithm for MHC class
        II peptide binding prediction. BMC Bioinformatics, 10(1), 296.

        Nielsen, M., Lundegaard, C., & Lund, O. (2007). Prediction of MHC class II binding affinity using SMM-align,
        a novel stabilization matrix alignment method. BMC Bioinformatics, 8(1), 238.
    """
    __name = "netmhcII"
    __version = "2.2"
    __supported_length = frozenset([15])
    __allele_import_name = f"{__name}_{__version}".replace('.', '_')
    __alleles = getattr(__import__("epytope.Data.supportedAlleles.external." + __allele_import_name,
                                   fromlist=[__allele_import_name])
                        , __allele_import_name)
    __command = 'netMHCII {peptides} -a {alleles} {options} | grep -v "#" > {out}'

    @property
    def version(self):
        """The version of the predictor"""
        return self.__version

    @property
    def command(self):
        """
        Defines the commandline call for external tool
        """
        return self.__command

    @property
    def supportedLength(self):
        """
        A list of supported :class:`~epytope.Core.Peptide.Peptide` lengths
        """
        return self.__supported_length

    @property
    def supportedAlleles(self):
        """A list of valid :class:`~epytope.Core.Allele.Allele` models"""
        return self.__alleles

    @property
    def name(self):
        """The name of the predictor"""
        return self.__name

    def _represent(self, allele):
        """
        Internal function transforming an allele object into its representative string
        :param allele: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is
                        needed
        :type alleles: :class:`~epytope.Core.Allele.Allele`
        :return: str
        """
        if isinstance(allele, MouseAllele):
            return "H-2-%s%s%s" % (allele.locus, allele.supertype, allele.subtype)
        else:
            return "HLA-%s%s%s" % (allele.locus, allele.supertype, allele.subtype)

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

    def parse_external_result(self, file):
        """
        Parses external results and returns the result containing the predictors string representation
        of alleles and peptides.

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results
        :rtype: dict
        """
        f = csv.reader(open(file, "r"), delimiter='\t')
        scores = defaultdict(defaultdict)
        for r in f:
            if not r:
                continue
            
            row = r[0].split()
            if not len(row):
                continue
            
            if "HLA" not in row[HLAIndex.NETMHCII_2_2]:
                continue
            allele = row[HLAIndex.NETMHCII_2_2]
            pep = row[PeptideIndex.NETMHCII_2_2]
            scores[allele][pep] = float(row[ScoreIndex.NETMHCII_2_2])

        result = {allele: {"Score":list(scores.values())[j]} for j, allele in enumerate(scores.keys())}

        return result

    def get_external_version(self, path=None):
        """
        Returns the external version of the tool by executing
        >{command} --version

        might be dependent on the method and has to be overwritten
        therefore it is declared abstract to enforce the user to
        overwrite the method. The function in the base class can be called
        with super()

        :param str path: Optional specification of executable path if deviant from :attr:`self.__command`
        :return: The external version of the tool or None if tool does not support versioning
        :rtype: str
        """
        return None

    def prepare_input(self, input, file):
        """
        Prepares input for external tools
        and writes them to _file in the specific format

        No return value!

        :param: list(str) input: The :class:`~epytope.Core.Peptide.Peptide` sequences to write into file
        :param File file: File-handler to input file for external tool
        """
        file.write("\n".join(">pepe_%i\n%s" % (i, p) for i, p in enumerate(input)))


class NetMHCII_2_3(NetMHCII_2_2):
    """
    Implements a wrapper for NetMHCII 2.3

    .. note::

        Jensen KK, Andreatta M, Marcatili P, Buus S, Greenbaum JA, Yan Z, Sette A, Peters B, and Nielsen M. (2018)
        Improved methods for predicting peptide binding affinity to MHC class II molecules. 
    """
    __name = "netmhcII"
    __version = "2.3"
    __supported_length = frozenset([15])
    __allele_import_name = f"{__name}_{__version}".replace('.', '_')
    __alleles = getattr(__import__("epytope.Data.supportedAlleles.external." + __allele_import_name,
                                   fromlist=[__allele_import_name])
                        , __allele_import_name)
    __command = 'netMHCII {peptides} -a {alleles} {options} | grep -v "#" > {out}'


    @property
    def version(self):
        """The version of the predictor"""
        return self.__version

    @property
    def command(self):
        """
        Defines the commandline call for external tool
        """
        return self.__command

    @property
    def supportedLength(self):
        """
        A list of supported :class:`~epytope.Core.Peptide.Peptide` lengths
        """
        return self.__supported_length

    @property
    def supportedAlleles(self):
        """A list of valid :class:`~epytope.Core.Allele.Allele` models"""
        return self.__alleles

    @property
    def name(self):
        """The name of the predictor"""
        return self.__name

    def _represent(self, allele):
        """
        Internal function transforming an allele object into its representative string
        :param allele: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is
                        needed
        :type alleles: :class:`~epytope.Core.Allele.Allele`
        :return: str
        """
        if isinstance(allele, MouseAllele):
            return "H-2-%s%s%s" % (allele.locus, allele.supertype, allele.subtype)
        elif isinstance(allele, CombinedAllele):
            return '%s-%s%s%s-%s%s%s' % (allele.organism, allele.alpha_locus, allele.alpha_supertype, allele.alpha_subtype,
                                          allele.beta_locus, allele.beta_supertype, allele.beta_subtype)
        else:
            return "%s_%s%s" % (allele.locus, allele.supertype, allele.subtype)


    def parse_external_result(self, file):
        """
        Parses external results and returns the result containing the predictors string representation
        of alleles and peptides.

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results
        :rtype: dict
        """
        f = csv.reader(open(file, "r"), delimiter='\t')
        scores = defaultdict(defaultdict)
        ranks = defaultdict(defaultdict)

        for r in f:
            if not r:
                continue
            
            row = r[0].split()
            if not len(row):
                continue
            
            if all(prefix not in row[HLAIndex.NETMHCII_2_3] for prefix in ['HLA-', 'H-2', 'D']):
                continue

            allele = row[HLAIndex.NETMHCII_2_3]
            
            pep = row[PeptideIndex.NETMHCII_2_3]
            scores[allele][pep] = float(row[ScoreIndex.NETMHCII_2_3])
            ranks[allele][pep] = float(row[ScoreIndex.NETMHCII_2_3])
            

        result = {allele: {metric:(list(scores.values())[j] if metric == "Score" else list(ranks.values())[j]) for metric in ["Score", "Rank"]} for j, allele in enumerate(scores.keys())}

        return result


class NetMHCIIpan_3_0(AExternalEpitopePrediction):
    """
    Implements a wrapper for NetMHCIIpan.

    .. note::

        Andreatta, M., Karosiene, E., Rasmussen, M., Stryhn, A., Buus, S., & Nielsen, M. (2015).
        Accurate pan-specific prediction of peptide-MHC class II binding affinity with improved binding
        core identification. Immunogenetics, 1-10.
    """


    __name = "netmhcIIpan"
    __version = "3.0"
    __supported_length = frozenset([9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    __allele_import_name = f"{__name}_{__version}".replace('.', '_')
    __alleles = getattr(__import__("epytope.Data.supportedAlleles.external." + __allele_import_name,
                                   fromlist=[__allele_import_name])
                        , __allele_import_name)
    __command = "netMHCIIpan -f {peptides} -inptype 1 -a {alleles} {options} -xls -xlsfile {out}"

    @property
    def version(self):
        """The version of the predictor"""
        return self.__version

    @property
    def command(self):
        """
        Defines the commandline call for external tool
        """
        return self.__command

    @property
    def supportedLength(self):
        """
        A list of supported :class:`~epytope.Core.Peptide.Peptide` lengths
        """
        return self.__supported_length

    @property
    def supportedAlleles(self):
        """A list of valid :class:`~epytope.Core.Allele.Allele` models"""
        return self.__alleles

    @property
    def name(self):
        """The name of the predictor"""
        return self.__name

    def _represent(self, allele):
        """
        Internal function transforming an allele object into its representative string
        :param allele: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is
                        needed
        :type alleles: :class:`~epytope.Core.Allele.Allele`
        :return: str
        """
        if isinstance(allele, MouseAllele):
            return "H-2-%s%s%s" % (allele.locus, allele.supertype.upper(), allele.subtype)
        elif isinstance(allele, CombinedAllele):
            return "HLA-%s%s%s-%s%s%s" % (allele.alpha_locus, allele.alpha_supertype, allele.alpha_subtype,
                                          allele.beta_locus, allele.beta_supertype, allele.beta_subtype)
        else:
            return "%s_%s%s" % (allele.locus, allele.supertype, allele.subtype)

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

    def parse_external_result(self, file):
        """
        Parses external results and returns the result containing the predictors string representation
        of alleles and peptides.

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results
        :rtype: dict
        """
        def parse_allele_from_external_result(alleles_str_out):
            """
            Parses allele string from external result to allele string representation of input

            :param str allele_str_out: The allele string representation from the external result output
            :return: str allele_str_in: The allele string representation from the external result input
            :rtype: str
            """
            alleles_str_in = []
            for allele_str_out in alleles_str_out:
                if allele_str_out.startswith('HLA-'):
                    allele_str_in = allele_str_out.replace('*','').replace(':','')
                elif allele_str_out.startswith('D'):
                    allele_str_in = allele_str_out.replace('*','_').replace(':','')
                else:
                    allele_str_in = allele_str_out
                alleles_str_in.append(allele_str_in)

            return(alleles_str_in)

        f = csv.reader(open(file, "r"), delimiter='\t')
        scores = defaultdict(defaultdict)
        ranks = defaultdict(defaultdict)
        alleles = [x for x in set([x for x in next(f) if x != ""])]
        # Convert output representation of allele to input representation of allele, because they differ
        alleles = parse_allele_from_external_result(alleles)
        
        next(f)
        for row in f:
            pep_seq = row[PeptideIndex.NETMHCIIPAN_3_0]
            for i, a in enumerate(alleles):
                scores[a][pep_seq] = float(row[ScoreIndex.NETMHCIIPAN_3_0 + i * Offset.NETMHCIIPAN_3_0])
                ranks[a][pep_seq] = float(row[RankIndex.NETMHCIIPAN_3_0 + i * Offset.NETMHCIIPAN_3_0])
                # Create dictionary with hierarchy: {'Allele1': {'Score': {'Pep1': Score1, 'Pep2': Score2,..}, 'Rank': {'Pep1': RankScore1, 'Pep2': RankScore2,..}}, 'Allele2':...}
        result = {allele: {metric:(list(scores.values())[j] if metric == "Score" else list(ranks.values())[j]) for metric in ["Score", "Rank"]} for j, allele in enumerate(alleles)}

        return result

    def get_external_version(self, path=None):
        """
        Returns the external version of the tool by executing
        >{command} --version

        might be dependent on the method and has to be overwritten
        therefore it is declared abstract to enforce the user to
        overwrite the method. The function in the base class can be called
        with super()

        :param str path: Optional specification of executable path if deviant from :attr:`self.__command`
        :return: The external version of the tool or None if tool does not support versioning
        :rtype: str
        """
        return None

    def prepare_input(self, input, file):
        """
        Prepares input for external tools
        and writes them to _file in the specific format

        No return value!

        :param: list(str) input: The :class:`~epytope.Core.Peptide.Peptide` sequences to write into file
        :param File file: File-handler to input file for external tool
        """
        file.write("\n".join(input))


class NetMHCIIpan_3_1(NetMHCIIpan_3_0):
    """
    Implementation of NetMHCIIpan 3.1 adapter.

    .. note::

        Andreatta, M., Karosiene, E., Rasmussen, M., Stryhn, A., Buus, S., & Nielsen, M. (2015). Accurate pan-specific
        prediction of peptide-MHC class II binding affinity with improved binding core identification.
        Immunogenetics, 1-10.
    """
    __name = "netmhcIIpan"
    __version = "3.1"
    __supported_length = frozenset([9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    __allele_import_name = f"{__name}_{__version}".replace('.', '_')
    __alleles = getattr(__import__("epytope.Data.supportedAlleles.external." + __allele_import_name,
                                   fromlist=[__allele_import_name])
                        , __allele_import_name)
    __command = "netMHCIIpan -f {peptides} -inptype 1 -a {alleles} {options} -xls -xlsfile {out}"

    @property
    def version(self):
        """The version of the predictor"""
        return self.__version

    @property
    def command(self):
        """
        Defines the commandline call for external tool
        """
        return self.__command

    @property
    def supportedLength(self):
        """
        A list of supported :class:`~epytope.Core.Peptide.Peptide` lengths
        """
        return self.__supported_length

    @property
    def supportedAlleles(self):
        """A list of valid :class:`~epytope.Core.Allele.Allele` models"""
        return self.__alleles

    @property
    def name(self):
        """The name of the predictor"""
        return self.__name

    def parse_external_result(self, file):
        """
        Parses external results and returns the result containing the predictors string representation
        of alleles and peptides.

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results
        :rtype: dict
        """
        f = csv.reader(open(file, "r"), delimiter='\t')
        scores = defaultdict(defaultdict)
        ranks = defaultdict(defaultdict)
        alleles = [x for x in set([x for x in next(f) if x != ""])]
 
        next(f)
        for row in f:
            pep_seq = row[PeptideIndex.NETMHCIIPAN_3_1]
            for i, a in enumerate(alleles):
                scores[a][pep_seq] = float(row[ScoreIndex.NETMHCIIPAN_3_1 + i * Offset.NETMHCIIPAN_3_1])
                ranks[a][pep_seq] = float(row[RankIndex.NETMHCIIPAN_3_1 + i * Offset.NETMHCIIPAN_3_1])
                # Create dictionary with hierarchy: {'Allele1': {'Score': {'Pep1': Score1, 'Pep2': Score2,..}, 'Rank': {'Pep1': RankScore1, 'Pep2': RankScore2,..}}, 'Allele2':...}
        result = {allele: {metric:(list(scores.values())[j] if metric == "Score" else list(ranks.values())[j]) for metric in ["Score", "Rank"]} for j, allele in enumerate(alleles)}

        return result

class NetMHCIIpan_4_0(NetMHCIIpan_3_1):
    """
    Implementation of NetMHCIIpan 4.0 adapter.

    .. note::

        Reynisson B, Barra C, Kaabinejadian S, Hildebrand WH, Peters B, Nielsen M (2020). Improved prediction of MHC II antigen presentation
        through integration and motif deconvolution of mass spectrometry MHC eluted ligand data.
        Immunogenetics, 1-10.
    """
    __name = "netmhcIIpan"
    __version = "4.0"
    __supported_length = frozenset(list(range(9,57)))
    __allele_import_name = f"{__name}_{__version}".replace('.', '_')
    __alleles = getattr(__import__("epytope.Data.supportedAlleles.external." + __allele_import_name,
                                   fromlist=[__allele_import_name])
                        , __allele_import_name)
    __command = "netMHCIIpan -f {peptides} -inptype 1 -a {alleles} {options} -xls -xlsfile {out}"

    @property
    def version(self):
        """The version of the predictor"""
        return self.__version

    @property
    def command(self):
        """
        Defines the commandline call for external tool
        """
        return self.__command

    @property
    def supportedLength(self):
        """
        A list of supported :class:`~epytope.Core.Peptide.Peptide` lengths
        """
        return self.__supported_length

    @property
    def supportedAlleles(self):
        """A list of valid :class:`~epytope.Core.Allele.Allele` models"""
        return self.__alleles

    @property
    def name(self):
        """The name of the predictor"""
        return self.__name

    def parse_external_result(self, file):
        """
        Parses external results and returns the result containing the predictors string representation
        of alleles and peptides.

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results
        :rtype: dict
        """
        f = csv.reader(open(file, "r"), delimiter='\t')
        scores = defaultdict(defaultdict)
        ranks = defaultdict(defaultdict)
        alleles = [x for x in set([x for x in next(f) if x != ""])]
        next(f)
        for row in f:
            pep_seq = row[PeptideIndex.NETMHCIIPAN_4_0]
            for i, a in enumerate(alleles):
                scores[a][pep_seq] = float(row[ScoreIndex.NETMHCIIPAN_4_0 + i * Offset.NETMHCIIPAN_4_0])
                ranks[a][pep_seq] = float(row[RankIndex.NETMHCIIPAN_4_0 + i * Offset.NETMHCIIPAN_4_0])
                # Create dictionary with hierarchy: {'Allele1': {'Score': {'Pep1': Score1, 'Pep2': Score2,..}, 'Rank': {'Pep1': RankScore1, 'Pep2': RankScore2,..}}, 'Allele2':...}
        result = {allele: {metric:(list(scores.values())[j] if metric == "Score" else list(ranks.values())[j]) for metric in ["Score", "Rank"]} for j, allele in enumerate(alleles)}

        return result

class NetMHCIIpan_4_1(NetMHCIIpan_4_0):
    """
    Implementation of NetMHCIIpan 4.1 adapter.
    """

    __command = "netMHCIIpan -f {peptides} -inptype 1 -a {alleles} {options} -xls -xlsfile {out}"
    __version = "4.1"

    @property
    def version(self):
        """The version of the predictor"""
        return self.__version

    @property
    def command(self):
        """
        Defines the commandline call for external tool
        """
        return self.__command 

    def parse_external_result(self, file):
        """
        Parses external results and returns the result containing the predictors string representation
        of alleles and peptides.

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results
        :rtype: dict
        """
        f = csv.reader(open(file, "r"), delimiter='\t')
        scores = defaultdict(defaultdict)
        ranks = defaultdict(defaultdict)
        alleles = [x for x in set([x for x in next(f) if x != ""])]
        next(f)
        for row in f:
            pep_seq = row[PeptideIndex.NETMHCIIPAN_4_1]
            for i, a in enumerate(alleles):
                scores[a][pep_seq] = float(row[ScoreIndex.NETMHCIIPAN_4_1 + i * Offset.NETMHCIIPAN_4_1])
                ranks[a][pep_seq] = float(row[RankIndex.NETMHCIIPAN_4_1 + i * Offset.NETMHCIIPAN_4_1])
                # Create dictionary with hierarchy: {'Allele1': {'Score': {'Pep1': Score1, 'Pep2': Score2,..}, 'Rank': {'Pep1': RankScore1, 'Pep2': RankScore2,..}}, 'Allele2':...}
        result = {allele: {metric:(list(scores.values())[j] if metric == "Score" else list(ranks.values())[j]) for metric in ["Score", "Rank"]} for j, allele in enumerate(alleles)}

        return result

class NetMHCIIpan_4_2(NetMHCIIpan_4_1):
    """
    Implementation of NetMHCIIpan 4.2 adapter.
    """
    __name = "netmhcIIpan"
    __version = "4.2"
    __supported_length = frozenset(list(range(9,57)))
    __allele_import_name = f"{__name}_{__version}".replace('.', '_')
    __alleles = getattr(__import__("epytope.Data.supportedAlleles.external." + __allele_import_name,
                                   fromlist=[__allele_import_name])
                        , __allele_import_name)

    __command = "netMHCIIpan -f {peptides} -inptype 1 -a {alleles} {options} -xls -xlsfile {out}"


    @property
    def version(self):
        """The version of the predictor"""
        return self.__version

    @property
    def command(self):
        """
        Defines the commandline call for external tool
        """
        return self.__command


class NetMHCIIpan_4_3(NetMHCIIpan_4_2):
    """
    Implementation of NetMHCIIpan 4.3 adapter.
    """
    __name = "netmhcIIpan"
    __version = "4.3"
    __supported_length = frozenset(list(range(9,57)))

    __command = "netMHCIIpan -f {peptides} -inptype 1 -a {alleles} {options} -xls -xlsfile {out}"

    @property
    def version(self):
        """The version of the predictor"""
        return self.__version

    @property
    def command(self):
        """
        Defines the commandline call for external tool
        """
        return self.__command
    
    def parse_external_result(self, file):
        """
        Parses external results and returns the result containing the predictors string representation
        of alleles and peptides.

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results
        :rtype: dict
        """
        f = csv.reader(open(file, "r"), delimiter='\t')
        scores = defaultdict(defaultdict)
        ranks = defaultdict(defaultdict)
        alleles = [x for x in set([x for x in next(f) if x != ""])]
        next(f)
        for row in f:
            pep_seq = row[PeptideIndex.NETMHCIIPAN_4_3]
            for i, a in enumerate(alleles):
                scores[a][pep_seq] = float(row[ScoreIndex.NETMHCIIPAN_4_3 + i * Offset.NETMHCIIPAN_4_3])
                ranks[a][pep_seq] = float(row[RankIndex.NETMHCIIPAN_4_3 + i * Offset.NETMHCIIPAN_4_3])
                # Create dictionary with hierarchy: {'Allele1': {'Score': {'Pep1': Score1, 'Pep2': Score2,..}, 'Rank': {'Pep1': RankScore1, 'Pep2': RankScore2,..}}, 'Allele2':...}
        result = {allele: {metric:(list(scores.values())[j] if metric == "Score" else list(ranks.values())[j]) for metric in ["Score", "Rank"]} for j, allele in enumerate(alleles)}

        return result


class PickPocket_1_1(AExternalEpitopePrediction):
    """
    Implementation of PickPocket adapter.

    .. note::

        Zhang, H., Lund, O., & Nielsen, M. (2009). The PickPocket method for predicting binding specificities
        for receptors based on receptor pocket similarities: application to MHC-peptide binding.
        Bioinformatics, 25(10), 1293-1299.

    """
    __name = "pickpocket"
    __version = "1.1"
    __supported_length = frozenset([8, 9, 10, 11])
    __allele_import_name = f"{__name}_{__version}".replace('.', '_')
    __alleles = getattr(__import__("epytope.Data.supportedAlleles.external." + __allele_import_name,
                                   fromlist=[__allele_import_name])
                        , __allele_import_name)
    __command = 'PickPocket -p {peptides} -a {alleles} {options} | grep -v "#" > {out}'


    @property
    def version(self):
        """The version of the predictor"""
        return self.__version

    @property
    def command(self):
        """
        Defines the commandline call for external tool
        """
        return self.__command

    @property
    def supportedLength(self):
        """
        A list of supported :class:`~epytope.Core.Peptide.Peptide` lengths
        """
        return self.__supported_length

    @property
    def supportedAlleles(self):
        """
        A list of supported :class:`~epytope.Core.Allele.Allele`
        """
        return self.__alleles

    @property
    def name(self):
        """The name of the predictor"""
        return self.__name

    def _represent(self, allele):
        """
        Internal function transforming an allele object into its representative string
        :param allele: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is
                        needed
        :type alleles: :class:`~epytope.Core.Allele.Allele`
        :return: str
        """
        if isinstance(allele, MouseAllele):
            return "H-2-%s%s%s" % (allele.locus, allele.supertype, allele.subtype)
        else:
            return "HLA-%s%s:%s" % (allele.locus, allele.supertype, allele.subtype)

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

    def parse_external_result(self, file):
        """
        Parses external results and returns the result containing the predictors string representation
        of alleles and peptides.

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results
        :rtype: dict
        """
        scores = defaultdict(defaultdict)
        alleles = []
        with open(file, "r") as f:
            for row in f:
                if row[0] in ["#", "-"] or row.strip() == "" or "pos" in row:
                    continue
                else:
                    allele = row.split()[HLAIndex.PICKPOCKET_1_1].replace('*','')
                    pep = row.split()[PeptideIndex.PICKPOCKET_1_1]
                    score = float(row.split()[ScoreIndex.PICKPOCKET_1_1])
                    if allele not in alleles:
                        alleles.append(allele)

                    scores[allele][pep] = score

            result = {allele: {"Score": list(scores.values())[j]} for j, allele in enumerate(alleles)}

        return result

    def get_external_version(self, path=None):
        """
        Returns the external version of the tool by executing
        >{command} --version

        might be dependent on the method and has to be overwritten
        therefore it is declared abstract to enforce the user to
        overwrite the method. The function in the base class can be called
        with super()

        :param str path: Optional specification of executable path if deviant from :attr:`elf.__command`
        :return: The external version of the tool or None if tool does not support versioning
        :rtype: str
        """
        return None

    def prepare_input(self, input, file):
        """
        Prepares input for external tools and writes them to file in the specific format

        No return value!

        :param: list(str) input: The :class:`~epytope.Core.Peptide.Peptide` sequences to write into _file
        :param File file: File-handler to input file for external tool
        """
        file.write("\n".join(input))


class NetCTLpan_1_1(AExternalEpitopePrediction):
    """
    Interface for NetCTLpan 1.1.

    .. note::

        NetCTLpan - Pan-specific MHC class I epitope predictions Stranzl T., Larsen M. V., Lundegaard C., Nielsen M.
        Immunogenetics. 2010 Apr 9. [Epub ahead of print]
    """
    __name = "netctlpan"
    __version = "1.1"
    __supported_length = frozenset([8, 9, 10, 11])
    __allele_import_name = f"{__name}_{__version}".replace('.', '_')
    __alleles = getattr(__import__("epytope.Data.supportedAlleles.external." + __allele_import_name,
                                   fromlist=[__allele_import_name])
                        , __allele_import_name)
    __command = "netCTLpan -f {peptides} -a {alleles} {options} > {out}"


    @property
    def version(self):
        """The version of the predictor"""
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
            return "H-2-%s%s%s" % (allele.locus, allele.supertype, allele.subtype)
        else:
            return "HLA-%s%s:%s" % (allele.locus, allele.supertype, allele.subtype)

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

    @property
    def supportedAlleles(self):
        """
        A list of supported :class:`~epytope.Core.Allele.Allele`
        """
        return self.__alleles

    @property
    def name(self):
        """The name of the predictor"""
        return self.__name

    @property
    def command(self):
        """
        Defines the commandline call for external tool
        """
        return self.__command

    @property
    def supportedLength(self):
        """
        A list of supported :class:`~epytope.Core.Peptide.Peptide` lengths
        """
        return self.__supported_length

    def parse_external_result(self, file):
        """
        Parses external results and returns the result containing the predictors string representation
        of alleles and peptides.

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results
        :rtype: dict
        """
        scores = defaultdict(defaultdict)
        alleles = []
        with open(file, "r") as f:
            for l in f:
                if l.startswith("#") or l.startswith("-") or l.strip() == "":
                    continue
                row = l.strip().split()
                if not row[0].isdigit():
                    continue
            
                epitope = row[PeptideIndex.NETCTLPAN_1_1]
                # Allele input representation differs from output representation. Needs to be in input representation to parse the output properly
                allele = row[HLAIndex.NETCTLPAN_1_1].replace('*','')
                comb_score = float(row[ScoreIndex.NETCTLPAN_1_1])
                if allele not in alleles:
                    alleles.append(allele)

                scores[allele][epitope] = comb_score

        result = {allele: {"Score": list(scores.values())[j]} for j, allele in enumerate(alleles)}
        
        return result

    def get_external_version(self, path=None):
        """
        Returns the external version of the tool by executing
        >{command} --version

        might be dependent on the method and has to be overwritten
        therefore it is declared abstract to enforce the user to
        overwrite the method. The function in the base class can be called
        with super()

        :param str path: Optional specification of executable path if deviant from :attr:`self.__command`
        :return: The external version of the tool or None if tool does not support versioning
        :rtype: str
        """
        return None

    def prepare_input(self, input, file):
        """
        Prepares input for external tools and writes them to file in the specific format

        No return value!

        :param: list(str) input: The :class:`~epytope.Core.Peptide.Peptide` sequences to write into file
        :param File file: File-handler to input file for external tool
        """
        file.write("\n".join(">pepe_%i\n%s" % (i, p) for i, p in enumerate(input)))


class PeptideIndex(IntEnum):
    """
    Specifies the index of the peptide sequence from the parsed output format
    """
    NETMHC_3_0 = 2
    NETMHC_3_4 = 2
    NETMHC_4_0 = 1
    NETMHCPAN_2_4 = 1
    NETMHCPAN_2_8 = 1
    NETMHCPAN_3_0 = 1
    NETMHCPAN_4_0 = 1
    NETMHCPAN_4_1 = 1
    NETMHCSTABPAN_1_0 = 1
    NETMHCII_2_2 = 2
    NETMHCII_2_3 = 2
    NETMHCIIPAN_3_0 = 1
    NETMHCIIPAN_3_1 = 1
    NETMHCIIPAN_4_0 = 1
    NETMHCIIPAN_4_1 = 1
    NETMHCIIPAN_4_3 = 1
    PICKPOCKET_1_1 = 2
    NETCTLPAN_1_1 = 3

class ScoreIndex(IntEnum):
    """
    Specifies the score index from the parsed output format
    """
    NETMHC_3_0 = 2
    NETMHC_3_4 = 3
    NETMHCPAN_2_4 = 3
    NETMHCPAN_2_8 = 3
    NETMHCPAN_3_0 = 4
    NETMHCPAN_4_0 = 5
    NETMHCPAN_4_1 = 5
    NETMHCSTABPAN_1_0 = 6
    NETMHCII_2_2 = 4
    NETMHCII_2_3 = 5
    NETMHCIIPAN_3_0 = 3
    NETMHCIIPAN_3_1 = 3
    NETMHCIIPAN_4_0 = 4
    NETMHCIIPAN_4_1 = 5
    NETMHCIIPAN_4_3 = 6
    PICKPOCKET_1_1 = 4
    NETCTLPAN_1_1 = 7

class RankIndex(IntEnum):
    """
    Specifies the rank index from the parsed output format if there is a rank score provided by the predictor
    """
    NETMHCPAN_2_8 = 5
    NETMHCPAN_3_0 = 6
    NETMHCPAN_4_0 = 7
    NETMHCPAN_4_1 = 6
    NETMHCSTABPAN_1_0 = 5
    NETMHCII_2_3 = 7
    NETMHCIIPAN_3_0 = 5
    NETMHCIIPAN_3_1 = 5
    NETMHCIIPAN_4_0 = 5
    NETMHCIIPAN_4_1 = 6
    NETMHCIIPAN_4_3 = 7

class Offset(IntEnum):
    """
    Specifies the offset of columns for multiple predicted HLA-alleles in the given predictors in order to
    correctly access score and rank per HLA-allele
    """
    NETMHC_4_0 = 3
    NETMHCPAN_2_8 = 3
    NETMHCPAN_3_0 = 4
    NETMHCPAN_4_0 = 5
    NETMHCPAN_4_1 = 4
    NETMHCSTABPAN_1_0_W_SCORE = 8
    NETMHCSTABPAN_1_0_WO_SCORE = 3
    NETMHCIIPAN_3_0 = 3
    NETMHCIIPAN_3_1 = 3
    NETMHCIIPAN_4_0 = 2
    NETMHCIIPAN_4_1 = 3
    NETMHCIIPAN_4_3 = 3

class HLAIndex(IntEnum):
    """
    Specifies the HLA-allele index in the parsed output of the predictor
    """
    NETMHCII_2_2 = 0
    NETMHCII_2_3 = 0
    PICKPOCKET_1_1 = 1
    NETCTLPAN_1_1 = 2
