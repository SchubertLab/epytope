# This code is part of the epytope distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
.. module:: Core.Base
   :synopsis: This module contains base classes for all other modules.
.. moduleauthor:: schubert, szolek, walzer


https://docs.python.org/3/library/abc.html

"""


import abc
import inspect
import os
import subprocess
from collections import defaultdict
import warnings
import pandas as pd


COMPLEMENT = str.maketrans('atgcATGC', 'tacgTACG')


class MetadataLogger(object):
    """
    This class provides a simple interface for assigning additional metadata to
    any object in our data model. Examples: storing ANNOVAR columns like depth,
    base count, dbSNP id, quality information for variants, additional prediction information
    for peptides etc. This functionality is not used from core methods of epytope.

    The saved values are accessed via :meth:`~epytope.Core.MetadataLogger.log_metadata` and
    :meth:`~epytope.Core.MetadataLogger.get_metadata`
    
    """
    def __init__(self):
        """
        """
        self.__metadata = defaultdict(list)

    def log_metadata(self, label, value):
        """
        Inserts a new metadata

        :param str label: key for the metadata that will be added
        :param list(object) value: any kindy of additional value that should be kept
        """
        self.__metadata[label].append(value)

    def get_metadata(self, label, only_first=False):
        """
        Getter for the saved metadata with the key :attr:`label`

        :param str label: key for the metadata that is inferred
        :param bool only_first: true if only the the first element of the matadata list is to be returned
        """
        # although defaultdict *would* return [] if it didn't find label in 
        # self.metadata, it would come with the side effect of adding label as 
        #a key to the defaultdict, so a getter is justified in this case.
        if not only_first:
            return self.__metadata[label] if label in self.__metadata else []
        else:
            return self.__metadata[label][0] if self.__metadata[label] else None


#Metaclass for Plugins
class APluginRegister(abc.ABCMeta):
    """
        This class allows automatic registration of new plugins.
    """

    def __init__(cls, name, bases, nmspc):
        super(APluginRegister, cls).__init__(name, bases, nmspc)

        if not hasattr(cls, 'registry'):
            cls.registry = dict()
        if not inspect.isabstract(cls):
            cls.registry.setdefault(str(cls().name).lower(), {}).update({str(cls().version).lower():cls})

    def __getitem__(cls, args):
        name, version = args
        if version is None:
            return cls.registry[name][max(cls.registry[name].keys())]
        return cls.registry[name][version]

    def __iter__(cls):
        return iter(cls.registry.values())

    def __str__(cls):
        if cls in cls.registry:
            return cls.__name__
        return cls.__name__ + ": " + ", ".join([sc.__name__ for sc in cls])


class ACleavageSitePrediction(object, metaclass=APluginRegister):
    @abc.abstractproperty
    def name(self):
        """
        The name of the predictor
        """
        raise NotImplementedError

    @abc.abstractproperty
    def version(self):
        """
        Parameter specifying the version of the prediction method
        """
        raise NotImplementedError

    @abc.abstractproperty
    def supportedLength(self):
        """
        The supported lengths of the predictor
        """
        raise NotImplementedError

    @abc.abstractproperty
    def cleavagePos(self):
        """
        Parameter specifying the position of aa (within the prediction window) after which the sequence is cleaved
        (starting from 1)
        """
        raise NotImplementedError


    @abc.abstractmethod
    def predict(self, aa_seq, **kwargs):
        """
        Predicts the proteasomal cleavage site of the given sequences

        :param aa_seq: The sequence to be cleaved
        :type aa_seq: :class:`~epytope.Core.Peptide.Peptide` or :class:`~epytope.Core.Protein.Protein`
        :return: Returns a :class:`~epytope.Core.Result.AResult` object for the specified Bio.Seq
        :rtype: :class:`~epytope.Core.Result.AResult`
        """
        raise NotImplementedError


class ACleavageFragmentPrediction(object, metaclass=APluginRegister):
    @abc.abstractproperty
    def name(self):
        """
        The name of the predictor
        """
        raise NotImplementedError

    @abc.abstractproperty
    def version(self):
        """
        Parameter specifying the version of the prediction method
        """
        raise NotImplementedError

    @abc.abstractproperty
    def supportedLength(self):
        """
        The supported lengths of the predictor
        """
        raise NotImplementedError


    @abc.abstractproperty
    def cleavagePos(self):
        """
        Parameter specifying the position of aa (within the prediction window) after which the sequence is cleaved
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, aa_seq, **kwargs):
        """
        Predicts the probability that the fragment can be produced by the proteasom

        :param aa_seq: The sequence to be cleaved
        :type aa_seq: :class:`~epytope.Core.Peptide.Peptide`
        :return: Returns a :class:`~epytope.Core.Result.AResult` object for the specified Bio.Seq
        :rtype: :class:`~epytope.Core.Result.AResult`
        """
        raise NotImplementedError


class AEpitopePrediction(object, metaclass=APluginRegister):
    @abc.abstractproperty
    def name(self):
        """
        The name of the predictor
        """
        raise NotImplementedError

    @abc.abstractproperty
    def version(cls):
        """The version of the predictor"""
        raise NotImplementedError

    @abc.abstractproperty
    def supportedAlleles(self):
        """
        A list of valid allele models
        """
        raise NotImplementedError

    @abc.abstractproperty
    def supportedLength(self):
        """
        A list of supported peptide lengths
        """
        raise NotImplementedError

    @abc.abstractmethod
    def convert_alleles(self, alleles):
        """
        Converts alleles into the internal allele representation of the predictor
        and returns a string representation

        :param alleles: The alleles for which the internal predictor representation is needed
        :type alleles: list(:class:`~epytope.Core.Allele.Allele`)
        :return: Returns a string representation of the input alleles
        :rtype: list(str)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, peptides, alleles=None, **kwargs):
        """
        Predicts the binding affinity for a given peptide or peptide lists for a given list of alleles.
        If alleles is not given, predictions for all valid alleles of the predictor is performed. If, however,
        a list of alleles is given, predictions for the valid allele subset is performed.

        :param peptides: The peptide objects for which predictions should be performed
        :type peptides: :class:`~epytope.Core.Peptide.Peptide` or list(:class:`~epytope.Core.Peptide.Peptide`)
        :param alleles: An :class:`~epytope.Core.Allele.Allele` or list of :class:`~epytope.Core.Allele.Allele` for which
                        prediction models should be used
        :type alleles: :class:`~epytope.Core.Allele.Allele`/list(:class:`~epytope.Core.Allele.Allele`)
        :return: Returns a :class:`~epytope.Core.Result.AResult` object for the specified
                 :class:`~epytope.Core.Peptide.Peptide` and :class:`~epytope.Core.Allele.Allele`
        :rtype: :class:`~epytope.Core.Result.AResult`
        """
        raise NotImplementedError


class ASVM(object, metaclass=abc.ABCMeta):
    """
        Base class for SVM prediction tools
    """

    @abc.abstractmethod
    def encode(self, peptides):
        """
        Returns the feature encoding for peptides

        :param peptides: List of or a single :class:`~epytope.Core.Peptide.Peptide` object
        :type peptides: list(:class:`~epytope.Core.Peptide.Peptide`)/:class:`~epytope.Core.Peptide.Peptide`
        :return: Feature encoding of the Peptide objects
        :rtype: list(Object)
        """
        raise NotImplementedError


class AExternal(object, metaclass=abc.ABCMeta):
    """
     Base class for external tools
    """

    @abc.abstractproperty
    def command(self):
        """
        Defines the commandline call for external tool
        """
        raise NotImplementedError

    @abc.abstractmethod
    def parse_external_result(self, file):
        """
        Parses external results and returns the result

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results
        :rtype: dict
        """
        raise NotImplementedError

    def is_in_path(self):
        """
        Checks whether the specified execution command can be found in PATH

        :return: Whether or not command could be found in PATH
        :rtype: bool
        """
        exe = self.command.split()[0]
        for try_path in os.environ["PATH"].split(os.pathsep):
            try_path = try_path.strip('"')
            exe_try = os.path.join(try_path, exe).strip()
            if os.path.isfile(exe_try) and os.access(exe_try, os.X_OK):
                return True
        return False

    @abc.abstractmethod
    def get_external_version(self, path=None):
        """
        Returns the external version of the tool by executing
        >{command} --version

        might be dependent on the method and has to be overwritten
        therefore it is declared abstract to enforce the user to
        overwrite the method. The function in the base class can be called
        with super()

        :param str path: - Optional specification of executable path if deviant from self.__command
        :return: The external version of the tool or None if tool does not support versioning
        :rtype: str
        """
        exe = self.command.split()[0] if path is None else path
        try:
            p = subprocess.Popen(exe + ' --version', shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            #p.wait() #block the rest
            stdo, stde = p.communicate()
            stdr = p.returncode
            if stdr > 0:
                raise RuntimeError("Could not check version of " + exe + " - Please check your installation and epytope "
                                                                         "wrapper implementation.")
        except Exception as e:
                raise RuntimeError(e)
        return str(stdo).strip()


class ATAPPrediction(object, metaclass=APluginRegister):
    @abc.abstractproperty
    def name(self):
        """
        The name of the predictor
        """
        raise NotImplementedError

    @abc.abstractproperty
    def version(self):
        """
        Parameter specifying the version of the prediction method
        """
        raise NotImplementedError

    @abc.abstractproperty
    def supportedLength(self):
        """
        The supported lengths of the predictor
        """
        raise NotImplementedError


    @abc.abstractmethod
    def predict(self, peptides, **kwargs):
        """
        Predicts the TAP affinity for the given sequences

        :param peptides: :class:`~epytope.Core.Peptide.Peptide` for which TAP affinity should be predicted
        :type peptides: list(:class:`~epytope.Core.Peptide.Peptide`)/:class:`~epytope.Core.Peptide.Peptide`
        :return: Returns a :class:`~epytope.Core.Result.TAPResult` object
        :rtype: :class:`~epytope.Core.Result.TAPResult`
        """
        raise NotImplementedError


class AHLATyping(object, metaclass=APluginRegister):
    @abc.abstractproperty
    def name(self):
        """
        The name of the predictor
        """
        raise NotImplementedError

    @abc.abstractproperty
    def version(self):
        """
        Parameter specifying the version of the prediction method
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, ngsFile, output, **kwargs):
        """
        Prediction method for inferring the HLA typing

        :param str ngsFile: The path to the input file containing the NGS reads
        :param str output: The path to the output file or directory
        :return: A list of HLA alleles representing the genotype predicted by the algorithm
        :rtype: list(:class:`~epytope.Core.Allele.Allele`)
        """
        raise NotImplementedError


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    def new_func(*args, **kwargs):
        warnings.simplefilter('default')  #this will render these deprecation warnings visible to everyone (default is switched off in python >=2.7)
        warnings.warn("Call to deprecated function {n} of {f}.".format(n=func.__name__, f=func.__doc__),
                      category=DeprecationWarning)
        return func(*args, **kwargs)
    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func

class ATCRSpecificityPrediction(object, metaclass=APluginRegister):
    @property
    @abc.abstractmethod
    def name(self):
        """
        The name of the predictor
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def version(cls):
        """
        The version of the predictor
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def supportedPeptides(self):
        """
        A list of valid Peptides
        """
        raise NotImplementedError

    @abc.abstractmethod
    def invalid(self, seq: str) -> bool:
        """
        checks if the passed sequence is an invalid protein sequence
        :param str seq: a String representing the protein sequence
        :return: Returns true if the passed sequence is not a protein sequence
        :rtype: bool
        """
        raise NotImplementedError

    @abc.abstractmethod
    def valid_epitope(seq: str) -> bool:
        """
        check if the epitope has a specific length and if the sequence is a protein sequence
        :param str seq: a string representing the epitope
        :return: true if the epitope is valid
        :rtype: bool
        """

        raise NotImplementedError

    @abc.abstractmethod
    def canonical_CDR(seq: str) -> bool:
        """
        check if the CDR sequence starts with a Cysteine and ends with a Phenylalanine
        :param str seq: a string representing the CDR sequence
        :return: true if the CDR sequence is canonical
        :rtype: bool
        """

        raise NotImplementedError

    @abc.abstractmethod
    def trimming_cdr3(seq: str) -> str:
        """
        check if the CDR sequence starts with a Cysteine and ends with a Phenylalanine respectively tryptophan, if this
        is the case remove cysteine and Phenylalanine respectively tryptophan from the start and the end respectively
        :param str seq: a string representing the CDR sequence
        :return: seq[1:-1] if it starts with C and ends with F/W
        :rtype: str
        """

        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, peptides, TCRs, repository: str, all: bool, dataset: str = None, trained_on: str=None):
        """
        Predicts binding probability between a T-cell receptor CDR3 protein sequence and a peptide
        If alleles is not given, predictions for all valid alleles of the predictor is performed. If, however,
        a list of alleles is given, predictions for the valid allele subset is performed.
        :param peptides: The peptide objects for which predictions should be performed
        :type peptides: :class:`~epytope.Core.Peptide.Peptide` or list(:class:`~epytope.Core.Peptide.Peptide`)
        :param TCRs: T cell receptor objects
        :type  :class:'~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor' or
        list(:class:'~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor')
        :param str repository: a path to a local github repository of the desired predictor
        :param bool all: if true each TCR object will be joined with each peptide to perform the prediction, otherwise
        the prediction will be preformed in the same order of the passed peptides and TCRs objects
        :param str dataset: specifying the dataset the model trained on
        :param str trained_on: specifying the dataset the model trained on
        :return: Returns a :class:`~epytope.Core.Result.AResult` object for the specified
                 :class:`~epytope.Core.Peptide.Peptide` and :class:`~epytope.Core.Allele.Allele`
        :rtype: :class:`~epytope.Core.Result.AResult`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict_from_dataset(self, repository: str, path : str = None, df: pd.DataFrame = None, source: str = "",
                             score: int = 1, trained_on: str=None):
        """
        Predicts binding probability between a T-cell receptor CDR3 protein sequence and a peptide.
        The path should lead to csv file with fixed column names dataset.columns = ['TRA', 'TRB', "TRAV", "TRAJ",
        "TRBV", "TRBJ", "T-Cell-Type", "Peptide", "MHC", "Species", "Antigen.species", "Tissue"]. If some values for
        one or more variables are unavailable, leave them as blank cells.
        :param str repository: a path to a local github repository of the desired predictor
        :param str path: a string representing a path to the dataset(csv file), which will be precessed. Default value
        is None, when the dataframe object is given
        :param `pd.DataFrame` df: a dataframe object. Default value is None, when the path is given
        :param str source: the source of the dataset [vdjdb, mcpas, scirpy, IEDB]. If this parameter does not be passed,
         the dataset should be a csv file with the column names mentioned above
        :param int score: An integer representing a confidence score between 0 and 3 (0: critical information missing,
        1: medium confidence, 2: high confidence, 3: very high confidence). By processing all entries with a confidence
        score >= the passed parameter score will be kept. Default value is 1
        :param str trained_on: specifying the dataset the model trained on
        :return: A :class:`~epytope.Core.TCRSpecificityPredictionResult` object
        :rtype: :class:`~epytope.Core.TCRSpecificityPredictionResult`
        """
        raise NotImplementedError