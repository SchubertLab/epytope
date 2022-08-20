# This code is part of the epytope distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
.. module:: EpitopePrediction
   :synopsis: Factory classes for cleavage site and fragment prediction.
              This is the entry point to all cleavage prediction methods.
.. moduleauthor:: schubert

"""

from epytope.Core.Base import AEpitopePrediction
from epytope.EpitopePrediction.External import *
from epytope.EpitopePrediction.PSSM import *
# from epytope.EpitopePrediction.SVM import *
from epytope.EpitopePrediction.ANN import *
try:
    from fred_plugin import *
except ImportError:
    pass


class MetaclassEpitope(type):
    def __init__(cls, name, bases, nmspc):
        type.__init__(cls, name, bases, nmspc)

    def __call__(self, _predictor, *args, **kwargs):
        '''
        If a third person wants to write a new Epitope Predictior. He/She has to name the file fred_plugin and
        inherit from AEpitopePrediction. That's it nothing more.
        '''

        version = str(kwargs["version"]).lower() if "version" in kwargs else None
        try:
            return AEpitopePrediction[str(_predictor).lower(), version](*args)
        except KeyError as e:
            if version is None:
                raise ValueError("Predictor %s is not known. Please verify that such an Predictor is " % _predictor +
                                 "supported by epytope and inherits AEpitopePredictor.")
            else:
                raise ValueError("Predictor %s version %s is not known. Please verify that such an Predictor is " % (
                _predictor, version) +
                                 "supported by epytope and inherits AEpitopePredictor.")


class EpitopePredictorFactory(metaclass=MetaclassEpitope):

    @staticmethod
    def available_methods():
        """
        Returns a dictionary of available epitope predictors and their supported versions

        :return: dict(str,list(str) - dictionary of epitope predictors represented as string and a list of supported
                                      versions
        """
        return {k: sorted(versions.keys()) for k, versions in AEpitopePrediction.registry.items()}