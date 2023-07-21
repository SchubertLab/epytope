__author__ = 'albahah'

from epytope.TCRSimilarityMeasurement.SimilarityTools import *
from epytope.Core.Base import ATCRSimilarityMeasurement
try:
    from fred_plugin import *
except ImportError:
    pass


class MetaclassTCRSimilarity(type):
    def __init__(cls, name, bases, nmspc):
        type.__init__(cls, name, bases, nmspc)

    def __call__(self, _predictor, *args, **kwargs):
        """
        If a third person wants to write a new distance measurement method. One has to name the file fred_plugin and
        inherit from ATCRSimilarityMeasurement. That's it nothing more.
        """

        version = str(kwargs["version"]).lower() if "version" in kwargs else None
        try:
            return ATCRSimilarityMeasurement[str(_predictor).lower(), version](*args)
        except KeyError as e:
            if version is None:
                raise ValueError(
                    "Predictor %s is not known. Please verify that such an Predictor is " % _predictor +
                    "supported by epytope and inherits ATCRSimilarityMeasurement.")
            else:
                raise ValueError(
                    "Predictor %s version %s is not known. Please verify that such an Predictor is " % (
                        _predictor, version) +
                    "supported by epytope and inherits ATCRSimilarityMeasurement.")


class TCRSimilarityMeasurementFactory(metaclass=MetaclassTCRSimilarity):

    @staticmethod
    def available_methods():
        """
        Returns a list of available TCR similarity measurement methods
        :return: dict(str,list(int)) - dict of TCR similarity measurement methods represented as string and the
        supported versions
        """
        return {k: sorted(versions.keys()) for k, versions in ATCRSimilarityMeasurement.registry.items()}