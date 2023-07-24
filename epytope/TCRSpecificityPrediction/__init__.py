__author__ = 'albahah'


from epytope.Core.Base import ATCRSpecificityPrediction
from epytope.TCRSpecificityPrediction.External import *

try:
    from fred_plugin import *
except ImportError:
    pass


class MetaclassTCRSpecificity(type):
    def __init__(cls, name, bases, nmspc):
        type.__init__(cls, name, bases, nmspc)

    def __call__(self, _predictor, *args, **kwargs):
        """
        If a third person wants to write a new TCR specificity Predictor. One has to name the file fred_plugin and
        inherit from ATCRSpecificityPrediction. That's it nothing more.
        """

        version = str(kwargs["version"]).lower() if "version" in kwargs else None
        try:
            return ATCRSpecificityPrediction[str(_predictor).lower(), version](*args)
        except KeyError as e:
            if version is None:
                raise ValueError(
                    "Predictor %s is not known. Please verify that such an Predictor is " % _predictor +
                    "supported by epytope and inherits ATCRSpecificityPrediction.")
            else:
                raise ValueError(
                    "Predictor %s version %s is not known. Please verify that such an Predictor is " % (
                        _predictor, version) +
                    "supported by epytope and inherits ATCRSpecificityPrediction.")


class TCRSpecificityPredictorFactory(metaclass=MetaclassTCRSpecificity):
    @staticmethod
    def available_methods():
        """
        Returns a list of available TCR specificity prediction methods
        :return: dict(str,list(int)) - dict of TCR specificity predictor represented as string and the supported
        versions
        """
        return {k: sorted(versions.keys()) for k, versions in ATCRSpecificityPrediction.registry.items()}


