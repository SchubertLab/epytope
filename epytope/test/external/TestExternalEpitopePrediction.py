"""
Unittest for external epitope prediction methods
"""

import unittest
import os

from epytope.Core import Allele, CombinedAllele
from epytope.Core import Peptide
from epytope.Core import Transcript

from epytope.EpitopePrediction import EpitopePredictorFactory
from epytope.EpitopePrediction import AExternalEpitopePrediction
from epytope.EpitopePrediction import NetMHC_3_4


#only for internal testing
class NetMHC_0_1(NetMHC_3_4):

    __version = "0.1"

    @property
    def version(self):
        return self.__version


class TestExternalEpitopePredictionClass(unittest.TestCase):

    def setUp(self):
        self.peptides_mhcI = [Peptide("SYFPEITHI"), Peptide("IHTIEPFYS")]
        self.peptides_mhcII = [Peptide("AAAAAASYFPEITHI"), Peptide("IHTIEPFYSAAAAAA")]
        self.mhcI = [Allele("HLA-B*07:02"), Allele("HLA-A*02:01")]
        self.mhcII = [Allele("HLA-DRB1*07:01"), Allele("HLA-DRB1*15:01")]
        self.mhcII_combined_alleles = [CombinedAllele("DPA1*01:03-DPB1*01:01"), CombinedAllele("DQA1*06:02-DQB1*06:31")]
<<<<<<< HEAD
=======
        self.mouse = [Allele("H2-Kd"), Allele("H2-Kb")]
>>>>>>> repo-b2/main
        self.transcript = Transcript("")

    def test_multiple_inputs(self):
        for m in EpitopePredictorFactory.available_methods():
            for v in EpitopePredictorFactory.available_methods()[m]:
                mo = EpitopePredictorFactory(m, version=v)
                if isinstance(mo, AExternalEpitopePrediction) and not (mo.version=="0.1" and mo.name=="netmhc"):
                    print("Testing", mo.name, "version", mo.version)
                    try:
                        if any(a.name in mo.supportedAlleles for a in self.mhcII):
                            mo.predict(self.peptides_mhcII, alleles=self.mhcII)
                        if any(a.name in mo.supportedAlleles for a in self.mhcII_combined_alleles):
                            mo.predict(self.peptides_mhcII, alleles=self.mhcII_combined_alleles)
                        if any(a.name in mo.supportedAlleles for a in self.mhcI):
                            mo.predict(self.peptides_mhcI, alleles=self.mhcI)
                        print("Success")
                    except RuntimeError as e: #catch only those stemming from binary unavailability
                        if "could not be found in PATH" not in e.message:
                            raise e #all others do not except
                        else:
                            print(mo.name, "not available")

    def test_single_epitope_input(self):
        for m in EpitopePredictorFactory.available_methods():
            for v in EpitopePredictorFactory.available_methods()[m]:
                mo = EpitopePredictorFactory(m, version=v)
                if isinstance(mo, AExternalEpitopePrediction) and not (mo.version=="0.1" and mo.name=="netmhc"):
                    print("Testing", mo.name, "version", mo.version)
                    try:
                        if any(a.name in mo.supportedAlleles for a in self.mhcII):
                            mo.predict(self.peptides_mhcII[0], alleles=self.mhcII)
                        if any(a.name in mo.supportedAlleles for a in self.mhcII_combined_alleles):
                            mo.predict(self.peptides_mhcII[0], alleles=self.mhcII_combined_alleles)
                        if any(a.name in mo.supportedAlleles for a in self.mhcI):
                            mo.predict(self.peptides_mhcI[0], alleles=self.mhcI)
                        print("Success")
                    except RuntimeError as e: #catch only those stemming from binary unavailability
                        if "could not be found in PATH" not in e.message:
                            raise e #all others do not except
                        else:
                            print(mo.name, "not available")

    def test_single_allele_input(self):
        for m in EpitopePredictorFactory.available_methods():
            for v in EpitopePredictorFactory.available_methods()[m]:
                mo = EpitopePredictorFactory(m, version=v)
                if isinstance(mo, AExternalEpitopePrediction) and not (mo.version=="0.1" and mo.name=="netmhc"):
                    print("Testing", mo.name, "version", mo.version)
                    try:
                        if any(a.name in mo.supportedAlleles for a in self.mhcII):
                            mo.predict(self.peptides_mhcII, alleles=self.mhcII[0])
                        if any(a.name in mo.supportedAlleles for a in self.mhcII_combined_alleles):
                            mo.predict(self.peptides_mhcII, alleles=self.mhcII_combined_alleles[0])
                        if any(a.name in mo.supportedAlleles for a in self.mhcI):
                            mo.predict(self.peptides_mhcI, alleles=self.mhcI[0])
                        print("Success")
                    except RuntimeError as e: #catch only those stemming from binary unavailability
                        if "could not be found in PATH" not in e.message:
                            raise e #all others do not except
                        else:
                            print(mo.name, "not available")

    def test_wrong_epitope_input(self):
        with self.assertRaises(ValueError):
            EpitopePredictorFactory("NetMHC").predict(self.transcript, alleles=self.mhcI)

    def test_wrong_allele_input(self):
        with self.assertRaises(ValueError):
            EpitopePredictorFactory("NetMHC").predict(self.mhcI, alleles=self.transcript)

    def test_wrong_internal_to_external_version(self):
        with self.assertRaises(RuntimeError):
            EpitopePredictorFactory("NetMHC", version="0.1").predict(self.peptides_mhcI, alleles=self.mhcI)

<<<<<<< HEAD
#--sort flag not supported by newer versions

    # def test_path_option_and_optional_parameters_netmhc(self):
    #     netmhc = EpitopePredictorFactory("NetMHC")
    #     exe = netmhc.command.split()[0]
    #     for try_path in os.environ["PATH"].split(os.pathsep):
    #         try_path = try_path.strip('"')
    #         exe_try = os.path.join(try_path, exe).strip()
    #         if os.path.isfile(exe_try) and os.access(exe_try, os.X_OK):
    #             r = netmhc.predict(self.peptides_mhcI, alleles=self.mhcI, command=exe_try, options="--sort", chunksize=1)
    #             self.assertTrue(len(r) == len(self.peptides_mhcI))
    #             self.assertAlmostEqual(r["A*02:01"]["SYFPEITHI"]["netmhc"], 0.150579105869, places=7, msg=None, delta=None)
    #             self.assertAlmostEqual(r["A*02:01"]["IHTIEPFYS"]["netmhc"], 0.0619540879359, places=7, msg=None, delta=None)
=======
    def test_mouse_allele_prediction(self):
        for m in EpitopePredictorFactory.available_methods():
            for v in EpitopePredictorFactory.available_methods()[m]:
                mo = EpitopePredictorFactory(m, version=v)
                if isinstance(mo, AExternalEpitopePrediction) and not (mo.version=="0.1" and mo.name=="netmhc"):
                    print("Testing", mo.name, "version", mo.version)
                    try:
                        if any(a.name in mo.supportedAlleles for a in self.mouse):
                            mo.predict(self.peptides_mhcI[0], alleles=self.mouse)
                        print("Success")
                    except RuntimeError as e: #catch only those stemming from binary unavailability
                        if "could not be found in PATH" not in e.message:
                            raise e #all others do not except
                        else:
                            print(mo.name, "not available")
>>>>>>> repo-b2/main

    def test_path_and_optional_parameters_netctl(self):
        netctlpan = EpitopePredictorFactory("NetCTLpan")
        exe = netctlpan.command.split()[0]
        for try_path in os.environ["PATH"].split(os.pathsep):
            try_path = try_path.strip('"')
            exe_try = os.path.join(try_path, exe).strip()
            if os.path.isfile(exe_try) and os.access(exe_try, os.X_OK):
                print(netctlpan.predict(self.peptides_mhcI, alleles=self.mhcI,
                                        commad=exe_try,
                                        options="-wt 0.05 -wc 0.225 -ethr 0.5"))

if __name__ == '__main__':
    unittest.main()