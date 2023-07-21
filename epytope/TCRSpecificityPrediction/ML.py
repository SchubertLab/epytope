# coding=utf-8
# This code is part of the epytope distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
.. module:: TCRSpecificityPrediction.ML
   :synopsis: This module contains all classes for ML-based TCR-epitope binding prediction.
.. moduleauthor:: albahah
"""

import os
import subprocess
import tempfile
import numpy as np
import pandas as pd
from tempfile import NamedTemporaryFile
from epytope.Core.Base import AExternal
from epytope.TCRSpecificityPrediction.External import AExternalTCRSpecificityPrediction
from epytope.Core.Result import TCRSpecificityPredictionResult
import re
from pytoda.proteins.utils import aas_to_smiles
from pathlib import Path
from typing import Tuple


class Ergo2(AExternalTCRSpecificityPrediction, AExternal):
    """
    Implements ERGO, a deep learning based method for predicting TCR and epitope peptide binding.

    """
    __name = "ERGO-II"
    __command = "python Predict.py"
    __version = " "

    @property
    def version(self) -> str:
        """
        The version of the Method
        """
        return self.__version

    @property
    def command(self) -> str:
        """
        Defines the commandline call for external tool
        """
        return self.__command

    @property
    def name(self) -> str:
        """The name of the predictor"""
        return self.__name

    @property
    def supportedPeptides(self) -> list:
        """
        A list of valid Peptides
        """
        return []

    def parse_external_result(self, file: str, df: pd.DataFrame):
        """
        Parses external results and returns the result
        :param str file: The file path or the external prediction results
        :param pd.DataFrame df: the complete processed dataframe
        :return: A dictionary containing the prediction results
        :rtype: dict{(str, str, str, str): float} {(Receptor_ID, TRA, TRB, Peptide): score}
        """
        result = pd.read_csv(file)[["TRA", "TRB", "Peptide", "Score"]]
        result.index = df.index
        result["Score"].astype(float)
        result.insert(0, "Receptor_ID", df["Receptor_ID"])
        result.fillna("", inplace=True)
        result["Receptor_ID"].astype(str)
        return {self.name: {
                            row[:4]: float("{:.4f}".format(row[4]))
                            for row in result.itertuples(index=False)
                           }
               }

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
        return None

    def predict(self, peptides, TCRs, repository: str, all: bool, **kwargs):
        """
        Overwrites ATCRSpecificityPrediction.predict

        Predicts binding probability between a T-cell receptor CDR3 protein sequence and a peptide.
        :param peptides: The TCREpitope objects for which predictions should be performed
        :type peptides: :class:`~epytope.Core.TCREpitope.TCREpitope` or list(:class:`~epytope.Core.TCREpitope.TCREpitope`)
        :param TCRs: T cell receptor objects
        :type  :class:'~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor' or
        list(:class:'~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor')
        :param str repository: a path to a local github repository of ERGO-II predictor
        :param bool all: if true each TCR object will be joined with each peptide to perform the prediction, otherwise
        the prediction will be preformed in the same order of the passed peptides and TCRs objects
        :param str trained_on: specifying the dataset the model trained on. This parameter is specific for ERGO, which
        has two models, one is trained on vdjdb and the other is trained on McPAS dataset.
        :param trained_model a string representing a path to the trained model directory. For ERGO this parameter will
        be ignored
        :param down: a boolean value for choosing from two different models of ImRex. If it is set to Ture, the model,
        trained of down sampled dataset of vdjdb will be selected, otherwise the other trained model will be chosen.
        Default value is False.
        :param nettcr_chain: a string specifying the chain(s) to use (a, b, ab). Default: b.
        :param pMTnet_interpreter: a string representing a path to python interpreter for pMTnet.
        :param chain: a string representing the type of the used cdr3-sequence in the prediction. It can be set to 'a'
        for alpha- or 'b' for beta-sequences. This parameter specifics the input chain for ATM-TCR.
        :param padding str: can be set to one of the following values ['front, end, mid, alignment'] and specifics the
        padding type for ATM-TCR.
        :param cuda bool: it can be set to True, if the running device has cuda.
        :return: A :class:`~epytope.Core.TCRSpecificityPredictionResult` object
        :rtype: :class:`~epytope.Core.TCRSpecificityPredictionResult`
        """
        df = super().predict(peptides=peptides, TCRs=TCRs, repository=repository, all=all)
        if "trained_on" in kwargs:
            trained_on = kwargs["trained_on"]
        else:
            trained_on = "vdjdb"
        df_result = self.predict_from_dataset(repository=repository, df=df, score=-1, trained_on=trained_on)
        return df_result

    def predict_from_dataset(self, repository: str, path: str = None, df: pd.DataFrame = None, source: str = "",
                             score: int = 1, **kwargs):
        """
        Predicts binding probability between a T-cell receptor CDR3 protein sequence and a peptide.
        The path should lead to csv file with fixed column names dataset.columns = ['TRA', 'TRB', "TRAV", "TRAJ",
        "TRBV", "TRBJ", "T-Cell-Type", "Peptide", "MHC", "Species", "Antigen.species", "Tissue"]. If some values for
        one or more variables are unavailable, leave them as blank cells.
        :param str repository: a path to a local github repository of ERGO-II predictor
        :param str path: a string representing a path to the dataset(csv file), which will be precessed. Default value
        is None, when the dataframe object is given
        :param `pd.DataFrame` df: a dataframe object. Default value is None, when the path is given
        :param str source: the source of the dataset [vdjdb, mcpas, scirpy, IEDB]. If this parameter is not passed,
         the dataset should be a csv file with the column names mentioned above
        :param int score: An integer representing a confidence score between 0 and 3 (0: critical information missing,
        1: medium confidence, 2: high confidence, 3: very high confidence). By processing all entries with a confidence
        score >= the passed parameter score will be kept. Default value is 1
        :param str trained_on: specifying the dataset the model trained on. This parameter is specific for ERGO, which
        has two models, one is trained on vdjdb and the other is trained on McPAS dataset.
        :param trained_model a string representing a path to the trained model directory. For ERGO this parameter will
        be ignored
        :param down: a boolean value for choosing from two different models of ImRex. If it is set to Ture, the model,
        trained of down sampled dataset of vdjdb will be selected, otherwise the other trained model will be chosen.
        Default value is False.
        :param nettcr_chain: a string specifying the chain(s) to use (a, b, ab). Default: b.
        :param pMTnet_interpreter: a string representing a path to python interpreter for pMTnet.
        :param chain: a string representing the type of the used cdr3-sequence in the prediction. It can be set to 'a'
        for alpha- or 'b' for beta-sequences. This parameter specifics the input chain for ATM-TCR.
        :param padding str: can be set to one of the following values ['front, end, mid, alignment'] and specifics the
        padding type for ATM-TCR.
        :param cuda bool: it can be set to True, if the running device has cuda.
        :return: A :class:`~epytope.Core.TCRSpecificityPredictionResult` object
        :rtype: :class:`~epytope.Core.TCRSpecificityPredictionResult`
        """
        def substring(column):
            """
            helper function to get gene allele annotation from family name of v,j regions
            :param column: pd.Series, where entries are the family name of v,j regions
            """
            return column.apply(lambda x: re.search(r"^\w*(-\d+)*", str(x)).group() if x != "" else x)

        if path is None and df is None:
            raise FileNotFoundError("A path to a csv file or a dataframe should be passed")
        if df is None:
            if os.path.isfile(path):
                df = process_dataset_TCR(path=path, source=source, score=score)
            else:
                raise FileNotFoundError("A path to a csv file or a dataframe should be passed")
        else:
            df = process_dataset_TCR(df=df, source=source, score=score)
        if not os.path.isdir(repository):
            raise NotADirectoryError("please pass a path as a string to a local ERGO-II repository. To clone the "
                                     "repository type: 'git clone https://github.com/IdoSpringer/ERGO-II.git' in the "
                                     "terminal")
        # get only gene allele annotation form family name of v, j regions respectively
        df[["TRAV", "TRAJ", "TRBV", "TRBJ"]] = df[["TRAV", "TRAJ", "TRBV", "TRBJ"]].apply(substring)
        ergo_df = df[['TRA', 'TRB', "TRAV", "TRAJ", "TRBV", "TRBJ", "T-Cell-Type", "Peptide", "MHC"]]
        ergo_df = ergo_df[(ergo_df["Peptide"] != "") & (ergo_df["TRB"] != "")]
        script_path = os.path.join(repository, "Predict.py")
        if not os.path.exists(script_path):
            raise FileNotFoundError("Please pass a path to the local ERGO-II repository on your computer")
        else:
            # editing Predict script to save the result as csv file into the tmp_out file
            os.chdir(repository)
            script = []
            with open("Predict.py", "r") as f:
                script.extend(f.readlines())
            if "    df.to_csv(sys.argv[3], sep=',', index=False)\n" not in script:
                idx = script.index('    df = predict(sys.argv[1], sys.argv[2])\n')
                script.insert(idx + 1, "    df.to_csv(sys.argv[3], sep=',', index=False)\n")
                with open("Predict.py", "w") as f:
                    f.writelines(script)
        tmp_file = NamedTemporaryFile(delete=False)
        ergo_df.to_csv(tmp_file.name, sep=",", index=False)
        tmp_out = NamedTemporaryFile(delete=False)
        if "trained_on" in kwargs:
            if kwargs["trained_on"].lower() in ["vdjdb", "mcpas"]:
                trained_on = kwargs["trained_on"]
            else:
                trained_on = "vdjdb"
        elif source and source.lower() in ["vdjdb", "mcpas"]:
            trained_on = source.lower()
        else:
            trained_on = "vdjdb"
        try:
            cmd = self.__command
            cmd += f" {trained_on.lower()} {tmp_file.name} {tmp_out.name}"
            p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
            stdo, stde = p.communicate()
            stdr = p.returncode
            if stdr > 0:
                raise RuntimeError("Unsuccessful execution of " + cmd + " (EXIT!=0) with output:\n" + stdo.decode())
            if os.path.getsize(tmp_out.name) == 0:
                raise RuntimeError(
                    "Unsuccessful execution of " + cmd + " (empty output file) with output:\n" +
                    stdo.decode())
        except Exception as e:
            raise RuntimeError(e)
        os.remove(tmp_file.name)
        result = self.parse_external_result(tmp_out.name, df)
        tmp_out.close()
        os.remove(tmp_out.name)
        df_result = TCRSpecificityPredictionResult.from_dict(result)
        df_result.index = pd.MultiIndex.from_tuples([tuple((ID, TRA, TRB, pep)) for ID, TRA, TRB, pep in df_result.index],
                                                        names=["Receptor_ID", 'TRA', 'TRB', "Peptide"])
        return df_result


class TITAN(AExternalTCRSpecificityPrediction, AExternal):
    """
    Implements Tcr epITope bimodal Attention Networks (TITAN). The provided trained model can be downloaded from

    """
    __name = "TITAN"
    __command = "python scripts/flexible_model_eval.py"
    __version = " "
    _v_regions = ['TRBV1*01', 'TRBV10-1*01', 'TRBV10-1*02', 'TRBV10-2*01', 'TRBV10-2*02', 'TRBV10-3*01', 'TRBV10-3*02',
                  'TRBV10-3*03', 'TRBV10-3*04', 'TRBV11-1*01', 'TRBV11-2*01', 'TRBV11-2*02', 'TRBV11-2*03',
                  'TRBV11-3*01', 'TRBV11-3*02', 'TRBV11-3*03', 'TRBV11-3*04', 'TRBV12-1*01', 'TRBV12-2*01',
                  'TRBV12-3*01', 'TRBV12-4*01', 'TRBV12-4*02', 'TRBV12-5*01', 'TRBV13*01', 'TRBV13*02', 'TRBV14*01',
                  'TRBV14*02', 'TRBV15*01', 'TRBV15*02', 'TRBV15*03', 'TRBV16*01', 'TRBV16*02', 'TRBV16*03',
                  'TRBV17*01', 'TRBV17*02', 'TRBV18*01', 'TRBV19*01', 'TRBV19*02', 'TRBV19*03', 'TRBV2*01', 'TRBV2*02',
                  'TRBV2*03', 'TRBV20-1*01', 'TRBV20-1*02', 'TRBV20-1*03', 'TRBV20-1*04', 'TRBV20-1*05', 'TRBV20-1*06',
                  'TRBV20-1*07', 'TRBV20/OR9-2*01', 'TRBV20/OR9-2*02', 'TRBV20/OR9-2*03', 'TRBV21-1*01', 'TRBV21-1*02',
                  'TRBV21/OR9-2*01', 'TRBV23-1*01', 'TRBV23/OR9-2*01', 'TRBV23/OR9-2*02', 'TRBV24-1*01', 'TRBV24-1*02',
                  'TRBV24/OR9-2*01', 'TRBV25-1*01', 'TRBV26*01', 'TRBV26/OR9-2*01', 'TRBV26/OR9-2*02', 'TRBV27*01',
                  'TRBV28*01', 'TRBV29-1*01', 'TRBV29-1*02', 'TRBV29-1*03', 'TRBV29/OR9-2*01', 'TRBV29/OR9-2*02',
                  'TRBV3-1*01', 'TRBV3-1*02', 'TRBV3-2*01', 'TRBV3-2*02', 'TRBV3-2*03', 'TRBV30*01', 'TRBV30*02',
                  'TRBV30*03', 'TRBV30*04', 'TRBV30*05', 'TRBV4-1*01', 'TRBV4-1*02', 'TRBV4-2*01', 'TRBV4-2*02',
                  'TRBV4-3*01', 'TRBV4-3*02', 'TRBV4-3*03', 'TRBV4-3*04', 'TRBV5-1*01', 'TRBV5-1*02', 'TRBV5-3*01',
                  'TRBV5-3*02', 'TRBV5-4*01', 'TRBV5-4*02', 'TRBV5-4*03', 'TRBV5-4*04', 'TRBV5-5*01', 'TRBV5-5*02',
                  'TRBV5-5*03', 'TRBV5-6*01', 'TRBV5-7*01', 'TRBV5-8*01', 'TRBV5-8*02', 'TRBV6-1*01', 'TRBV6-2*01',
                  'TRBV6-3*01', 'TRBV6-4*01', 'TRBV6-4*02', 'TRBV6-5*01', 'TRBV6-6*01', 'TRBV6-6*02', 'TRBV6-6*03',
                  'TRBV6-6*04', 'TRBV6-6*05', 'TRBV6-7*01', 'TRBV6-8*01', 'TRBV6-9*01', 'TRBV7-1*01', 'TRBV7-2*01',
                  'TRBV7-2*02', 'TRBV7-2*03', 'TRBV7-2*04', 'TRBV7-3*01', 'TRBV7-3*02', 'TRBV7-3*03', 'TRBV7-3*04',
                  'TRBV7-3*05', 'TRBV7-4*01', 'TRBV7-6*01', 'TRBV7-6*02', 'TRBV7-7*01', 'TRBV7-7*02', 'TRBV7-8*01',
                  'TRBV7-8*02', 'TRBV7-8*03', 'TRBV7-9*01', 'TRBV7-9*02', 'TRBV7-9*03', 'TRBV7-9*04', 'TRBV7-9*05',
                  'TRBV7-9*06', 'TRBV7-9*07', 'TRBV9*01', 'TRBV9*02', 'TRBV9*03']
    _j_regions = ['TRBJ1-1*01', 'TRBJ1-2*01', 'TRBJ1-3*01', 'TRBJ1-4*01', 'TRBJ1-5*01', 'TRBJ1-6*01', 'TRBJ1-6*02',
                  'TRBJ2-1*01', 'TRBJ2-2*01', 'TRBJ2-2P*01', 'TRBJ2-3*01', 'TRBJ2-4*01', 'TRBJ2-5*01', 'TRBJ2-6*01',
                  'TRBJ2-7*01', 'TRBJ2-7*02']

    @property
    def version(self) -> str:
        """
        The version of the Method
        """
        return self.__version

    @property
    def command(self) -> str:
        """
        Defines the commandline call for external tool
        """
        return self.__command

    @property
    def name(self) -> str:
        """The name of the predictor"""
        return self.__name

    @property
    def supportedPeptides(self) -> list:
        """
        A list of valid Peptides
        """
        return []

    def parse_external_result(self, file: str, df: pd.DataFrame):
        """
        Parses external results and returns the result
        :param str file: The file path or the external prediction results
        :param pd.DataFrame df: the complete processed dataframe
        :return: A dictionary containing the prediction results
        :rtype: dict{(str, str, str, str): float} {(Receptor_ID, TRA, TRB, Peptide): score}
        """
        result = df.loc[:, ["Receptor_ID", "TRA", "TRB", "Peptide"]]
        mask = (df["TRBV"].isin(self._v_regions)) & (df["TRBJ"].isin(self._j_regions))
        result.loc[:, "Score"] = -1
        if df.shape[0] == 1:
            result["Score"] = np.load((file + ".npy"))[0][0]
        else:
            result.loc[mask, "Score"] = np.load((file + ".npy"))[0]
        result["Score"].astype(float)
        result.fillna("", inplace=True)
        result["Receptor_ID"].astype(str)
        return {self.name: {
                            row[:4]: float("{:.4f}".format(row[4]))
                            for row in result.itertuples(index=False)
                           }
               }

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
        return None

    def get_TRB_full_seq(self, df: pd.DataFrame,
                         directory: str = "/home/mahmoud/Documents/BA/LetzerVersuch/epytope/data",
                         v_seg_header: str = "TRBV",
                         j_seg_header: str = "TRBJ",
                         cdr3_header: str = "TRB",
                         titan_interpreter: str = "python",
                         local_titan_repository: str = "/home/mahmoud/Documents/BA/TITAN/TITAN") \
            -> Tuple[pd.DataFrame, pd.core.series.Series]:
        """
        get the full CDR3 sequence give the CDR3 sequence and its corresponding V, J segments
        :param directory: Directory containing V_segment_sequences.fasta and J_segment_sequences.fasta files downloaded from
         IMGT (https://imgt.org/vquest/refseqh.html#refdir)
        :param df: a dataframe containing CDR3 info, V and J segment names
        :param v_seg_header: a string representing the header for column containing V segments
        :param j_seg_header: a string representing the header for column containing J segments
        :param cdr3_header: a string representing the header for column containing CDR3 beta sequence
        :param titan_interpreter: a string representing a path to a python interpreter under the virtual environment of
        TITAN
        :param local_titan_repository: a string representing a path to a local TITAN's repository
        :return: (pd.DataFrame, pd.core.series.Series), where the dataframe has all samples, which their v- and j-regions
        are includes in the human v- and in j-regions given by IMGT. The function returns additionally series, which
        holds true values for the accepted samples, otherwise false values.
        :rtype pd.DataFrame, pd.core.series.Series
        """
        output = os.path.join(directory, f"full_{cdr3_header}_seq.csv")
        input_ = os.path.join(directory, "input.csv")
        # reformat V- and J-region to match the IMGT gene and allele name
        df[v_seg_header] = df[v_seg_header].apply(lambda x: x if re.search(r"\*\d+$", x) else x + "*01")
        df[j_seg_header] = df[j_seg_header].apply(lambda x: x if re.search(r"\*\d+$", x) else x + "*01")
        # exclude all samples, their v- or j-regions of the beta-sequences are not included in human v- or j-regions
        # given by IMGT
        mask = (df["TRBV"].isin(self._v_regions)) & (df["TRBJ"].isin(self._j_regions))
        accepted_samples = df.loc[mask, ]
        excluded_samples = df.loc[~mask, ]
        accepted_samples.to_csv(input_, index=False)
        cmd = f"{titan_interpreter} {os.path.join(local_titan_repository, 'scripts/cdr3_to_full_seq.py')} {directory} " \
              f"{input_} {v_seg_header} {j_seg_header} {cdr3_header} {output}"
        try:
            p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
            stdo, stde = p.communicate()
            stdr = p.returncode
            if stdr > 0:
                raise RuntimeError("Unsuccessful execution of " + cmd + " (EXIT!=0) with output:\n" + stdo.decode())
            if os.path.getsize(output) == 0:
                raise RuntimeError(
                    "Unsuccessful execution of " + cmd + " (empty output file) with output:\n" +
                    stdo.decode())
        except Exception as e:
            raise RuntimeError(e)
        os.remove(input_)
        df_full_seq = pd.read_csv(output).iloc[:, 1:]
        os.remove(output)
        return df_full_seq, mask

    def predict(self, peptides, TCRs, repository: str, all: bool, **kwargs):
        """
        Overwrites ATCRSpecificityPrediction.predict

        Predicts binding probability between a T-cell receptor CDR3 protein sequence and a peptide
        :param peptides: The TCREpitope objects for which predictions should be performed
        :type peptides: :class:`~epytope.Core.TCREpitope.TCREpitope` or list(:class:`~epytope.Core.TCREpitope.TCREpitope`)
        :param TCRs: T cell receptor objects
        :type  :class:'~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor' or
        list(:class:'~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor')
        :param str repository: a path to a local github repository of TITAN predictor
        :param bool all: if true each TCR object will be joined with each peptide to perform the prediction, otherwise
        the prediction will be preformed in the same order of the passed peptides and TCRs objects
        :param str trained_on: specifying the dataset the model trained on. This parameter is specific for ERGO, which
        has two models, one is trained on vdjdb and the other is trained on McPAS dataset.
        :param trained_model a string representing a path to the trained model directory. For ERGO this parameter will
        be ignored
        :param down: a boolean value for choosing from two different models of ImRex. If it is set to Ture, the model,
        trained of down sampled dataset of vdjdb will be selected, otherwise the other trained model will be chosen.
        Default value is False.
        :param nettcr_chain: a string specifying the chain(s) to use (a, b, ab). Default: b.
        :param pMTnet_interpreter: a string representing a path to python interpreter for pMTnet.
        :param chain: a string representing the type of the used cdr3-sequence in the prediction. It can be set to 'a'
        for alpha- or 'b' for beta-sequences. This parameter specifics the input chain for ATM-TCR.
        :param padding str: can be set to one of the following values ['front, end, mid, alignment'] and specifics the
        padding type for ATM-TCR.
        :param cuda bool: it can be set to True, if the running device has cuda.
        :return: A :class:`~epytope.Core.TCRSpecificityPredictionResult` object
        :rtype: :class:`~epytope.Core.TCRSpecificityPredictionResult`
        """
        df = super().predict(peptides=peptides, TCRs=TCRs, repository=repository, all=all)
        if "trained_model" in kwargs:
            if not os.path.exists(kwargs["trained_model"]) or not os.path.isdir(kwargs["trained_model"]):
                raise NotADirectoryError("please pass a path to a directory contains parameters for the trained model "
                                         "of TITAN, which can be downloaded for "
                                         "'https://ibm.ent.box.com/v/titan-dataset'. The folder, that contains the "
                                         "trained model called 'trained_model', contains the weights and parameters "
                                         "for the trained model of TITAN.")
            else:
                trained_model = kwargs["trained_model"]

        df_result = self.predict_from_dataset(repository=repository, df=df, score=-1, trained_model=trained_model)
        return df_result

    def prepare_dataset_TITAN(self, df: pd.DataFrame,
                              directory: str = "/home/mahmoud/PycharmProjects/Benchmark/data/TITAN/Running") -> str:
        """
        process the dataset in the way to be suitable with TITAN's input
        :param directory: a string representing a path to a directory, where the precessed files will be saved
        :param df: a dataframe contains TRB seqs and corresponding epitopes to predict, if they bind or not
        :return: returns the directory parameter, where one can find all processed files, used for prediction
        :rtype: str
        """
        # select only the required feature to run TITAN https://github.com/PaccMann/TITAN
        test_set = df[["full_seq", "Peptide"]]
        uni_epitopes = list(df.loc[:, "Peptide"].unique())
        # Convert an amino acid sequence (IUPAC) into SMILES.
        uni_epitopes_smi = [aas_to_smiles(pep) for pep in uni_epitopes]
        peptide_ID = dict(zip(uni_epitopes, [i for i in range(len(uni_epitopes))]))
        epitopes_smi = pd.DataFrame({"Peptide": uni_epitopes_smi, "Peptide_ID": peptide_ID.values()})
        epitopes_smi.to_csv(os.path.join(directory, "epitopes.smi"), header=False, sep="\t", index=False)
        uni_tcrs = list(df.loc[:, "full_seq"].unique())
        tcr_ID = dict(zip(uni_tcrs, [i for i in range(len(uni_tcrs))]))
        tcrs = pd.DataFrame({"full_seq": uni_tcrs, "TRB_ID": tcr_ID.values()})
        tcrs.to_csv(os.path.join(directory, "tcrs.csv"), header=False, index=False, sep="\t")
        ligand_name = df.loc[:, "Peptide"].map(lambda x: peptide_ID[x])
        sequence_id = df.loc[:, "full_seq"].map(lambda x: tcr_ID[x])
        label = [1 for _ in range(df.shape[0])]
        test_dataset = pd.DataFrame({"ligand_name": ligand_name, "sequence_id": sequence_id, "label": label})
        test_dataset.to_csv(os.path.join(directory, "input.csv"))
        return directory

    def predict_from_dataset(self, repository: str, path: str = None, df: pd.DataFrame = None, source: str = None,
                             score: int = 1, **kwargs):
        """
        Predicts binding probability between a T-cell receptor CDR3 protein sequence and a peptide.
        The path should lead to csv file with fixed column names dataset.columns = ['TRA', 'TRB', "TRAV", "TRAJ",
        "TRBV", "TRBJ", "T-Cell-Type", "Peptide", "MHC", "Species", "Antigen.species", "Tissue"]. If some values for
        one or more variables are unavailable, leave them as blank cells.
        :param str repository: a path to a local github repository of TITAN predictor
        :param str path: a string representing a path to the dataset(csv file), which will be processed. Default value
        is None, when the dataframe object is given
        :param `pd.DataFrame` df: a dataframe object. Default value is None, when the path is given
        :param str source: the source of the dataset [vdjdb, mcpas, scirpy, IEDB]. If this parameter is not passed,
         the dataset should be a csv file with the column names mentioned above
        :param int score: An integer representing a confidence score between 0 and 3 (0: critical information missing,
        1: medium confidence, 2: high confidence, 3: very high confidence). By processing all entries with a confidence
        score >= the passed parameter score will be kept. Default value is 1
        :param str trained_on: specifying the dataset the model trained on. This parameter is specific for ERGO, which
        has two models, one is trained on vdjdb and the other is trained on McPAS dataset.
        :param trained_model a string representing a path to the trained model directory. For ERGO this parameter will
        be ignored
        :param down: a boolean value for choosing from two different models of ImRex. If it is set to Ture, the model,
        trained of down sampled dataset of vdjdb will be selected, otherwise the other trained model will be chosen.
        Default value is False.
        :param nettcr_chain: a string specifying the chain(s) to use (a, b, ab). Default: b.
        :param pMTnet_interpreter: a string representing a path to python interpreter for pMTnet.
        :param chain: a string representing the type of the used cdr3-sequence in the prediction. It can be set to 'a'
        for alpha- or 'b' for beta-sequences. This parameter specifics the input chain for ATM-TCR.
        :param padding str: can be set to one of the following values ['front, end, mid, alignment'] and specifics the
        padding type for ATM-TCR.
        :param cuda bool: it can be set to True, if the running device has cuda.
        :return: A :class:`~epytope.Core.TCRSpecificityPredictionResult` object
        :rtype: :class:`~epytope.Core.TCRSpecificityPredictionResult`
        """

        if path is None and df is None:
            raise FileNotFoundError("A path to a csv file or a dataframe should be passed")
        if df is None:
            if os.path.isfile(path):
                df = process_dataset_TCR(path=path, source=source, score=score)
            else:
                raise FileNotFoundError("A path to a csv file or a dataframe should be passed")
        else:
            df = process_dataset_TCR(df=df, source=source, score=score)
        if "trained_model" in kwargs:
            if not os.path.exists(kwargs["trained_model"]) or not os.path.isdir(kwargs["trained_model"]):
                raise NotADirectoryError("please pass a path to a directory contains parameters for the trained model "
                                         "of TITAN, which can be downloaded for "
                                         "'https://ibm.ent.box.com/v/titan-dataset'. The folder, that contains the "
                                         "trained model called 'trained_model', contains the weights and parameters "
                                         "for the trained model of TITAN.")
            else:
                trained_model = kwargs["trained_model"]
        if not os.path.isdir(repository):
            raise NotADirectoryError("please pass a path as a string to a local TITAN repository. To clone the "
                                     "repository type: 'git clone https://github.com/PaccMann/TITAN.git' in the "
                                     "terminal")
        # get the full TCR-sequence depending on the corresponding v- and j-region
        directory = os.path.join(Path(__file__).parent, "data")
        df_titan, mask = self.get_TRB_full_seq(df, directory=directory, local_titan_repository=repository)
        # remove non aa characters from full cdr3 seqs
        df_titan.loc[:, "full_seq"] = df_titan["full_seq"].map(lambda x: x.replace("*", ""))
        tmp_dir = tempfile.TemporaryDirectory()
        # TITAN throws an Exception, if the dataset has only one sample. To deal with this issue the dataset will be
        # duplicated, but the end result contains only one sample
        if len(df_titan) == 1:
            df_titan = pd.concat([df_titan, df_titan])
        self.prepare_dataset_TITAN(df=df_titan, directory=tmp_dir.name)
        epitopes = os.path.join(tmp_dir.name, "epitopes.smi")
        tcrs = os.path.join(tmp_dir.name, "tcrs.csv")
        test_set = os.path.join(tmp_dir.name, "input.csv")
        scores = os.path.join(tmp_dir.name, "scores")
        try:
            cmd = f"python {os.path.join(repository, 'scripts/flexible_model_eval.py')} {test_set} {tcrs} {epitopes} " \
                  f"{trained_model} bimodal_mca {scores}"
            p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
            stdo, stde = p.communicate()
            stdr = p.returncode
            if stdr > 0:
                raise RuntimeError("Unsuccessful execution of " + cmd + " (EXIT!=0) with output:\n" + stdo.decode())
            if os.path.getsize((scores + ".npy")) == 0:
                raise RuntimeError(
                    "Unsuccessful execution of " + cmd + " (empty output file) with output:\n" +
                    stdo.decode())
        except Exception as e:
            raise RuntimeError(e)
        result = self.parse_external_result(file=scores, df=df)
        tmp_dir.cleanup()
        if sum(mask) < df.shape[0]:
            print(f"TITAN's trained model can not make predictions for those samples, which their v- or "
                  f"j-regions are not included in the human v- or j-regions given by IMGT. Therefore the prediction "
                  f"score for these samples will be -1.")
        df_result = TCRSpecificityPredictionResult.from_dict(result)
        df_result.index = pd.MultiIndex.from_tuples([tuple((ID, TRA, TRB, pep)) for ID, TRA, TRB, pep in df_result.index],
                                                        names=["Receptor_ID", 'TRA', 'TRB', "Peptide"])
        return df_result


class ImRex(AExternalTCRSpecificityPrediction, AExternal):
    """
    Implements ImRex (Interaction Map Recognition). "https://github.com/pmoris/ImRex"

    """
    __name = "ImRex"
    __command = "python src/scripts/predict/predict.py"
    __version = " "

    @property
    def version(self) -> str:
        """
        The version of the Method
        """
        return self.__version

    @property
    def command(self) -> str:
        """
        Defines the commandline call for external tool
        """
        return self.__command

    @property
    def name(self) -> str:
        """The name of the predictor"""
        return self.__name

    @property
    def supportedPeptides(self) -> list:
        """
        A list of valid Peptides
        """
        return []

    def parse_external_result(self, file: str, df: pd.DataFrame):
        """
        Parses external results and returns the result
        :param str file: The file path or the external prediction results
        :param pd.DataFrame df: the complete processed dataframe
        :return: A dictionary containing the prediction results
        :rtype: dict{(str, str, str, str): float} {(Receptor_ID, TRA, TRB, Peptide): score}
        """
        outcome = pd.read_csv(file)[["cdr3", "antigen.epitope", "prediction_score"]]
        outcome.rename(columns={"cdr3":"TRB", "antigen.epitope": "Peptide", "prediction_score": "Score"}, inplace=True)
        result = df.loc[:, ["Receptor_ID", "TRA", "TRB", "Peptide"]]
        result = result.merge(outcome, on=["TRB", "Peptide"], how="left")
        result.loc[:, "Score"] = result["Score"].map(lambda x: -1 if pd.isna(x) else x)
        result["Score"].astype(float)
        result.fillna("", inplace=True)
        result["Receptor_ID"].astype(str)
        return {self.name: {
                            row[:4]: float("{:.4f}".format(row[4]))
                            for row in result.itertuples(index=False)
                           }
               }

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
        return None

    def predict(self, peptides, TCRs, repository: str, all: bool, **kwargs):
        """
        Overwrites ATCRSpecificityPrediction.predict

        Predicts binding probability between a T-cell receptor CDR3 protein sequence and a peptide
        :param peptides: The TCREpitope objects for which predictions should be performed
        :type peptides: :class:`~epytope.Core.TCREpitope.TCREpitope` or list(:class:`~epytope.Core.TCREpitope.TCREpitope`)
        :param TCRs: T cell receptor objects
        :type  :class:'~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor' or
        list(:class:'~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor')
        :param str repository: a path to a local github repository of ImRex predictor
        :param bool all: if true each TCR object will be joined with each peptide to perform the prediction, otherwise
        the prediction will be preformed in the same order of the passed peptides and TCRs objects
        :param str trained_on: specifying the dataset the model trained on. This parameter is specific for ERGO, which
        has two models, one is trained on vdjdb and the other is trained on McPAS dataset.
        :param trained_model a string representing a path to the trained model directory. For ERGO this parameter will
        be ignored
        :param down: a boolean value for choosing from two different models of ImRex. If it is set to Ture, the model,
        trained of down sampled dataset of vdjdb will be selected, otherwise the other trained model will be chosen.
        Default value is False.
        :param nettcr_chain: a string specifying the chain(s) to use (a, b, ab). Default: b.
        :param pMTnet_interpreter: a string representing a path to python interpreter for pMTnet.
        :param chain: a string representing the type of the used cdr3-sequence in the prediction. It can be set to 'a'
        for alpha- or 'b' for beta-sequences. This parameter specifics the input chain for ATM-TCR.
        :param padding str: can be set to one of the following values ['front, end, mid, alignment'] and specifics the
        padding type for ATM-TCR.
        :param cuda bool: it can be set to True, if the running device has cuda.
        :return: A :class:`~epytope.Core.TCRSpecificityPredictionResult` object
        :rtype: :class:`~epytope.Core.TCRSpecificityPredictionResult`
        """
        df = super().predict(peptides=peptides, TCRs=TCRs, repository=repository, all=all)
        if "down" in kwargs:
            down = kwargs["down"]
        else:
            down = False
        df_result = self.predict_from_dataset(repository=repository, df=df, score=-1, down=down)
        return df_result

    def prepare_dataset_IMRex(self, df: pd.DataFrame, filename: str = None) -> \
            Tuple[pd.DataFrame, pd.core.series.Series]:
        """
        process the dataset in the way to be suitable with ImRex
        :param df: a dataframe contains at least TRB seqs and corresponding epitopes to predict, if they bind or not
        :param filename: str representing the name of the file to save the processed dataset
        :return: (pd.DataFrame, pd.core.series.Series), where the dataframe has all samples, which have cdr3-beta-seqs,
        that are 10-20 aas long, and epitopes, which are 8-11 aas long. The function returns additionally series, which
        holds true values for the accepted samples, otherwise false values.
        :rtype pd.DataFrame, pd.core.series.Series
        """
        # accept only cdr3ß-sequences, which are 10-20 aas long and only epitopes, which are 8-11 aas long
        mask = (df["TRB"].str.len() >= 10) & (df["TRB"].str.len() <= 20) & (df["Peptide"].str.len() >= 8) & (
                    df["Peptide"].str.len() <= 11)
        test_set = df.loc[mask, ]
        # select only the required feature to run IMRex
        test_set = test_set.loc[:, ["TRB", "Peptide"]]
        # remove all observation, in which no cdr3 beta seq or peptide seq doesn't occur
        test_set = test_set.loc[(test_set["TRB"] != "") & (test_set["Peptide"] != ""), ]
        # change column names according to https://github.com/pmoris/ImRex#predictions-using-the-pre-built-model
        test_set.columns = ['cdr3', 'antigen.epitope']
        # remove all entries, where cdr3 beta does not present
        test_set.dropna(inplace=True)
        # remove duplicates
        # test_set = test_set.drop_duplicates(subset=["cdr3", "antigen.epitope"], keep='first').reset_index(drop=True)
        if filename:
            test_set.to_csv(filename, sep=";", index=False)
        return test_set, mask

    def predict_from_dataset(self, repository: str, path: str = None, df: pd.DataFrame = None, source: str = None,
                             score: int = 1, **kwargs):
        """
        Predicts binding probability between a T-cell receptor CDR3 protein sequence and a peptide.
        The path should lead to csv file with fixed column names dataset.columns = ['TRA', 'TRB', "TRAV", "TRAJ",
        "TRBV", "TRBJ", "T-Cell-Type", "Peptide", "MHC", "Species", "Antigen.species", "Tissue"]. If some values for
        one or more variables are unavailable, leave them as blank cells.
        :param str repository: a path to a local github repository of ImRex predictor
        :param str path: a string representing a path to the dataset(csv file), which will be processed. Default value
        is None, when the dataframe object is given
        :param `pd.DataFrame` df: a dataframe object. Default value is None, when the path is given
        :param str source: the source of the dataset [vdjdb, mcpas, scirpy, IEDB]. If this parameter is not passed,
         the dataset should be a csv file with the column names mentioned above
        :param int score: An integer representing a confidence score between 0 and 3 (0: critical information missing,
        1: medium confidence, 2: high confidence, 3: very high confidence). By processing all entries with a confidence
        score >= the passed parameter score will be kept. Default value is 1
        :param str trained_on: specifying the dataset the model trained on. This parameter is specific for ERGO, which
        has two models, one is trained on vdjdb and the other is trained on McPAS dataset.
        :param trained_model a string representing a path to the trained model directory. For ERGO this parameter will
        be ignored
        :param down: a boolean value for choosing from two different models of ImRex. If it is set to Ture, the model,
        trained of down sampled dataset of vdjdb will be selected, otherwise the other trained model will be chosen.
        Default value is False.
        :param nettcr_chain: a string specifying the chain(s) to use (a, b, ab). Default: b.
        :param pMTnet_interpreter: a string representing a path to python interpreter for pMTnet.
        :param chain: a string representing the type of the used cdr3-sequence in the prediction. It can be set to 'a'
        for alpha- or 'b' for beta-sequences. This parameter specifics the input chain for ATM-TCR.
        :param padding str: can be set to one of the following values ['front, end, mid, alignment'] and specifics the
        padding type for ATM-TCR.
        :param cuda bool: it can be set to True, if the running device has cuda.
        :return: A :class:`~epytope.Core.TCRSpecificityPredictionResult` object
        :rtype: :class:`~epytope.Core.TCRSpecificityPredictionResult`
        """

        if path is None and df is None:
            raise FileNotFoundError("A path to a csv file or a dataframe should be passed")
        if df is None:
            if os.path.isfile(path):
                df = process_dataset_TCR(path=path, source=source, score=score)
            else:
                raise FileNotFoundError("A path to a csv file or a dataframe should be passed")
        else:
            df = process_dataset_TCR(df=df, source=source, score=score)
        if not os.path.isdir(repository):
            raise NotADirectoryError("please pass a path as a string to a local ImRex repository. To clone the "
                                     "repository type: 'git clone https://github.com/pmoris/ImRex.git' in the terminal")
        # prepare the test set to be suitable with ImRex
        tmp_file = NamedTemporaryFile(delete=False)
        test_set, mask = self.prepare_dataset_IMRex(df=df, filename=tmp_file.name)
        if "down" in kwargs:
            down = kwargs["down"]
        else:
            down = False
        if down:
            model = os.path.join(repository, 'models/pretrained/2020-07-24_19-18-39_trbmhcidown-shuffle-padded-b32-lre4'
                                             '-reg001/2020-07-24_19-18-39_trbmhcidown-shuffle-padded-b32-lre4-reg001.h5')
        else:
            model = os.path.join(repository, 'models/pretrained/2020-07-30_11-30-27_trbmhci-shuffle-padded-b32-lre4-'
                                             'reg001/2020-07-30_11-30-27_trbmhci-shuffle-padded-b32-lre4-reg001.h5')
        script = os.path.join(repository, "src/scripts/predict/predict.py")
        tmp_out = NamedTemporaryFile(delete=False)
        cmd = f"python {script} --model {model} --input {tmp_file.name} --output {tmp_out.name}"
        try:
            p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
            stdo, stde = p.communicate()
            stdr = p.returncode
            if stdr > 0:
                raise RuntimeError("Unsuccessful execution of " + cmd + " (EXIT!=0) with output:\n" + stdo.decode())
            if os.path.getsize(tmp_out.name) == 0:
                raise RuntimeError(
                    "Unsuccessful execution of " + cmd + " (empty output file) with output:\n" +
                    stdo.decode())
        except Exception as e:
            raise RuntimeError(e)
        os.remove(tmp_file.name)
        result = self.parse_external_result(file=tmp_out.name, df=df)
        tmp_out.close()
        os.remove(tmp_out.name)
        df_result = TCRSpecificityPredictionResult.from_dict(result)
        df_result.index = pd.MultiIndex.from_tuples(
            [tuple((ID, TRA, TRB, pep)) for ID, TRA, TRB, pep in df_result.index],
            names=["Receptor_ID", 'TRA', 'TRB', "Peptide"])
        if sum(mask) < df.shape[0]:
            print(f"ImRex's trained model could not make predictions for some samples, which have either "
                  f"cdr3-beta-seqs, that are not 10-20 aas long or epitopes, that are not 8-11 aas long. These samples "
                  f"have prediction score of -1")
        return df_result


class NetTCR(AExternalTCRSpecificityPrediction, AExternal):
    """
    Implements NetTCR-2.0. "https://github.com/mnielLab/NetTCR-2.0"

    """
    __name = "NetTCR2"
    __command = "python nettcr.py"
    __version = "2.0"

    @property
    def version(self) -> str:
        """
        The version of the Method
        """
        return self.__version

    @property
    def command(self) -> str:
        """
        Defines the commandline call for external tool
        """
        return self.__command

    @property
    def name(self) -> str:
        """The name of the predictor"""
        return self.__name

    @property
    def supportedPeptides(self) -> list:
        """
        A list of valid Peptides
        """
        return []

    def parse_external_result(self, file: str, df: pd.DataFrame):
        """
        Parses external results and returns the result
        :param str file: The file path or the external prediction results
        :param pd.DataFrame df: the complete processed dataframe
        :return: A dictionary containing the prediction results
        :rtype: dict{(str, str, str, str): float} {(Receptor_ID, TRA, TRB, Peptide): score}
        """
        outcome = pd.read_csv(file)
        outcome.rename(columns={'prediction': 'Score', "CDR3b": "TRB", "peptide": "Peptide", "CDR3a": "TRA"},
                       inplace=True)
        result = df.loc[:, ["Receptor_ID", "TRA", "TRB", "Peptide"]]
        result = result.merge(outcome, on=["TRA", "TRB", "Peptide"], how="left")
        result.loc[:, "Score"] = result["Score"].map(lambda x: -1 if pd.isna(x) else x)
        result["Score"].astype(float)
        result.fillna("", inplace=True)
        result["Receptor_ID"].astype(str)
        return {self.name: {
                            row[:4]: float("{:.4f}".format(row[4]))
                            for row in result.itertuples(index=False)
                           }
               }

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
        return None

    def predict(self, peptides, TCRs, repository: str, all: bool, **kwargs):
        """
        Overwrites ATCRSpecificityPrediction.predict

        Predicts binding probability between a T-cell receptor CDR3 protein sequence and a peptide
        :param peptides: The TCREpitope objects for which predictions should be performed
        :type peptides: :class:`~epytope.Core.TCREpitope.TCREpitope` or list(:class:`~epytope.Core.TCREpitope.TCREpitope`)
        :param TCRs: T cell receptor objects
        :type  :class:'~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor' or
        list(:class:'~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor')
        :param str repository: a path to a local github repository of NetTCR predictor
        :param bool all: if true each TCR object will be joined with each peptide to perform the prediction, otherwise
        the prediction will be preformed in the same order of the passed peptides and TCRs objects
        :param str trained_on: specifying the dataset the model trained on. This parameter is specific for ERGO, which
        has two models, one is trained on vdjdb and the other is trained on McPAS dataset.
        :param trained_model a string representing a path to the trained model directory. For ERGO this parameter will
        be ignored
        :param down: a boolean value for choosing from two different models of ImRex. If it is set to Ture, the model,
        trained of down sampled dataset of vdjdb will be selected, otherwise the other trained model will be chosen.
        Default value is False.
        :param nettcr_chain: a string specifying the chain(s) to use (a, b, ab). Default: b.
        :param pMTnet_interpreter: a string representing a path to python interpreter for pMTnet.
        :param chain: a string representing the type of the used cdr3-sequence in the prediction. It can be set to 'a'
        for alpha- or 'b' for beta-sequences. This parameter specifics the input chain for ATM-TCR.
        :param padding str: can be set to one of the following values ['front, end, mid, alignment'] and specifics the
        padding type for ATM-TCR.
        :param cuda bool: it can be set to True, if the running device has cuda.
        :return: A :class:`~epytope.Core.TCRSpecificityPredictionResult` object
        :rtype: :class:`~epytope.Core.TCRSpecificityPredictionResult`
        """
        df = super().predict(peptides=peptides, TCRs=TCRs, repository=repository, all=all)
        if "nettcr_chain" in kwargs:
            nettcr_chain = kwargs["nettcr_chain"]
            if nettcr_chain not in ["a", "b", "ab"]:
                raise ValueError(f"nettcr_chain argument can only be set to one of the following values "
                                 f"['a', 'b', 'ab']")
        else:
            nettcr_chain = "b"
        df_result = self.predict_from_dataset(repository=repository, df=df, score=-1, nettcr_chain=nettcr_chain)
        return df_result

    def prepare_dataset_NetTCR(self, df: pd.DataFrame, filename: str = None, nettcr_chain: str = "b") -> \
            Tuple[pd.DataFrame, pd.core.series.Series]:
        """
        process the dataset in the way to be suitable with NetTCR
        :param df: a dataframe contains TRA, TRB seqs and corresponding epitopes to predict, if they bind or not
        :param filename: str representing the name of the file to save the processed dataset
        :param nettcr_chain: a string specifying the chain(s) to use (a, b, ab). Default: b.
        :return: (pd.DataFrame, pd.core.series.Series), where the dataframe has all samples, which have
        cdr3-(beta, alpha)-seqs shorter than 31 aas, and epitopes, which are shorter than 9 aas.
        The function returns additionally series, which holds true values for the accepted samples, otherwise false
        values.
        :rtype pd.DataFrame, pd.core.series.Series
        """

        def is_aa_seq(seq):
            seq = str(seq)
            if len(seq) == 0:
                return False
            aas = set("ARNDCEQGHILKMFPSTWYV")
            return all([i in aas for i in seq])

        def replace(seq):
            if not is_aa_seq(seq):
                return ""
            else:
                return str(seq)
        # NetTCR accepts only cdr3-alpha and cdr3-beta sequences, that are shorter than 30 aas and peptides, that are
        # shorter than 9 aas. remove all observation, in which no cdr3 beta seq or peptide seq doesn't occur, or if it
        # is not an amino acid sequence.
        mask = (df["TRB"].str.len() <= 30) & (df["Peptide"].str.len() <= 9) & (df["TRA"].str.len() <= 30) & \
               (df["TRB"].apply(is_aa_seq)) & (df["Peptide"].apply(is_aa_seq))
        if nettcr_chain in ["a", "ab"]:
            mask = mask & (df["TRA"].apply(is_aa_seq))
        test_set = df.loc[mask, ]
        # select only the required feature to run NetTCR
        test_set = test_set[["TRA", "TRB", "Peptide"]]
        """
        # remove all observation, in which no cdr3 beta seq or peptide seq doesn't occur
        test_set = test_set.loc[(test_set["TRB"].apply(is_aa_seq)) & (test_set["Peptide"].apply(is_aa_seq))]
        test_set["TRA"] = test_set["TRA"].apply(replace)
        """
        # remove duplicates
        # test_set.drop_duplicates(subset=["TRB", "Peptide"], keep='first', inplace=True, ignore_index=True)
        # change column names according to https://github.com/mnielLab/NetTCR-2.0
        test_set.columns = ["CDR3a", "CDR3b", "peptide"]

        if filename:
            test_set.to_csv(filename, index=False)
        return test_set, mask

    def predict_from_dataset(self, repository: str, path: str = None, df: pd.DataFrame = None, source: str = None,
                             score: int = 1, **kwargs):
        """
        Predicts binding probability between a T-cell receptor CDR3 protein sequence and a peptide.
        The path should lead to csv file with fixed column names dataset.columns = ['TRA', 'TRB', "TRAV", "TRAJ",
        "TRBV", "TRBJ", "T-Cell-Type", "Peptide", "MHC", "Species", "Antigen.species", "Tissue"]. If some values for
        one or more variables are unavailable, leave them as blank cells.
        :param str repository: a path to a local github repository of NetTCR predictor
        :param str path: a string representing a path to the dataset(csv file), which will be processed. Default value
        is None, when the dataframe object is given
        :param `pd.DataFrame` df: a dataframe object. Default value is None, when the path is given
        :param str source: the source of the dataset [vdjdb, mcpas, scirpy, IEDB]. If this parameter is not passed,
         the dataset should be a csv file with the column names mentioned above
        :param int score: An integer representing a confidence score between 0 and 3 (0: critical information missing,
        1: medium confidence, 2: high confidence, 3: very high confidence). By processing all entries with a confidence
        score >= the passed parameter score will be kept. Default value is 1
        :param str trained_on: specifying the dataset the model trained on. This parameter is specific for ERGO, which
        has two models, one is trained on vdjdb and the other is trained on McPAS dataset.
        :param trained_model a string representing a path to the trained model directory. For ERGO this parameter will
        be ignored
        :param down: a boolean value for choosing from two different models of ImRex. If it is set to Ture, the model,
        trained of down sampled dataset of vdjdb will be selected, otherwise the other trained model will be chosen.
        Default value is False.
        :param nettcr_chain: a string specifying the chain(s) to use (a, b, ab). Default: b.
        :param pMTnet_interpreter: a string representing a path to python interpreter for pMTnet.
        :param chain: a string representing the type of the used cdr3-sequence in the prediction. It can be set to 'a'
        for alpha- or 'b' for beta-sequences. This parameter specifics the input chain for ATM-TCR.
        :param padding str: can be set to one of the following values ['front, end, mid, alignment'] and specifics the
        padding type for ATM-TCR.
        :param cuda bool: it can be set to True, if the running device has cuda.
        :return: A :class:`~epytope.Core.TCRSpecificityPredictionResult` object
        :rtype: :class:`~epytope.Core.TCRSpecificityPredictionResult`
        """

        if path is None and df is None:
            raise FileNotFoundError("A path to a csv file or a dataframe should be passed")
        if df is None:
            if os.path.isfile(path):
                df = process_dataset_TCR(path=path, source=source, score=score)
            else:
                raise FileNotFoundError("A path to a csv file or a dataframe should be passed")
        else:
            df = process_dataset_TCR(df=df, source=source, score=score)
        if not os.path.isdir(repository):
            raise NotADirectoryError("please pass a path as a string to a local NetTCR2 repository. To clone the "
                                     "repository type: 'git clone https://github.com/mnielLab/NetTCR-2.0.git' in the "
                                     "terminal")
        if "nettcr_chain" in kwargs:
            nettcr_chain = kwargs["nettcr_chain"]
            if nettcr_chain not in ["a", "b", "ab"]:
                raise ValueError(f"nettcr_chain argument can only be set to one of the following values "
                                 f"['a', 'b', 'ab']")
        else:
            nettcr_chain = "b"
        df.reset_index(drop=True, inplace=True)
        # process dataframe in the way to be suitable with NetTCR's input
        tmp_file = NamedTemporaryFile(delete=False)
        _, mask = self.prepare_dataset_NetTCR(df, tmp_file.name, nettcr_chain=nettcr_chain)
        tmp_out = NamedTemporaryFile(delete=False)
        training_set = os.path.join(repository, "test", "sample_train.csv")
        script = os.path.join(repository, "nettcr.py")
        try:
            cmd = f"python {script} --trainfile {training_set} --testfile {tmp_file.name} --chain {nettcr_chain} " \
                  f"--outfile {tmp_out.name}"
            p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
            stdo, stde = p.communicate()
            stdr = p.returncode
            if stdr > 0:
                raise RuntimeError("Unsuccessful execution of " + cmd + " (EXIT!=0) with output:\n" + stdo.decode())
            if os.path.getsize(tmp_out.name) == 0:
                raise RuntimeError(
                    "Unsuccessful execution of " + cmd + " (empty output file) with output:\n" +
                    stdo.decode())
        except Exception as e:
            raise RuntimeError(e)
        os.remove(tmp_file.name)
        result = self.parse_external_result(file=tmp_out.name, df=df)
        tmp_out.close()
        os.remove(tmp_out.name)
        df_result = TCRSpecificityPredictionResult.from_dict(result)
        df_result.index = pd.MultiIndex.from_tuples(
            [tuple((ID, TRA, TRB, pep)) for ID, TRA, TRB, pep in df_result.index],
            names=["Receptor_ID", 'TRA', 'TRB', "Peptide"])
        if sum(mask) < df.shape[0]:
            print(f"NetTCR-2's trained model could not make predictions for some samples, which have either "
                  f"cdr3-(beta, alpha)-seqs, that are longer than 30 aas or epitopes, that are longer than 9 aas. These "
                  f"samples, have prediction score of -1")
        return df_result


class pMTnet(AExternalTCRSpecificityPrediction, AExternal):
    """
    Implements pMTnet. "https://github.com/tianshilu/pMTnet"

    """
    __name = "pMTnet"
    __command = "python main.py"
    __version = " "

    @property
    def version(self) -> str:
        """
        The version of the Method
        """
        return self.__version

    @property
    def command(self) -> str:
        """
        Defines the commandline call for external tool
        """
        return self.__command

    @property
    def name(self) -> str:
        """The name of the predictor"""
        return self.__name

    @property
    def supportedPeptides(self) -> list:
        """
        A list of valid Peptides
        """
        return []

    def parse_external_result(self, file: str, df: pd.DataFrame, output_log: str):
        """
        Parses external results and returns the result
        :param str file: The file path or the external prediction results
        :param pd.DataFrame df: the complete processed dataframe
        :param str output_log: log file with CDR, Antigen, HLA information
        :return: A dictionary containing the prediction results
        :rtype: dict{(str, str, str, str): float} {(Receptor_ID, TRA, TRB, Peptide): score}
        """
        outcome = pd.read_csv(file)
        #outcome.to_csv("/home/mahmoud/Documents/out.csv")
        outcome.columns = ["TRB", "Peptide", "MHC", "Score"]
        result = df.loc[:, ["Receptor_ID", "TRA", "TRB", "Peptide"]]
        if result.shape[0] > outcome.shape[0]:
            print(f"{result.loc[result['Peptide'].str.len() > 15].shape[0]} Antigens are longer than 15 aas, thus the "
                  f"corresponding samples will have prediction score of -1. All samples with HLA, that is not in "
                  f"HLA_seq_lib, will have score -1 too.")
            print(open(output_log, "r").read())
        result = result.merge(outcome.loc[:, ["TRB", "Peptide", "Score"]], on=["TRB", "Peptide"], how="left")
        result.loc[:, "Score"] = result["Score"].map(lambda x: -1 if pd.isna(x) else x)
        result["Score"].astype(float)
        result.fillna("", inplace=True)
        result["Receptor_ID"].astype(str)
        return {self.name: {
                            row[:4]: float("{:.4f}".format(row[4]))
                            for row in result.itertuples(index=False)
                           }
               }

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
        return None

    def predict(self, peptides, TCRs, repository: str, all: bool, **kwargs):
        """
        Overwrites ATCRSpecificityPrediction.predict

        Predicts binding probability between a T-cell receptor CDR3 protein sequence and a peptide
        :param peptides: The TCREpitope objects for which predictions should be performed
        :type peptides: :class:`~epytope.Core.TCREpitope.TCREpitope` or list(:class:`~epytope.Core.TCREpitope.TCREpitope`)
        :param TCRs: T cell receptor objects
        :type  :class:'~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor' or
        list(:class:'~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor')
        :param str repository: a path to a local github repository of pMTnet predictor
        :param bool all: if true each TCR object will be joined with each peptide to perform the prediction, otherwise
        the prediction will be preformed in the same order of the passed peptides and TCRs objects
        :param str trained_on: specifying the dataset the model trained on. This parameter is specific for ERGO, which
        has two models, one is trained on vdjdb and the other is trained on McPAS dataset.
        :param trained_model a string representing a path to the trained model directory. For ERGO this parameter will
        be ignored
        :param down: a boolean value for choosing from two different models of ImRex. If it is set to Ture, the model,
        trained of down sampled dataset of vdjdb will be selected, otherwise the other trained model will be chosen.
        Default value is False.
        :param nettcr_chain: a string specifying the chain(s) to use (a, b, ab). Default: b.
        :param pMTnet_interpreter: a string representing a path to python interpreter for pMTnet.
        :param chain: a string representing the type of the used cdr3-sequence in the prediction. It can be set to 'a'
        for alpha- or 'b' for beta-sequences. This parameter specifics the input chain for ATM-TCR.
        :param padding str: can be set to one of the following values ['front, end, mid, alignment'] and specifics the
        padding type for ATM-TCR.
        :param cuda bool: it can be set to True, if the running device has cuda.
        :return: A :class:`~epytope.Core.TCRSpecificityPredictionResult` object
        :rtype: :class:`~epytope.Core.TCRSpecificityPredictionResult`
        """
        df = super().predict(peptides=peptides, TCRs=TCRs, repository=repository, all=all)
        if "pMTnet_interpreter" in kwargs:
            pMTnet_interpreter = kwargs["pMTnet_interpreter"]
            if not os.path.exists(pMTnet_interpreter) or not os.path.basename(pMTnet_interpreter).startswith("python"):
                raise FileNotFoundError("pass a path to python interpreter for pMTnet")
        df_result = self.predict_from_dataset(repository=repository, df=df, score=-1,
                                              pMTnet_interpreter=pMTnet_interpreter)
        return df_result

    def prepare_dataset(self, df: pd.DataFrame, filename: str = None) -> pd.DataFrame:
        """
        process the dataset in the way to be suitable with pMTnet
        :param df: a dataframe contains TRA, TRB seqs and corresponding epitopes to predict, if they bind or not
        :param filename: str representing the name of the file to save the processed dataset
        :return: pd.DataFrame a dataframe, that has three columns (CDR3, Antigen, HLA)
        :rtype pd.DataFrame
        """

        def getAllele(s):
            candidate = ""
            pattern = re.compile(r"(-)([\w\d*:.-]+)")
            for _, allele in re.findall(pattern, s):
                if '*' in allele:
                    return allele
                else:
                    candidate = allele
            if candidate != "":
                return candidate
            else:
                return s

        test_set = df.loc[:, ["TRB", "Peptide", "MHC"]]
        test_set["MHC"] = test_set["MHC"].map(lambda x: getAllele(x))
        # change column names according to https://github.com/tianshilu/pMTnet/blob/master/test/input/test_input.csv
        test_set.columns = ["CDR3", "Antigen", "HLA"]
        if filename:
            test_set.to_csv(filename, index=False)
        return test_set

    def predict_from_dataset(self, repository: str, path: str = None, df: pd.DataFrame = None, source: str = None,
                             score: int = 1, **kwargs):
        """
        Predicts binding probability between a T-cell receptor CDR3 protein sequence and a peptide.
        The path should lead to csv file with fixed column names dataset.columns = ['TRA', 'TRB', "TRAV", "TRAJ",
        "TRBV", "TRBJ", "T-Cell-Type", "Peptide", "MHC", "Species", "Antigen.species", "Tissue"]. If some values for
        one or more variables are unavailable, leave them as blank cells.
        :param str repository: a path to a local github repository of pMTnet predictor
        :param str path: a string representing a path to the dataset(csv file), which will be processed. Default value
        is None, when the dataframe object is given
        :param `pd.DataFrame` df: a dataframe object. Default value is None, when the path is given
        :param str source: the source of the dataset [vdjdb, mcpas, scirpy, IEDB]. If this parameter is not passed,
         the dataset should be a csv file with the column names mentioned above
        :param int score: An integer representing a confidence score between 0 and 3 (0: critical information missing,
        1: medium confidence, 2: high confidence, 3: very high confidence). By processing all entries with a confidence
        score >= the passed parameter score will be kept. Default value is 1
        :param str trained_on: specifying the dataset the model trained on. This parameter is specific for ERGO, which
        has two models, one is trained on vdjdb and the other is trained on McPAS dataset.
        :param trained_model a string representing a path to the trained model directory. For ERGO this parameter will
        be ignored
        :param down: a boolean value for choosing from two different models of ImRex. If it is set to Ture, the model,
        trained of down sampled dataset of vdjdb will be selected, otherwise the other trained model will be chosen.
        Default value is False.
        :param nettcr_chain: a string specifying the chain(s) to use (a, b, ab). Default: b.
        :param pMTnet_interpreter: a string representing a path to python interpreter for pMTnet.
        :param chain: a string representing the type of the used cdr3-sequence in the prediction. It can be set to 'a'
        for alpha- or 'b' for beta-sequences. This parameter specifics the input chain for ATM-TCR.
        :param padding str: can be set to one of the following values ['front, end, mid, alignment'] and specifics the
        padding type for ATM-TCR.
        :param cuda bool: it can be set to True, if the running device has cuda.
        :return: A :class:`~epytope.Core.TCRSpecificityPredictionResult` object
        :rtype: :class:`~epytope.Core.TCRSpecificityPredictionResult`
        """

        if path is None and df is None:
            raise FileNotFoundError("A path to a csv file or a dataframe should be passed")
        if df is None:
            if os.path.isfile(path):
                df = process_dataset_TCR(path=path, source=source, score=score)
            else:
                raise FileNotFoundError("A path to a csv file or a dataframe should be passed")
        else:
            df = process_dataset_TCR(df=df, source=source, score=score)
        if not os.path.isdir(repository):
            raise NotADirectoryError("please pass a path as a string to a local pMTnet repository. To clone the "
                                     "repository type: 'git clone https://github.com/tianshilu/pMTnet.git' in the "
                                     "terminal")
        if "pMTnet_interpreter" in kwargs:
            pMTnet_interpreter = kwargs["pMTnet_interpreter"]
            if not os.path.exists(pMTnet_interpreter) or not os.path.basename(pMTnet_interpreter).startswith("python"):
                raise FileNotFoundError("pass a path to python interpreter for pMTnet")
        df.reset_index(drop=True, inplace=True)
        # process dataframe in the way to be suitable with pMTnet's input
        tmp_file = NamedTemporaryFile(delete=False)
        library = os.path.join(repository, "library")
        self.prepare_dataset(df, tmp_file.name)
        tmp_dir = tempfile.TemporaryDirectory()
        output_log = os.path.join(tmp_dir.name, "output.log")
        script = os.path.join(repository, "pMTnet.py")
        output = os.path.join(tmp_dir.name, "prediction.csv")
        try:
            cmd = f"{pMTnet_interpreter} {script} -input {tmp_file.name} -library {library} -output {tmp_dir.name} " \
                  f"-output_log {output_log}"
            p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
            stdo, stde = p.communicate()
            stdr = p.returncode
            if stdr > 0:
                raise RuntimeError("Unsuccessful execution of " + cmd + " (EXIT!=0) with output:\n" + stdo.decode())
            if os.path.getsize(output) == 0:
                raise RuntimeError(
                    "Unsuccessful execution of " + cmd + " (empty output file) with output:\n" +
                    stdo.decode())
        except Exception as e:
            raise RuntimeError(e)
        tmp_file.close()
        os.remove(tmp_file.name)
        result = self.parse_external_result(file=output, df=df, output_log=output_log)
        tmp_dir.cleanup()
        df_result = TCRSpecificityPredictionResult.from_dict(result)
        df_result.index = pd.MultiIndex.from_tuples(
            [tuple((ID, TRA, TRB, pep)) for ID, TRA, TRB, pep in df_result.index],
            names=["Receptor_ID", 'TRA', 'TRB', "Peptide"])
        return df_result


class ATM_TCR(AExternalTCRSpecificityPrediction, AExternal):
    """
    Implements ATM_TCR. "https://github.com/Lee-CBG/ATM-TCR"

    """
    __name = "ATM_TCR"
    __command = "python main.py"
    __version = " "

    @property
    def version(self) -> str:
        """
        The version of the Method
        """
        return self.__version

    @property
    def command(self) -> str:
        """
        Defines the commandline call for external tool
        """
        return self.__command

    @property
    def name(self) -> str:
        """The name of the predictor"""
        return self.__name

    @property
    def supportedPeptides(self) -> list:
        """
        A list of valid Peptides
        """
        return []

    def parse_external_result(self, file: str, df: pd.DataFrame, chain: str = "b"):
        """
        Parses external results and returns the result
        :param str file: The file path or the external prediction results
        :param pd.DataFrame df: the complete processed dataframe
        :param str output_log: log file with CDR, Antigen, HLA information
        :param chain: a string representing the type of the used cdr3-sequence in the prediction. It can be set to 'a'
        for alpha- or 'b' for beta-sequences
        :return: A dictionary containing the prediction results
        :rtype: dict{(str, str, str, str): float} {(Receptor_ID, TRA, TRB, Peptide): score}
        """
        outcome = pd.read_csv(file, header=None, sep="\t", names=["Peptide", "Seq", "Binding", "Prediction", "Score"])
        if chain == "a":
            seq = "TRA"
        else:
            seq = "TRB"
        result = df.loc[:, ["Receptor_ID", "TRA", "TRB", "Peptide"]]
        outcome.rename(columns={'Seq': seq}, inplace=True)
        result = result.merge(outcome.loc[:, [seq, "Peptide", "Score"]], on=[seq, "Peptide"], how="left")
        result.loc[:, "Score"] = result["Score"].map(lambda x: -1 if pd.isna(x) else x)
        result["Score"].astype(float)
        result.fillna("", inplace=True)
        result["Receptor_ID"].astype(str)
        if result.shape[0] > outcome.shape[0]:
            print(f"{result.loc[result['Score'] == -1].shape[0]} samples have no prediction scores due in part to the "
                  f"fact that the {seq} sequence is shorter than 3 amino acids. Moreover some peptides sequences can be"
                  f" modified during the prediction, thus can not match the corresponding peptides sequences in the "
                  f"test set")
        return {self.name: {
                            row[:4]: float("{:.4f}".format(row[4]))
                            for row in result.itertuples(index=False)
                           }
               }

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
        return None

    def predict(self, peptides, TCRs, repository: str, all: bool, **kwargs):
        """
        Overwrites ATCRSpecificityPrediction.predict

        Predicts binding probability between a T-cell receptor CDR3 protein sequence and a peptide
        :param peptides: The TCREpitope objects for which predictions should be performed
        :type peptides: :class:`~epytope.Core.TCREpitope.TCREpitope` or list(:class:`~epytope.Core.TCREpitope.TCREpitope`)
        :param TCRs: T cell receptor objects
        :type  :class:'~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor' or
        list(:class:'~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor')
        :param str repository: a path to a local github repository of pMTnet predictor
        :param bool all: if true each TCR object will be joined with each peptide to perform the prediction, otherwise
        the prediction will be preformed in the same order of the passed peptides and TCRs objects
        :param str trained_on: specifying the dataset the model trained on. This parameter is specific for ERGO, which
        has two models, one is trained on vdjdb and the other is trained on McPAS dataset.
        :param trained_model a string representing a path to the trained model directory. For ERGO this parameter will
        be ignored
        :param down: a boolean value for choosing from two different models of ImRex. If it is set to Ture, the model,
        trained of down sampled dataset of vdjdb will be selected, otherwise the other trained model will be chosen.
        Default value is False.
        :param nettcr_chain: a string specifying the chain(s) to use (a, b, ab). Default: b.
        :param pMTnet_interpreter: a string representing a path to python interpreter for pMTnet.
        :param chain: a string representing the type of the used cdr3-sequence in the prediction. It can be set to 'a'
        for alpha- or 'b' for beta-sequences. This parameter specifics the input chain for ATM-TCR.
        :param padding str: can be set to one of the following values ['front, end, mid, alignment'] and specifics the
        padding type for ATM-TCR.
        :param cuda bool: it can be set to True, if the running device has cuda.
        :return: A :class:`~epytope.Core.TCRSpecificityPredictionResult` object
        :rtype: :class:`~epytope.Core.TCRSpecificityPredictionResult`
        """
        df = super().predict(peptides=peptides, TCRs=TCRs, repository=repository, all=all)
        df_result = self.predict_from_dataset(repository=repository, df=df, score=-1, **kwargs)
        return df_result

    def prepare_dataset(self, df: pd.DataFrame, filename: str = None, chain: str = "b") -> \
            Tuple[pd.DataFrame, pd.core.series.Series]:
        """
        process the dataset in the way to be suitable with ATM-TCR
        :param df: a dataframe contains at least TRB seqs and corresponding epitopes to predict, if they bind or not
        :param filename: str representing the name of the file to save the processed dataset
        :param chain: a string representing the type of the used cdr3-sequence in the prediction. It can be set to 'a' 
        for alpha- or 'b' for beta-sequences 
        :return: (pd.DataFrame, pd.core.series.Series), where the dataframe has all samples, which have cdr3-beta-seqs,
        that are shorter than 37 aas, and epitopes, which are not longer than 22 aas. The function returns additionally
        series, which holds true values for the accepted samples, otherwise false values.
        :rtype pd.DataFrame, pd.core.series.Series
        """
        # select only the required feature to run ATM-TCR
        cdr3 = "TRB"
        if chain == "a":
            cdr3 = "TRA"
        test_set = df.loc[:, ["Peptide", cdr3]]
        mask = (test_set[cdr3].str.len() > 2)
        test_set = test_set.loc[mask, ]
        test_set["Binding"] = 1
        # remove all observation, in which no cdr3 beta seq or peptide seq doesn't occur
        test_set = test_set.loc[(test_set[cdr3] != "") & (test_set["Peptide"] != ""), ]
        # remove all entries, where cdr3 beta does not present
        test_set.dropna(inplace=True)
        if filename:
            test_set.to_csv(filename, sep=",", index=False, header=False)
        return test_set, mask

    def predict_from_dataset(self, repository: str, path: str = None, df: pd.DataFrame = None, source: str = None,
                             score: int = 1, **kwargs):
        """
        Predicts binding probability between a T-cell receptor CDR3 protein sequence and a peptide.
        The path should lead to csv file with fixed column names dataset.columns = ['TRA', 'TRB', "TRAV", "TRAJ",
        "TRBV", "TRBJ", "T-Cell-Type", "Peptide", "MHC", "Species", "Antigen.species", "Tissue"]. If some values for
        one or more variables are unavailable, leave them as blank cells.
        :param str repository: a path to a local github repository of pMTnet predictor
        :param str path: a string representing a path to the dataset(csv file), which will be processed. Default value
        is None, when the dataframe object is given
        :param `pd.DataFrame` df: a dataframe object. Default value is None, when the path is given
        :param str source: the source of the dataset [vdjdb, mcpas, scirpy, IEDB]. If this parameter is not passed,
         the dataset should be a csv file with the column names mentioned above
        :param int score: An integer representing a confidence score between 0 and 3 (0: critical information missing,
        1: medium confidence, 2: high confidence, 3: very high confidence). By processing all entries with a confidence
        score >= the passed parameter score will be kept. Default value is 1
        :param str trained_on: specifying the dataset the model trained on. This parameter is specific for ERGO, which
        has two models, one is trained on vdjdb and the other is trained on McPAS dataset.
        :param trained_model a string representing a path to the trained model directory. For ERGO this parameter will
        be ignored
        :param down: a boolean value for choosing from two different models of ImRex. If it is set to Ture, the model,
        trained of down sampled dataset of vdjdb will be selected, otherwise the other trained model will be chosen.
        Default value is False.
        :param nettcr_chain: a string specifying the chain(s) to use (a, b, ab). Default: b.
        :param pMTnet_interpreter: a string representing a path to python interpreter for pMTnet.
        :param chain: a string representing the type of the used cdr3-sequence in the prediction. It can be set to 'a'
        for alpha- or 'b' for beta-sequences. This parameter specifics the input chain for ATM-TCR.
        :param padding str: can be set to one of the following values ['front, end, mid, alignment'] and specifics the
        padding type for ATM-TCR.
        :param cuda bool: it can be set to true, if the running device has cuda.
        :return: A :class:`~epytope.Core.TCRSpecificityPredictionResult` object
        :rtype: :class:`~epytope.Core.TCRSpecificityPredictionResult`
        """
        if path is None and df is None:
            raise FileNotFoundError("A path to a csv file or a dataframe should be passed")
        if df is None:
            if os.path.isfile(path):
                df = process_dataset_TCR(path=path, source=source, score=score)
            else:
                raise FileNotFoundError("A path to a csv file or a dataframe should be passed")
        else:
            df = process_dataset_TCR(df=df, source=source, score=score)
        if not os.path.isdir(repository):
            raise NotADirectoryError("please pass a path as a string to a local ATM-TCR repository. To clone the "
                                     "repository type: 'git clone https://github.com/Lee-CBG/ATM-TCR.git' in the "
                                     "terminal")
        df.reset_index(drop=True, inplace=True)
        # process dataframe in the way to be suitable with ATM-TCR's input
        tmp_file = NamedTemporaryFile(delete=False)
        chain = "b"
        if chain in kwargs:
            chain = kwargs["chain"]
            if chain not in ["a", "b"]:
                raise ValueError(f"{chain} can be set only to one of the following values ['a', 'b']")
        max_len_tcr = 20
        max_len_pep = 22
        if "max_len_tcr" in kwargs:
            max_len_tcr = kwargs["max_len_tcr"]
        if type(max_len_tcr) is not int:
            raise ValueError(f"{max_len_tcr} should be an integer")
        if "max_len_pep" in kwargs:
            max_len_pep = kwargs["max_len_pep"]
            if type(max_len_pep) is not int:
                raise ValueError(f"{max_len_pep} should be an integer")
        _, mask = self.prepare_dataset(df, tmp_file.name, chain=chain)
        cuda = False
        if "cuda" in kwargs:
            cuda = kwargs["cuda"]
        padding = "mid"
        if "padding" in kwargs:
            padding = kwargs["padding"]
            if padding not in ['front, end, mid, alignment']:
                raise ValueError(f"{padding} should be set one of the following values ['front, end, mid, alignment']")
        infile = os.path.join(repository, "data/combined_dataset.csv")
        output = os.path.join(repository, f"result/pred_original_{os.path.basename(tmp_file.name)}")
        os.chdir(repository)
        try:
            cmd = f"python main.py --infile {infile} --indepfile {tmp_file.name} --mode test --cuda {cuda} --padding " \
                  f"{padding} --save_model {False} --max_len_tcr {max_len_tcr} --max_len_pep {max_len_pep}"
            p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
            stdo, stde = p.communicate()
            stdr = p.returncode
            if stdr > 0:
                raise RuntimeError("Unsuccessful execution of " + cmd + " (EXIT!=0) with output:\n" + stdo.decode())
            if os.path.getsize(output) == 0:
                raise RuntimeError(
                    "Unsuccessful execution of " + cmd + " (empty output file) with output:\n" +
                    stdo.decode())
        except Exception as e:
            raise RuntimeError(e)
        tmp_file.close()
        os.remove(tmp_file.name)
        result = self.parse_external_result(file=output, df=df, chain=chain)
        os.remove(output)
        df_result = TCRSpecificityPredictionResult.from_dict(result)
        df_result.index = pd.MultiIndex.from_tuples(
            [tuple((ID, TRA, TRB, pep)) for ID, TRA, TRB, pep in df_result.index],
            names=["Receptor_ID", 'TRA', 'TRB', "Peptide"])
        if sum(mask) < df.shape[0]:
            print(f"ATM_TCR's trained model could not make predictions for cdr3 sequences, which are less than 3 aas "
                  f"long. Moreover some peptides sequences can be modified during the prediction, thus can not "
                  f"match the corresponding peptides sequences in the test set")
        return df_result
