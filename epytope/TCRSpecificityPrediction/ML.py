# coding=utf-8
# This code is part of the epytope distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
.. module:: TCRSpecificityPrediction.ML
   :synopsis: This module contains all classes for ML-based TCR-epitope binding prediction.
.. moduleauthor:: albahah, drost
"""

import abc
import os
import subprocess
import tempfile
import numpy as np
import pandas as pd
from epytope.Core.Base import ATCRSpecificityPrediction
from epytope.Core.TCREpitope import TCREpitope
from epytope.Core.ImmuneReceptor import ImmuneReceptor
from epytope.IO.IRDatasetAdapter import IRDataset
from epytope.Core.Result import TCRSpecificityPredictionResult
import re
# from pytoda.proteins.utils import aas_to_smiles
from pathlib import Path
from typing import Tuple


class ACmdTCRSpecificityPrediction(ATCRSpecificityPrediction):
    """
        Abstract base class for external TCR specificity prediction methods.
        Implements predict functionality.
    """

    def predict(self, tcrs, epitopes, pairwise=True, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        """
        Predicts binding score between a T-cell receptor and an epitope.
        :param tcrs: the T cell receptors containing the sequence and gene information
        :type tcrs: :class:`~epytope.Core.ImmuneReceptor.TCellreceptor`
        or list(:class:`~epytope.Core.ImmuneReceptor.TCellreceptor`) or :class:`~epytope.IO.IRDatasetAdapter.IRDataset`
        :param epitopes: epitope sequences
        :type  epitopes: str or :class:'~epytope.Core.TCREpitope.TCREpitope' or
        list(:class:'~epytope.Core.TCREpitope.TCREpitope')
        :param bool pairwise: predict binding between all tcr-epitope pairs.
        Otherwise, tcrs[i] will be tested against epitopes[i] (requires len(tcrs)==len(epitopes))
        :param str interpreter: path to the python interpreter, where the predictor is installed
        :param str conda: conda environment of the predictor
        :param str cmd_prefix: Prefix for the command line input before the predictor is executed, which can be used
        to activate the environments (e.g. venv, poetry, ...) where the predictor is installed
        :param kwargs: attributes that will be passed to the predictor without a check
        :return: Returns a :class:`~epytope.Core.Result.TCRSpecificityPredictionResult` object for the specified
                 :class:`~epytope.Core.ImmuneReceptor.TCRReceptor` and :class:`~epytope.Core.TCREpitope.TCREpitope`
        :rtype: :class:`~epytope.Core.Result.TCRSpecificityPredictionResult`
        """
        if isinstance(epitopes, TCREpitope):
            epitopes = [epitopes]
        if isinstance(tcrs, ImmuneReceptor):
            tcrs = [tcrs]
        if isinstance(tcrs, list):
            tcrs = IRDataset(tcrs)
        if pairwise:
            epitopes = list(set(epitopes))

        self.input_check(tcrs, epitopes, pairwise)
        data = self.format_tcr_data(tcrs, epitopes, pairwise)
        filenames, tmp_folder = self.save_tmp_files(data)
        cmd = self.get_base_cmd(filenames, tmp_folder, interpreter, conda, cmd_prefix, **kwargs)
        self.run_exec_cmd(cmd, filenames, interpreter, conda, cmd_prefix, **kwargs)
        df_results = self.format_results(filenames, tcrs, pairwise)
        self.clean_up(tmp_folder, filenames)
        return df_results

    def format_tcr_data(self, tcrs, epitopes, pairwise):
        raise NotImplementedError

    def save_tmp_files(self, data):
        """
        Saves a pd.DataFrame to a temporary directory.
        :param pd.DataFrame data: Data frame containing tcr and epitope data
        :return: list(str), str, containing the created files and the path to the tmp dir
        """
        tmp_folder = self.get_tmp_folder_path()
        path_in = os.path.join(tmp_folder.name, f"{self.name}_input.csv")
        path_out = os.path.join(tmp_folder.name, f"{self.name}_output.csv")
        data.to_csv(path_in)
        return [path_in, path_out], tmp_folder

    def get_base_cmd(self, filenames, tmp_folder, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        raise NotImplementedError

    def get_package_dir(self, name, interpreter=None, conda=None, cmd_prefix=None):
        """
        Returns the path of a named model in this environment.
        """
        interpreter = "python" if interpreter is None else interpreter
        cmds = []
        if cmd_prefix is not None:
            cmds.append(cmd_prefix)
        if conda is not None:
            cmds.append(f"conda activate {conda}")
        cmd = f"import pkgutil; import sys; print(pkgutil.get_loader('{name}').get_filename())"
        cmd = f'{interpreter} -c "{cmd}"'
        cmds.append(cmd)
        cmd = " && ".join(cmds)
        try:
            p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
            stdo, stde = p.communicate()
            stdr = p.returncode
            if stdr > 0 or len(stdo.decode()) == 0:
                raise RuntimeError(f"Not able to find {name} for {self.name}. Did you install the package?"
                                   f"{stdo.decode()}"
                                   f"{stde.decode()}")
        except Exception as e:
            raise RuntimeError(e)
        return stdo.decode()

    def run_exec_cmd(self, cmd, filenames, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        interpreter = "python" if interpreter is None else interpreter
        cmds = []
        if cmd_prefix is not None:
            cmds.append(cmd_prefix)
        if conda is not None:
            cmds.append(f"conda activate {conda}")
        cmds.append(f"{interpreter} -m {cmd}")
        self.exec_cmd(" && ".join(cmds), filenames[1])

    def format_results(self, filenames, tcrs, pairwise):
        raise NotImplementedError
    
    def transform_output(self, result_df, tcrs, pairwise, joining_list, method=None):
        if not pairwise:
            df_tcrs = tcrs.to_pandas()
            df_out = df_tcrs.merge(result_df, on=joining_list, how="left")
            tuples = [('TCR', el) for el in df_tcrs.columns]
            tuples.extend([("Epitope", "Peptide"), ("Epitope", "MHC"), ("Method", method)])
            df_out.columns = pd.MultiIndex.from_tuples(tuples)
        else:
            df_out = tcrs.to_pandas()
            epitopes = result_df[["Peptide", "MHC"]].drop_duplicates()
            epitopes = [TCREpitope(row["Peptide"], row["MHC"]) for i, row in epitopes.iterrows()]
            tuples = [('TCR', el) for el in df_out.columns]
            for epitope in epitopes:
                df_tmp = result_df[(result_df["Peptide"] == epitope.peptide) &
                                    (result_df["MHC"].astype(str) == (epitope.allele if epitope.allele
                                                                                        else ""))].copy()
                tuples.extend([(str(epitope), method)])
                df_out = df_out.merge(df_tmp, on=joining_list, how="left")
                df_out = df_out.drop(columns=["MHC", "Peptide"])
            df_out.columns = pd.MultiIndex.from_tuples(tuples)
        return df_out

    def clean_up(self, tmp_folder, files=None):
        """
        Removes temporary directories and files.
        :param tmp_folder: path to the folder where temporary data of this predictor is stored
        :type tmp_folder: :class:`tempfile:TemporaryDirector`
        :param list(str) files: additional list of files that will be removed
        """
        tmp_folder.cleanup()
        if files is not None:
            for file in files:
                if tmp_folder.name not in file:
                    os.remove(file)

    def get_tmp_folder_path(self):
        """
        Create a new folder in tmp for intermediate results.
        :return: path to the folder
        """
        return tempfile.TemporaryDirectory()

    def exec_cmd(self, cmd, tmp_path_out):
        """
        Run a command in a subprocess.
        :param str cmd: shell command
        :param str tmp_path_out: path to the output file
        """
        try:
            p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, #PIPE,
                                 stderr=subprocess.STDOUT)
            stdo, stde = p.communicate()
            stdr = p.returncode
            if stdr > 0:
                raise RuntimeError("Unsuccessful execution of " + cmd + " (EXIT!=0) with output:\n" + stdo.decode())
            if not os.path.exists(tmp_path_out) or os.path.getsize(tmp_path_out) == 0:
                raise RuntimeError(
                    "Unsuccessful execution of " + cmd + " (empty output file) with output:\n" +
                    stdo.decode())
        except Exception as e:
            raise RuntimeError(e)


class ImRex(ACmdTCRSpecificityPrediction):
    """
    Author: Moris et al.
    Paper: https://doi.org/10.1093/bib/bbaa318
    Repo: https://github.com/pmoris/ImRex
    """
    __name = "ImRex"
    __version = ""
    __tcr_length = (10, 20)
    __epitope_length = (8, 11)
    __cmd = "todo"  #todo

    @property
    def version(self):
        return self.__version

    @property
    def name(self):
        return self.__name

    @property
    def tcr_length(self):
        return self.__tcr_length

    @property
    def epitope_length(self):
        return self.__epitope_length

    @property
    def cmd(self):
        return self.__cmd

    def format_tcr_data(self, tcrs, epitopes, pairwise):
        rename_columns = {
            "VDJ_cdr3": "cdr3",
        }
        df_tcrs = tcrs.to_pandas(rename_columns=rename_columns)

        if pairwise:
            df_tcrs = self.combine_tcrs_epitopes_pairwise(df_tcrs, epitopes)
        else:
            df_tcrs = self.combine_tcrs_epitopes_list(df_tcrs, epitopes)
        df_tcrs = df_tcrs.rename(columns={"Epitope": "antigen.epitope"})
        df_tcrs = df_tcrs[["cdr3", "antigen.epitope", "MHC"]]
        df_tcrs = self.filter_by_length(df_tcrs, None, "cdr3", "antigen.epitope")
        df_tcrs = df_tcrs[(~df_tcrs["cdr3"].isna()) & (df_tcrs["cdr3"] != "")]
        return df_tcrs

    def save_tmp_files(self, data):
        tmp_folder = self.get_tmp_folder_path()
        path_in = os.path.join(tmp_folder.name, f"{self.name}_input.csv")
        path_out = os.path.join(tmp_folder.name, f"{self.name}_output.csv")
        data.to_csv(path_in, index=False, sep=";")
        return [path_in, path_out], tmp_folder

    def get_base_cmd(self, filenames, tmp_folder, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        path_module = self.get_package_dir("src", interpreter, conda, cmd_prefix).split(os.sep)[:-1]
        path_module = os.sep.join(path_module)
        model = f"2020-07-24_19-18-39_trbmhcidown-shuffle-padded-b32-" \
                f"lre4-reg001/2020-07-24_19-18-39_trbmhcidown-shuffle-padded-b32-lre4-reg001.h5"
        if "model" in kwargs:
            model = kwargs["model"]
        model = f"{path_module}/../models/pretrained/{model}"

        cmd = f"src.scripts.predict.predict --model {model} --input {filenames[0]} --output {filenames[1]}"
        return cmd

    def format_results(self, filenames, tcrs, pairwise):
        results_predictor = pd.read_csv(filenames[1])
        results_predictor["MHC"] = results_predictor["MHC"].fillna("")
        results_predictor = results_predictor.rename(columns={"antigen.epitope": "Peptide",
                                                              "prediction_score": "Score"})
        df_out = TCRSpecificityPredictionResult.from_output(results_predictor, tcrs, pairwise, self.name)
        return df_out


class TITAN(ACmdTCRSpecificityPrediction):
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


class ATM_TCR():
    """
    Author: Cai
    Paper: https://www.frontiersin.org/articles/10.3389/fimmu.2022.893247/full
    Repo: https://github.com/Lee-CBG/ATM-TCR
    """
    __name = "ATM-TCR"
    __version = ""
    __trc_length = (0, 40) # todo
    __epitope_length = (0, 40) # todo

    @property
    def version(self):
        return self.__version

    @property
    def name(self):
        return self.__name

    @property
    def tcr_length(self):
        return self.__trc_length

    @property
    def epitope_length(self):
        return self.__epitope_length

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