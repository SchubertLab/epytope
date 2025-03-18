# coding=utf-8
# This code is part of the epytope distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
.. module:: TCRSpecificityPrediction.ML
   :synopsis: This module contains all classes for ML-based TCR-epitope binding prediction.
.. moduleauthor:: albahah, drost
"""

import os
import sys
import subprocess
import tempfile
import pandas as pd
import numpy as np
import yaml
from functools import reduce
import operator
from epytope.Core.Base import ATCRSpecificityPrediction
from epytope.Core.TCREpitope import TCREpitope
from epytope.Core.ImmuneReceptor import ImmuneReceptor
from epytope.IO.IRDatasetAdapter import IRDataset
from epytope.Core.Result import TCRSpecificityPredictionResult
import re


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

        self.input_check(tcrs, epitopes, pairwise, **kwargs)
        data = self.format_tcr_data(tcrs, epitopes, pairwise, **kwargs)
        filenames, tmp_folder = self.save_tmp_files(data, **kwargs)
        cmd = self.get_base_cmd(filenames, tmp_folder, interpreter, conda, cmd_prefix, **kwargs)
        self.run_exec_cmd(cmd, filenames, interpreter, conda, cmd_prefix, **kwargs)
        df_results = self.format_results(filenames, tmp_folder, tcrs, epitopes, pairwise, **kwargs)
        self.clean_up(tmp_folder, filenames)
        return df_results

    def format_tcr_data(self, tcrs, epitopes, pairwise, **kwargs):
        raise NotImplementedError

    def save_tmp_files(self, data, **kwargs):
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
        cmd_conda = ""
        if conda:
            if sys.platform.startswith("win"):
                cmd_conda = f"conda activate {conda} &&"
            else:
                cmd_conda = f"conda run -n {conda}"
        cmd = f"import pkgutil; import sys; print(pkgutil.get_loader('{name}').get_filename())"
        cmd = f'{cmd_conda} {interpreter} -c "{cmd}"'
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

        cmd_conda = ""
        if conda:
            if sys.platform.startswith("win"):
                cmd_conda = f"conda activate {conda} &&"
            else:
                cmd_conda = f"conda run -n {conda}"
        if "m_cmd" in kwargs and not kwargs["m_cmd"]:
            cmds.append(f"{cmd_conda} {interpreter} {cmd}")
        else:
            cmds.append(f"{cmd_conda} {interpreter} -m {cmd}")

        self.exec_cmd(" && ".join(cmds), filenames[1])

    def format_results(self, filenames, tmp_folder, tcrs, epitopes, pairwise, **kwargs):
        raise NotImplementedError

    def transform_output(self, result_df, tcrs, epitopes, pairwise, joining_list, **kwargs):
        df_out = tcrs.to_pandas()
        if not pairwise:
            df_out = self.combine_tcrs_epitopes_list(df_out, epitopes)
            df_out = df_out.merge(result_df, on=joining_list, how="left")
            tuples = [("TCR", el) for el in df_out.columns][:-3]
            tuples.extend([("Epitope", "Peptide"), ("Epitope", "MHC"), ("Method", self.name.lower())])
            df_out.columns = pd.MultiIndex.from_tuples(tuples)
        else:
            tuples = [('TCR', el) for el in df_out.columns]
            for epitope in epitopes:
                df_tmp = result_df[result_df["Epitope"] == epitope.peptide].copy()
                if "MHC" in joining_list:
                    df_tmp = df_tmp[df_tmp["MHC"].apply(lambda v: str(v) if v else None).isin([epitope.allele if epitope.allele else None])].copy()
                tuples.extend([(str(epitope), self.name.lower())])
                df_out = df_out.merge(df_tmp, on=[el for el in joining_list if el not in ["Epitope", "MHC"]],
                                      how="left")
                df_out = df_out.drop(columns=["MHC", "Epitope"], errors="ignore")
            df_out.columns = pd.MultiIndex.from_tuples(tuples)
        return TCRSpecificityPredictionResult(df_out)

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
            p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,  # PIPE,
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
    __organism = "H"

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
    def organism(self):
        return self.__organism

    def format_tcr_data(self, tcrs, epitopes, pairwise, **kwargs):
        rename_columns = {
            "VDJ_cdr3": "cdr3",
        }
        df_tcrs = tcrs.to_pandas(rename_columns=rename_columns)

        if pairwise:
            df_tcrs = self.combine_tcrs_epitopes_pairwise(df_tcrs, epitopes)
        else:
            df_tcrs = self.combine_tcrs_epitopes_list(df_tcrs, epitopes)
        df_tcrs = df_tcrs.rename(columns={"Epitope": "antigen.epitope"})
        df_tcrs = df_tcrs[["cdr3", "antigen.epitope"]]
        df_tcrs = self.filter_by_length(df_tcrs, None, "cdr3", "antigen.epitope")
        df_tcrs = df_tcrs.drop_duplicates()
        df_tcrs = df_tcrs[(~df_tcrs["cdr3"].isna()) & (df_tcrs["cdr3"] != "")]
        return df_tcrs

    def save_tmp_files(self, data, **kwargs):
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

    def format_results(self, filenames, tmp_folder, tcrs, epitopes, pairwise, **kwargs):
        results_predictor = pd.read_csv(filenames[1])
        # results_predictor["MHC"] = results_predictor["MHC"].fillna("")
        results_predictor = results_predictor.rename(columns={"antigen.epitope": "Epitope",
                                                              "prediction_score": "Score",
                                                              "cdr3": "VDJ_cdr3"})
        joining_list = ["VDJ_cdr3", "Epitope"]#, "MHC"]
        df_out = self.transform_output(results_predictor, tcrs, epitopes, pairwise, joining_list)
        return df_out


class TITAN(ACmdTCRSpecificityPrediction):
    """
    Author: Weber et al.
    Paper: https://doi.org/10.1093/bioinformatics/btab294
    Repo: https://github.com/PaccMann/TITAN
    """
    __name = "TITAN"
    __command = "python scripts/flexible_model_eval.py"
    __version = "1.0.0"
    __tcr_length = (0, 500)
    __epitope_length = (0, 1028)
    __organism = "H"
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
    def organism(self):
        return self.__organism

    def format_tcr_data(self, tcrs, epitopes, pairwise, **kwargs):
        df_tcrs = tcrs.to_pandas()
        if df_tcrs["organism"].unique()[0] == "MusMusculus" and df_tcrs["organism"].nunique() == 1:
            self.organism_in = "mouse"
        else:
            self.organism_in = "human"

        df_tcrs["VDJ_v_gene"] = df_tcrs["VDJ_v_gene"].apply(lambda x: x if re.search(r"\*\d+$", x) else x + "*01")
        df_tcrs["VDJ_j_gene"] = df_tcrs["VDJ_j_gene"].apply(lambda x: x if re.search(r"\*\d+$", x) else x + "*01")
        if self.organism_in == "human":
            df_tcrs = df_tcrs[df_tcrs["VDJ_v_gene"].isin(self._v_regions) & df_tcrs["VDJ_j_gene"].isin(self._j_regions)]
        df_tcrs = df_tcrs[(~df_tcrs["VDJ_cdr3"].isna()) & (df_tcrs["VDJ_cdr3"] != "")]

        df_tcrs_unique = df_tcrs.drop_duplicates().copy()
        df_tcrs_unique["sequence_id"] = df_tcrs_unique.index
        df_tcrs_unique = df_tcrs_unique[["sequence_id", "VDJ_v_gene", "VDJ_j_gene", "VDJ_cdr3"]]
        tcr_2_id = {",".join(row[1][["VDJ_v_gene", "VDJ_j_gene", "VDJ_cdr3"]]): row[1]["sequence_id"]
                    for row in df_tcrs_unique.iterrows()}

        df_epitopes = pd.DataFrame({"ligand": [epitope.peptide.__str__() for epitope in epitopes]})
        df_epitopes_unique = df_epitopes.drop_duplicates().copy()
        df_epitopes_unique["ligand_id"] = df_epitopes_unique.index
        df_epitopes_unique = df_epitopes_unique[["ligand", "ligand_id"]]
        epitope_2_id = {row[1]["ligand"]: row[1]["ligand_id"] for row in df_epitopes_unique.iterrows()}

        if pairwise:
            df_matchup = [pd.DataFrame({"sequence_id": tcr_2_id.values(), "ligand_name": ligand})
                          for ligand in epitope_2_id.values()]
            df_matchup = pd.concat(df_matchup)
        else:
            df_matchup = pd.DataFrame()
            df_matchup["sequence_id"] = df_tcrs.apply(lambda row:
                                                      tcr_2_id[",".join(row[["VDJ_v_gene", "VDJ_j_gene", "VDJ_cdr3"]])],
                                                      axis=1)
            df_matchup["ligand_name"] = df_epitopes["ligand"].map(epitope_2_id)
        df_matchup = df_matchup.drop_duplicates()
        df_matchup["label"] = 1

        # TITAN skips the last batch if it is incomplete To have predictions for all tcrs we need to fill dummy values
        batch_size = 128
        n_fill = 128 - len(df_matchup) % batch_size
        if n_fill != batch_size:
            df_fill = pd.concat([df_matchup[-1:]] * n_fill)
            df_matchup = pd.concat([df_matchup, df_fill])
        df_matchup = df_matchup.reset_index()
        df_matchup.loc[0, "label"] = 0
        return [df_tcrs_unique, df_epitopes_unique, df_matchup]

    def save_tmp_files(self, data, **kwargs):
        tmp_folder = self.get_tmp_folder_path()
        path_tcrs = os.path.join(tmp_folder.name, f"{self.name}_raw_tcrs.csv")
        path_tcrs_full = os.path.join(tmp_folder.name, f"{self.name}_input_tcrs.csv")
        path_epitopes = os.path.join(tmp_folder.name, f"{self.name}_raw_epitopes.csv")
        path_epitopes_smi = os.path.join(tmp_folder.name, f"{self.name}_input_epitopes.smi")
        path_test = os.path.join(tmp_folder.name, f"{self.name}_input_matchup.csv")
        path_out = os.path.join(tmp_folder.name, f"{self.name}_output")

        data[0].to_csv(path_tcrs)
        data[1].to_csv(path_epitopes)
        data[2].to_csv(path_test)
        return [path_tcrs, path_tcrs_full, path_epitopes, path_epitopes_smi, path_test, path_out], tmp_folder

    def get_base_cmd(self, filenames, tmp_folder, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        path_module = self.get_package_dir("paccmann_tcr", interpreter, conda, cmd_prefix).split(os.sep)[:-1] + [".."]

        if self.organism_in == "mouse":
            path_imgt = os.sep.join(path_module + ["datasets", "imgt_mouse"])
        else:
            path_imgt = os.sep.join(path_module + ["datasets", "imgt"])

        if not os.path.exists(os.sep.join([path_imgt, "V_segment_sequences.fasta"])) or \
                not os.path.exists(os.sep.join([path_imgt, "J_segment_sequences.fasta"])):
            raise NotADirectoryError(f"Please download the V and J segment files from "
                                     "https://www.imgt.org/vquest/refseqh.html \n"
                                     "F+ORF+in-frame P - Amino acids - TRBV and TRBJ - Human "
                                     f"to {path_imgt} as V_segment_sequences.fasta and J_segment_sequences.fasta")

        # path tcr to full seq
        cmd_tcr = f"{os.sep.join(path_module + ['scripts/cdr3_to_full_seq.py'])} {path_imgt} " \
                  f"{filenames[0]} VDJ_v_gene VDJ_j_gene VDJ_cdr3 {filenames[1]}"

        # cmd epitopes to smi
        cmd_epitope = ["from pytoda.proteins.utils import aas_to_smiles",
                       "import pandas as pd",
                       f"df_epitopes = pd.read_csv(r'{filenames[2]}', index_col=0)",
                       "epitopes_smi = [aas_to_smiles(pep) for pep in df_epitopes['ligand']]",
                       "epitopes_smi = pd.DataFrame({'ligand': epitopes_smi, 'ligand_id': df_epitopes['ligand_id']})",
                       f"epitopes_smi.to_csv(r'{filenames[3]}', header=False, sep='\\t', index=False)",
                       ]
        cmd_epitope = f' -c "{"; ".join(cmd_epitope)}"'

        # cmd prediction
        trained_model = os.sep.join(path_module + ["trained_model"])
        cmd_model = f"{os.sep.join(path_module + ['scripts/flexible_model_eval.py'])} " \
                    f"{filenames[4]} {filenames[1]} {filenames[3]} {trained_model} bimodal_mca {filenames[5]}"
        return [cmd_tcr, cmd_epitope, cmd_model]

    def run_exec_cmd(self, cmd, filenames, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        super().run_exec_cmd(cmd[0], [None, filenames[1]], interpreter, conda, cmd_prefix, m_cmd=False, **kwargs)
        df_tcrs_full = pd.read_csv(filenames[1])[["full_seq", "sequence_id"]]
        df_tcrs_full.to_csv(filenames[1], index=False, header=False, sep="\t")
        super().run_exec_cmd(cmd[1], [None, filenames[3]], interpreter, conda, cmd_prefix, m_cmd=False, **kwargs)
        filenames[5] = f"{filenames[5]}.npy"
        super().run_exec_cmd(cmd[2], [None, filenames[5]], interpreter, conda, cmd_prefix, m_cmd=False, **kwargs)

    def format_results(self, filenames, tmp_folder, tcrs, epitopes, pairwise, **kwargs):
        results_predictor = np.load(filenames[5])[0]
        df_matchup = pd.read_csv(filenames[4], index_col=0)
        df_tcrs = pd.read_csv(filenames[0], index_col=0)
        df_epitopes = pd.read_csv(filenames[2], index_col=0)

        df_matchup = df_matchup[["sequence_id", "ligand_name"]]
        df_matchup["MHC"] = None
        df_matchup["Score"] = results_predictor
        df_matchup = df_matchup.drop_duplicates()

        df_matchup = pd.merge(df_matchup, df_tcrs, left_on="sequence_id", right_on="sequence_id")
        df_matchup = pd.merge(df_matchup, df_epitopes, left_on="ligand_name", right_on="ligand_id")
        df_matchup = df_matchup[["VDJ_v_gene", "VDJ_j_gene", "VDJ_cdr3", "ligand", "Score"]]
        df_matchup = df_matchup.rename(columns={"ligand": "Epitope"})

        joining_list = ["VDJ_v_gene", "VDJ_j_gene", "VDJ_cdr3", "Epitope"]
        df_out = self.transform_output(df_matchup, tcrs, epitopes, pairwise, joining_list)
        return TCRSpecificityPredictionResult(df_out)


class TCellMatch(ACmdTCRSpecificityPrediction):
    """
    Author: Fischer et al.
    Paper: https://www.embopress.org/doi/full/10.15252/msb.20199416
    Repo: https://github.com/theislab/tcellmatch/
    """
    __name = "TCellMatch"
    __version = ""
    __tcr_length = (0, 40)
    __epitope_length = (0, 25)
    __organism = "H"

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
    def organism(self):
        return self.__organism

    def format_tcr_data(self, tcrs, epitopes, pairwise, **kwargs):
        rename_columns = {
            "VDJ_cdr3": "Chain 2 CDR3 Curated",
        }
        df_tcrs = tcrs.to_pandas(rename_columns=rename_columns)

        if pairwise:
            df_tcrs = self.combine_tcrs_epitopes_pairwise(df_tcrs, epitopes)
        else:
            df_tcrs = self.combine_tcrs_epitopes_list(df_tcrs, epitopes)
        df_tcrs = df_tcrs.rename(columns={"Epitope": "Description"})

        df_tcrs = df_tcrs[(~df_tcrs["Chain 2 CDR3 Curated"].isna()) & (df_tcrs["Chain 2 CDR3 Curated"] != "")]

        df_tcrs = df_tcrs[["Chain 2 CDR3 Curated", "Description"]]
        df_tcrs = self.filter_by_length(df_tcrs, None, "Chain 2 CDR3 Curated", "Description")
        df_tcrs = df_tcrs.drop_duplicates()
        df_tcrs = df_tcrs.reset_index(drop=True)
        return df_tcrs

    def save_tmp_files(self, data, **kwargs):
        tmp_folder = self.get_tmp_folder_path()
        path_in = os.path.join(tmp_folder.name, f"{self.name}_input.tsv")
        path_out = os.path.join(tmp_folder.name, f"{self.name}_output.npy")
        data.to_csv(path_in, index=False)
        return [path_in, path_out], tmp_folder

    def get_base_cmd(self, filenames, tmp_folder, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        path_module = self.get_package_dir("tcellmatch", interpreter, conda, cmd_prefix).split(os.sep)[:-1]
        path_module = os.sep.join(path_module)

        path_blosum = f"{path_module}/../blosum/BLOSUM50.txt"
        if not os.path.isfile(path_blosum):
            self.add_blosum(path_blosum)

        model = "iedb_BILSTM_CONCAT_LEARN_1_1_1_1_1_s_bilstm_cv0" if "model" not in kwargs else kwargs["model"]
        subfolder = model.split("_")[1].lower()
        subfolder += f"_separate" if "separate" in model.lower() else ""

        path_model = f"{path_module}/../models/iedb_best/s_{subfolder}/models/"
        path_model += model
        if not os.path.isfile(f"{path_model}_model_settings.pkl"):
            raise ValueError(
                f"Model {path_model}_model_settings.pkl does not exist. Please download model from  "
                "https://doi.org/10.6084/m9.figshare.24526015"
                f" or specify the path to a model via 'model=<path>'")
        path_utils = os.path.dirname(__file__)
        cmd = f"{path_utils}/Utils.py tcellmatch {path_model} {filenames[0]} {filenames[1]} {path_blosum}"
        return cmd

    def run_exec_cmd(self, cmd, filenames, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        super().run_exec_cmd(cmd, filenames, interpreter, conda, cmd_prefix, m_cmd=False, **kwargs)

    def format_results(self, filenames, tmp_folder, tcrs, epitopes, pairwise, **kwargs):
        results_predictor = pd.read_csv(filenames[0])
        scores = np.load(filenames[1])
        assert len(scores) == len(results_predictor), "Length mismatch between input and output."
        results_predictor["Score"] = scores
        results_predictor = results_predictor.rename(columns={"Chain 2 CDR3 Curated": "VDJ_cdr3", "Description": "Epitope"})
        joining_list = ["VDJ_cdr3", "Epitope"]
        results_predictor = results_predictor[joining_list + ["Score"]]
        df_out = self.transform_output(results_predictor, tcrs, epitopes, pairwise, joining_list)
        return df_out

    def add_blosum(self, path_file):
        path_folder = os.sep.join(path_file.split(os.sep)[:-1])
        os.makedirs(path_folder, exist_ok=True)
        text = ["0 # https://www.ncbi.nlm.nih.gov/IEB/ToolBox/C_DOC/lxr/source/data/BLOSUM50#L2, 2019/07/12\n",
                "1 # Entries for the BLOSUM50 matrix at a scale of ln(2)/3.0.\n",
                "2 _  A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  B  J  Z  X  *\n",
                "3 A  5 -2 -1 -2 -1 -1 -1  0 -2 -1 -2 -1 -1 -3 -1  1  0 -3 -2  0 -2 -2 -1 -1 -5\n",
                "4 R -2  7 -1 -2 -4  1  0 -3  0 -4 -3  3 -2 -3 -3 -1 -1 -3 -1 -3 -1 -3  0 -1 -5\n",
                "5 N -1 -1  7  2 -2  0  0  0  1 -3 -4  0 -2 -4 -2  1  0 -4 -2 -3  5 -4  0 -1 -5\n",
                "6 D -2 -2  2  8 -4  0  2 -1 -1 -4 -4 -1 -4 -5 -1  0 -1 -5 -3 -4  6 -4  1 -1 -5\n",
                "7 C -1 -4 -2 -4 13 -3 -3 -3 -3 -2 -2 -3 -2 -2 -4 -1 -1 -5 -3 -1 -3 -2 -3 -1 -5\n",
                "8 Q -1  1  0  0 -3  7  2 -2  1 -3 -2  2  0 -4 -1  0 -1 -1 -1 -3  0 -3  4 -1 -5\n",
                "9 E -1  0  0  2 -3  2  6 -3  0 -4 -3  1 -2 -3 -1 -1 -1 -3 -2 -3  1 -3  5 -1 -5\n",
                "10 G  0 -3  0 -1 -3 -2 -3  8 -2 -4 -4 -2 -3 -4 -2  0 -2 -3 -3 -4 -1 -4 -2 -1 -5\n",
                "11 H -2  0  1 -1 -3  1  0 -2 10 -4 -3  0 -1 -1 -2 -1 -2 -3  2 -4  0 -3  0 -1 -5\n",
                "12 I -1 -4 -3 -4 -2 -3 -4 -4 -4  5  2 -3  2  0 -3 -3 -1 -3 -1  4 -4  4 -3 -1 -5\n",
                "13 L -2 -3 -4 -4 -2 -2 -3 -4 -3  2  5 -3  3  1 -4 -3 -1 -2 -1  1 -4  4 -3 -1 -5\n",
                "14 K -1  3  0 -1 -3  2  1 -2  0 -3 -3  6 -2 -4 -1  0 -1 -3 -2 -3  0 -3  1 -1 -5\n",
                "15 M -1 -2 -2 -4 -2  0 -2 -3 -1  2  3 -2  7  0 -3 -2 -1 -1  0  1 -3  2 -1 -1 -5\n",
                "16 F -3 -3 -4 -5 -2 -4 -3 -4 -1  0  1 -4  0  8 -4 -3 -2  1  4 -1 -4  1 -4 -1 -5\n",
                "17 P -1 -3 -2 -1 -4 -1 -1 -2 -2 -3 -4 -1 -3 -4 10 -1 -1 -4 -3 -3 -2 -3 -1 -1 -5\n",
                "18 S  1 -1  1  0 -1  0 -1  0 -1 -3 -3  0 -2 -3 -1  5  2 -4 -2 -2  0 -3  0 -1 -5\n",
                "19 T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  2  5 -3 -2  0  0 -1 -1 -1 -5\n",
                "20 W -3 -3 -4 -5 -5 -1 -3 -3 -3 -3 -2 -3 -1  1 -4 -4 -3 15  2 -3 -5 -2 -2 -1 -5\n",
                "21 Y -2 -1 -2 -3 -3 -1 -2 -3  2 -1 -1 -2  0  4 -3 -2 -2  2  8 -1 -3 -1 -2 -1 -5\n",
                "22 V  0 -3 -3 -4 -1 -3 -3 -4 -4  4  1 -3  1 -1 -3 -2  0 -3 -1  5 -3  2 -3 -1 -5\n",
                "23 B -2 -1  5  6 -3  0  1 -1  0 -4 -4  0 -3 -4 -2  0  0 -5 -3 -3  6 -4  1 -1 -5\n",
                "24 J -2 -3 -4 -4 -2 -3 -3 -4 -3  4  4 -3  2  1 -3 -3 -1 -2 -1  2 -4  4 -3 -1 -5\n",
                "25 Z -1  0  0  1 -3  4  5 -2  0 -3 -3  1 -1 -4 -1  0 -1 -2 -2 -3  1 -3  5 -1 -5\n",
                "26 X -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -5\n",
                "27 * -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5  1"]
        with open(path_file, "w") as file_blosum:
            file_blosum.writelines(text)


class STAPLER(ACmdTCRSpecificityPrediction):
    """
    Author: Kwee et al.
    Paper: https://www.biorxiv.org/content/10.1101/2023.04.25.538237v1
    Repo: https://github.com/NKI-AI/STAPLER/
    """
    __name = "stapler"
    __version = ""
    __tcr_length = (5, 25)  # TODO
    __epitope_length = (9, 9) # TODO
    __organism = "H"

    _rename_columns = {
        "VDJ_cdr3": "cdr3_beta_aa",
        "VDJ_v_gene": "TRBV_IMGT",
        "VDJ_j_gene": "TRBJ_IMGT",
        "VJ_cdr3": "cdr3_alpha_aa",
        "VJ_v_gene": "TRAV_IMGT",
        "VJ_j_gene": "TRAJ_IMGT",
    }

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
    def organism(self):
        return self.__organism

    def format_tcr_data(self, tcrs, epitopes, pairwise, **kwargs):
        df_tcrs = tcrs.to_pandas(rename_columns=self._rename_columns)

        if pairwise:
            df_tcrs = self.combine_tcrs_epitopes_pairwise(df_tcrs, epitopes)
        else:
            df_tcrs = self.combine_tcrs_epitopes_list(df_tcrs, epitopes)

        for col in self._rename_columns.values():
            df_tcrs = df_tcrs[(~df_tcrs[col].isna()) & (df_tcrs[col] != "")]
        df_tcrs = df_tcrs.rename(columns={"Epitope": "epitope_aa"})
        df_tcrs = self.filter_by_length(df_tcrs, "cdr3_alpha_aa", "cdr3_beta_aa", "epitope_aa")
        df_tcrs = df_tcrs[list(self._rename_columns.values()) + ["epitope_aa", "organism"]]
        for col in self._rename_columns.values():
            df_tcrs = df_tcrs[(~df_tcrs[col].isna()) & (df_tcrs[col]!='nan')]
        df_tcrs["label_true_pair"] = 0
        df_tcrs = df_tcrs.drop_duplicates()
        return df_tcrs

    def save_tmp_files(self, data, **kwargs):
        tmp_folder = self.get_tmp_folder_path()
        path_in_raw = os.path.join(tmp_folder.name, f"{self.name}_raw_input.csv")
        path_in = os.path.join(tmp_folder.name, f"{self.name}_input.csv")
        path_out = os.path.join(tmp_folder.name, "predictions_5_fold_ensamble.csv")
        data.to_csv(path_in_raw)
        return [path_in_raw, path_in, path_out], tmp_folder

    def get_base_cmd(self, filenames, tmp_folder, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        path_utils = os.path.dirname(__file__)
        cmd_reconstruct = f"{path_utils}/Utils.py stapler {filenames[0]} {filenames[1]}"

        path_module = self.get_package_dir("stapler", interpreter, conda, cmd_prefix).split(os.sep)[:-1]
        path_module = os.sep.join(path_module)

        path_models = os.sep.join([path_module, "..", "model"])
        if not os.path.isdir(path_models):
            raise ValueError(f"Model {path_models} does not exist. Please download the models from  "
                             f"https://files.aiforoncology.nl/stapler/´to {path_models}")

        cmd_predict = f"{path_module}/../tools/test.py"

        self.change_configs(path_module, tmp_folder, filenames)
        return [cmd_reconstruct, cmd_predict]

    def change_configs(self, path_module, tmp_dir, filenames):
        path_config = f"{path_module}/../config"
        path_paths = f"{path_config}/paths/default.yaml"
        yaml_paths = {
            "root_dir": tmp_dir.name,
            "log_dir": f"{tmp_dir.name}/logs",
            "output_dir": tmp_dir.name,
            "work_dir": tmp_dir.name,
        }
        with open(path_paths, "w") as file_paths:
            yaml.dump(yaml_paths, file_paths)
        paths = {
                f"{path_config}/datamodule/train_dataset.yaml": {"test_data_path": filenames[1],
                                                                 "train_data_path": filenames[1], 
                                                                },
                f"{path_config}/test.yaml": {"test_from_ckpt_path": f"{path_module}/../model/finetuned_model_refactored/"},
                f"{path_config}/callbacks/train_model_checkpoint.yaml": {"model_checkpoint->dirpath": f"{path_module}/../model/"},
                f"{path_config}/model/train_medium_model.yaml": {"checkpoint_path": f"{path_module}/../model/pretrained_model/pre-cdr3_combined_epoch=437-train_mlm_loss=0.702.ckpt"},
        }
        for k, v in paths.items():
            self.change_yaml(k, v)

    def change_yaml(self, path_yaml, change_dict):
        def get_by_path(dictionary, keys):
            return reduce(operator.getitem, keys, dictionary)
        
        with open(path_yaml, "r") as file_yaml:
            yaml_content = yaml.safe_load(file_yaml)
        for k, v in change_dict.items():
            levels = k.split("->")
            get_by_path(yaml_content, levels[:-1])[levels[-1]] = v   
        with open(path_yaml, "w") as file_yaml:
            yaml.dump(yaml_content, file_yaml)

    def run_exec_cmd(self, cmd, filenames, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        super().run_exec_cmd(cmd[0], [None, filenames[1]], interpreter, conda, cmd_prefix, m_cmd=False, **kwargs)
        super().run_exec_cmd(cmd[1], [None, filenames[2]], interpreter, conda, cmd_prefix, m_cmd=False, **kwargs)

    def format_results(self, filenames, tmp_folder, tcrs, epitopes, pairwise, **kwargs):
        results_predictor = pd.read_csv(filenames[2], index_col=0)
        rename_dict = {v: k for k, v in self._rename_columns.items()}
        rename_dict["epitope_aa"] = "Epitope"
        rename_dict["pred_cls"] = "Score"
        results_predictor = results_predictor.rename(columns=rename_dict)
        joining_list = list(self._rename_columns.keys()) + ["Epitope"]
        results_predictor = results_predictor[joining_list + ["Score"]]
        df_out = self.transform_output(results_predictor, tcrs, epitopes, pairwise, joining_list)
        return df_out
