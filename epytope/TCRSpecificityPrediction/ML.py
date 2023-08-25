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
        data = self.format_tcr_data(tcrs, epitopes, pairwise)
        filenames, tmp_folder = self.save_tmp_files(data, **kwargs)
        cmd = self.get_base_cmd(filenames, tmp_folder, interpreter, conda, cmd_prefix, **kwargs)
        self.run_exec_cmd(cmd, filenames, interpreter, conda, cmd_prefix, **kwargs)
        df_results = self.format_results(filenames, tcrs, epitopes, pairwise)
        self.clean_up(tmp_folder, filenames)
        return df_results

    def format_tcr_data(self, tcrs, epitopes, pairwise):
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

    def format_results(self, filenames, tcrs, epitopes, pairwise):
        raise NotImplementedError

    def transform_output(self, result_df, tcrs, epitopes, pairwise, joining_list):
        df_out = tcrs.to_pandas()
        if not pairwise:
            df_out = self.combine_tcrs_epitopes_list(df_out, epitopes)
            df_out = df_out.merge(result_df, on=joining_list, how="left")
            tuples = [("TCR", el) for el in df_out.columns][:-3]
            tuples.extend([("Epitope", "Peptide"), ("Epitope", "MHC"), ("Method", self.name)])
            df_out.columns = pd.MultiIndex.from_tuples(tuples)
        else:
            tuples = [('TCR', el) for el in df_out.columns]
            for epitope in epitopes:
                df_tmp = result_df[result_df["Epitope"] == epitope.peptide].copy()
                if "MHC" in joining_list:
                    df_tmp = df_tmp[df_tmp["MHC"].astype(str) == (epitope.allele if epitope.allele else "")].copy()
                tuples.extend([(str(epitope), self.name)])
                df_out = df_out.merge(df_tmp, on=[el for el in joining_list if el not in ["Epitope", "MHC"]],
                                      how="left")
                df_out = df_out.drop(columns=["MHC", "Epitope"], errors="ignore")
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

    def format_results(self, filenames, tcrs, epitopes, pairwise):
        results_predictor = pd.read_csv(filenames[1])
        results_predictor["MHC"] = results_predictor["MHC"].fillna("")
        results_predictor = results_predictor.rename(columns={"antigen.epitope": "Epitope",
                                                              "prediction_score": "Score",
                                                              "cdr3": "VDJ_cdr3"})
        joining_list = ["VDJ_cdr3", "Epitope", "MHC"]
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
    __tcr_length = (0, 40)  # TODO
    __epitope_length = (0, 40)  # TODO
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

    def format_tcr_data(self, tcrs, epitopes, pairwise):
        df_tcrs = tcrs.to_pandas()
        df_tcrs["VDJ_v_gene"] = df_tcrs["VDJ_v_gene"].apply(lambda x: x if re.search(r"\*\d+$", x) else x + "*01")
        df_tcrs["VDJ_j_gene"] = df_tcrs["VDJ_j_gene"].apply(lambda x: x if re.search(r"\*\d+$", x) else x + "*01")
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

    def format_results(self, filenames, tcrs, epitopes, pairwise):
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
