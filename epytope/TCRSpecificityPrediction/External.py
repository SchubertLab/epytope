# This code is part of the epytope distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
.. module:: TCRSpecificityPrediction
   :synopsis: This module contains all classes for external TCR specificity prediction methods.
.. moduleauthor:: albahah, drost, chernysheva
"""

import abc
import os
import shutil
import sys
import warnings

import pandas as pd

from epytope.TCRSpecificityPrediction.ML import ACmdTCRSpecificityPrediction


class ARepoTCRSpecificityPrediction(ACmdTCRSpecificityPrediction):
    _repository_path = None

    @property
    @abc.abstractmethod
    def repo(self):
        """
        Path to the repository.
        """
        raise NotImplementedError

    """
        Abstract base class for external TCR specificity prediction methods that are not installable.
        They require the User to clone the git version and then specify a path to this repo.
        Implements predict functionality.
    """

    def input_check(self, tcrs, epitopes, pairwise, **kwargs):
        super().input_check(tcrs, epitopes, pairwise, **kwargs)
        if "repository" not in kwargs:
            raise AttributeError(f"Please provide 'repository' as a input argument to predict"
                                 f" for external tolls like {self.name} to point to the codebase."
                                 f"You can obtain the repo via:\n"
                                 f"'git clone {self.repo}'")
        repository = kwargs["repository"]
        if repository is None or repository == "" or not os.path.isdir(repository):
            raise NotADirectoryError(f"Repository: '{repository}' does not exist."
                                     f"Please provide a keyword argument repository with the path to the {self.name}.\n"
                                     f"You can obtain the repo via: \n"
                                     f"'git clone {self.repo}'")
        self.repository_path = os.path.abspath(repository)

    def run_exec_cmd(self, cmd, filenames, interpreter=None, conda=None, cmd_prefix=None, repository="", **kwargs):
        old_dir = os.getcwd()
        os.chdir(repository)

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
        cmds.append(f"{cmd_conda} {interpreter} {cmd}")
        self.exec_cmd(" && ".join(cmds), filenames[1])
        os.chdir(old_dir)


class Ergo2(ARepoTCRSpecificityPrediction):
    """
    Author: Springer et al.
    Paper: https://www.frontiersin.org/articles/10.3389/fimmu.2021.664514/full
    Repo: https://github.com/IdoSpringer/ERGO-II
    """
    __name = "ERGO-II"
    __version = ""
    __tcr_length = (0, 28)
    __epitope_length = (0, 99999)
    __repo = "https://github.com/IdoSpringer/ERGO-II.git"
    __organism = "HM"

    _rename_columns = {
        "VJ_cdr3": "TRA",
        "VDJ_cdr3": "TRB",
        "VJ_v_gene": "TRAV",
        "VJ_j_gene": "TRAJ",
        "VDJ_v_gene": "TRBV",
        "VDJ_j_gene": "TRBJ",
        "celltype": "T-Cell-Type",
    }

    @property
    def name(self):
        return self.__name

    @property
    def version(self):
        return self.__version

    @property
    def tcr_length(self):
        return self.__tcr_length

    @property
    def epitope_length(self):
        return self.__epitope_length

    @property
    def repo(self):
        return self.__repo

    @property
    def organism(self):
        return self.__organism

    def format_tcr_data(self, tcrs, epitopes, pairwise, **kwargs):
        required_columns = list(self._rename_columns.values()) + ["Peptide", "MHC"]
        df_tcrs = tcrs.to_pandas(rename_columns=self._rename_columns)

        def assign_cd4_cd8(celltype):
            celltype = "CD4" if "CD4" in celltype else "CD8" if "CD8" in celltype else ""
            return celltype

        df_tcrs["T-Cell-Type"] = df_tcrs["T-Cell-Type"].apply(assign_cd4_cd8)

        if pairwise:
            df_tcrs = self.combine_tcrs_epitopes_pairwise(df_tcrs, epitopes)
        else:
            df_tcrs = self.combine_tcrs_epitopes_list(df_tcrs, epitopes)
        df_tcrs = df_tcrs.rename(columns={"Epitope": "Peptide"})

        df_tcrs = df_tcrs[required_columns]
        df_tcrs = self.filter_by_length(df_tcrs, None, "TRB", "Peptide")
        df_tcrs = df_tcrs.drop_duplicates()
        df_tcrs = df_tcrs[(~df_tcrs["TRB"].isna()) & (df_tcrs["TRB"] != "")]
        return df_tcrs

    def get_base_cmd(self, filenames, tmp_folder, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        dataset = "vdjdb" if "dataset" not in kwargs else kwargs["dataset"]
        return f"Predict.py {dataset} {filenames[0]} {filenames[1]}"

    def run_exec_cmd(self, cmd, filenames, interpreter=None, conda=None, cmd_prefix=None, repository="", **kwargs):
        if repository is not None and repository != "" and os.path.isdir(repository):
            self.correct_code(repository)
        super().run_exec_cmd(cmd, filenames, interpreter, conda, cmd_prefix, repository)

    def format_results(self, filenames, tmp_folder, tcrs, epitopes, pairwise, **kwargs):
        results_predictor = pd.read_csv(filenames[1], index_col=0)
        results_predictor = results_predictor.fillna("")
        results_predictor = results_predictor.rename(columns={k: v for v, k in self._rename_columns.items()})
        results_predictor = results_predictor.rename(columns={"Peptide": "Epitope"})
        joining_list = list(self._rename_columns.keys()) + ["Epitope", "MHC"]
        joining_list.remove("celltype")
        results_predictor = results_predictor[joining_list + ["Score"]]
        df_out = self.transform_output(results_predictor, tcrs, epitopes, pairwise, joining_list)
        return df_out

    def correct_code(self, path_repo):
        """
        The github repo contains several bugs, which will be corrected here.
        """
        script = []
        with open(os.path.join(path_repo, "Predict.py"), "r") as f:
            script.extend(f.readlines())
        # make output to pandas
        if "    df.to_csv(sys.argv[3], sep=',', index=False)\n" not in script:
            idx = script.index("    df = predict(sys.argv[1], sys.argv[2])\n")
            script.insert(idx + 1, "    df.to_csv(sys.argv[3], sep=',', index=False)\n")
            with open(os.path.join(path_repo, "Predict.py"), "w") as f:
                f.writelines(script)

        # Cpu + gpu usable
        script = []
        with open(os.path.join(path_repo, "Models.py"), "r") as f:
            script.extend(f.readlines())
        if "        checkpoint = torch.load(ae_file)\n" in script:
            idx = script.index("        checkpoint = torch.load(ae_file)\n")
            script[idx] = "        checkpoint = torch.load(ae_file, " \
                          "map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))\n"
            with open(os.path.join(path_repo, "Models.py"), "w") as f:
                f.writelines(script)

        # rename folders
        if os.path.isdir(os.path.join(path_repo, "Models", "AE")):
            shutil.move(os.path.join(path_repo, "Models", "AE"), os.path.join(path_repo, "TCR_Autoencoder"))


class pMTnet(ARepoTCRSpecificityPrediction):
    """
    Author: Lu et al.
    Paper: https://www.nature.com/articles/s42256-021-00383-2
    Repo: https://github.com/tianshilu/pMTnet
    """
    __name = "pMTnet"
    __version = ""
    __trc_length = (0, 80)
    __epitope_length = (0, 15)
    __cmd = "pMTnet.py"
    __organism = "H"
    __repo = "https://github.com/tianshilu/pMTnet"

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

    @property
    def cmd(self):
        return self.__cmd

    @property
    def repo(self):
        return self.__repo

    @property
    def organism(self):
        return self.__organism

    def input_check(self, tcrs, epitopes, pairwise, **kwargs):
        super().input_check(tcrs, epitopes, pairwise, **kwargs)
        allowed_alleles = ["A*", "B*", "C*", "E*"]
        for epitope in epitopes:
            if epitope.allele is None:
                raise ValueError("Missing MHC-Annotation: pMTnet requires MHC information")
            if epitope.allele.name[:2] not in allowed_alleles:
                raise ValueError(f"Invalid MHC {epitope.allele}: pMTnet requires MHC out of {allowed_alleles}")

    def format_tcr_data(self, tcrs, epitopes, pairwise, **kwargs):
        rename_columns = {
            "VDJ_cdr3": "CDR3",
        }
        df_tcrs = tcrs.to_pandas(rename_columns=rename_columns)
        if pairwise:
            df_tcrs = self.combine_tcrs_epitopes_pairwise(df_tcrs, epitopes)
        else:
            df_tcrs = self.combine_tcrs_epitopes_list(df_tcrs, epitopes)
        df_tcrs = df_tcrs.rename(columns={"Epitope": "Antigen", "MHC": "HLA"})
        df_tcrs = df_tcrs[["CDR3", "Antigen", "HLA"]]
        df_tcrs = self.filter_by_length(df_tcrs, None, "CDR3", "Antigen")
        df_tcrs = df_tcrs[(~df_tcrs["CDR3"].isna()) & (df_tcrs["CDR3"] != "")]
        df_tcrs["HLA"] = df_tcrs["HLA"].astype(str).str[4:]
        df_tcrs = df_tcrs.drop_duplicates()
        return df_tcrs

    def save_tmp_files(self, data, **kwargs):
        filenames, tmp_dir = super().save_tmp_files(data, **kwargs)
        filenames[1] = os.sep.join(filenames[1].split(os.sep)[:-1])
        filenames.append(f"{tmp_dir.name}/{self.name}_logs.log")
        return filenames, tmp_dir

    def get_base_cmd(self, filenames, tmp_folder, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        repository = self.repository_path
        cmd = f"pMTnet.py -input {filenames[0]} -library {repository}/library " \
              f"-output {filenames[1]} -output_log {filenames[2]}"
        return cmd

    def format_results(self, filenames, tmp_folder, tcrs, epitopes, pairwise, **kwargs):
        results_predictor = pd.read_csv(f"{filenames[1]}/prediction.csv")
        results_predictor = results_predictor.rename(columns={"Rank": "Score", "HLA": "MHC", "Antigen": "Epitope",
                                                              "CDR3": "VDJ_cdr3"})
        results_predictor["Score"] = 1 - results_predictor["Score"]  # in github: lower rank = good prediction => invert
        joining_list = ["VDJ_cdr3", "Epitope", "MHC"]
        results_predictor["MHC"] = "HLA-" + results_predictor["MHC"]
        df_out = self.transform_output(results_predictor, tcrs, epitopes, pairwise, joining_list)
        return df_out


class EpiTCR(ARepoTCRSpecificityPrediction):
    """
    Author: Pham et al.
    Paper: https://academic.oup.com/bioinformatics/article/39/5/btad284/7140137
    Repo: https://github.com/ddiem-ri-4D/epiTCR
    """
    __name = "epiTCR"
    __version = ""
    __trc_length = (8, 19)
    __epitope_length = (8, 11)
    __organism = "HM"
    __repo = "https://github.com/ddiem-ri-4D/epiTCR.git"
    _model_input = None

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

    @property
    def repo(self):
        return self.__repo

    @property
    def organism(self):
        return self.__organism

    def format_tcr_data(self, tcrs, epitopes, pairwise, **kwargs):
        self._model_input = "cem" if "model" not in kwargs or "WithMHC" in kwargs["model"] else "ce"
        rename_columns = {
            "VDJ_cdr3": "CDR3b",
        }
        required_columns = list(rename_columns.values()) + ["epitope", "binder"] + (self._model_input == "cem") * [
            "HLA"]
        df_tcrs = tcrs.to_pandas(rename_columns=rename_columns)
        if pairwise:
            df_tcrs = self.combine_tcrs_epitopes_pairwise(df_tcrs, epitopes)
        else:
            df_tcrs = self.combine_tcrs_epitopes_list(df_tcrs, epitopes)
        df_tcrs = df_tcrs.rename(columns={"Epitope": "epitope", "MHC": "HLA"})
        df_tcrs = self.filter_by_length(df_tcrs, None, "CDR3b", "epitope")
        df_tcrs = df_tcrs[(~df_tcrs["CDR3b"].isna()) & (df_tcrs["CDR3b"] != "")]
        df_tcrs["binder"] = 1
        df_tcrs = df_tcrs[required_columns]
        df_tcrs.drop_duplicates(inplace=True, keep="first")

        if self._model_input == "cem":
            repository = self.repository_path
            mhcseq_filepath = os.path.join(repository, "data/hlaCovertPeudoSeq/HLAWithPseudoSeq.csv")
            if not os.path.exists(mhcseq_filepath):
                raise TypeError(f"Please unzip the file stored at {mhcseq_filepath}.zip to {mhcseq_filepath}")
            data_hla = pd.read_csv(mhcseq_filepath)
            data_hla = data_hla.rename(columns={"HLA name": "HLA_name",
                                                "HLA PSEUDO SEQ": "MHC_seq"})
            df_tcrs = df_tcrs[(~df_tcrs["HLA"].isna()) & (df_tcrs["HLA"] != "")]
            df_tcrs["HLA"] = df_tcrs["HLA"].astype(str).str[4:]
            df_tcrs = pd.merge(df_tcrs, data_hla, left_on="HLA", right_on="HLA_name", how="left")
            df_tcrs = df_tcrs.rename(columns={"HLA_name": "HLA_exactmatch", "MHC_seq": "MHC"})
            data_hla["HLA_name"] = data_hla["HLA_name"].astype(str).str[:7]
            data_hla.drop_duplicates(subset="HLA_name", inplace=True)
            df_tcrs = pd.merge(df_tcrs, data_hla, left_on="HLA", right_on="HLA_name", how="left")
            df_tcrs.drop_duplicates(subset=required_columns, ignore_index=True, inplace=True)

            def custom_formatwarning(message, category, filename, lineno, line=''):
                return category.__name__ + ": " + str(message) + "\n"

            warnings.formatwarning = custom_formatwarning
            if df_tcrs["MHC"].isna().sum() > 0:
                warnings.warn(f"{df_tcrs['MHC'].isna().sum()} entries do not have exact HLA allele matches in the MHC "
                              f"pseudosequence dictionary provided by {self.name} tool, the first best matches were chosen instead.")
                df_tcrs["MHC"].fillna(df_tcrs["MHC_seq"], inplace=True)
            required_columns.append("MHC")
            df_tcrs = df_tcrs[required_columns]
        if df_tcrs.shape[0] == 1:
            df_tcrs = pd.concat([df_tcrs] * 2).sort_index().reset_index(drop=True)
        df_tcrs.iat[0, df_tcrs.columns.get_loc("binder")] = 0
        return df_tcrs

    def get_base_cmd(self, filenames, tmp_folder, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        model = "rdforestWithMHCModel" if "model" not in kwargs else kwargs["model"]
        repository = self.repository_path
        model_filepath = os.path.join(repository, "models", f"{model}.pickle")
        if not os.path.exists(model_filepath):
            raise TypeError(f"Please unzip the models stored at {model_filepath}.zip to {model_filepath}")
        cmd = f"predict.py --testfile {filenames[0]} --modelfile {model_filepath}"
        cmd += f"--chain {self._model_input} --outfile {filenames[1]}"
        return cmd

    def format_results(self, filenames, tmp_folder, tcrs, epitopes, pairwise, **kwargs):
        results_predictor = pd.read_csv(filenames[1], index_col=False)
        rename_dict = {"CDR3b": "VDJ_cdr3",
                       "epitope": "Epitope",
                       "predict_proba": "Score"}

        required_columns = ["VDJ_cdr3", "Epitope", "Score"]  # TODO: Does the model use MHC, if so add here!
        joining_list = ["VDJ_cdr3", "Epitope"]
        if self._model_input == "cem":
            results_predictor["HLA"] = "HLA-" + results_predictor["HLA"].astype(str)
            results_predictor = results_predictor.drop(columns=["MHC"])
            joining_list.append("MHC")
            required_columns.append("MHC")
            rename_dict["HLA"] = "MHC"

        results_predictor = results_predictor.rename(columns=rename_dict)

        results_predictor = results_predictor[required_columns]
        results_predictor = results_predictor.drop_duplicates()
        df_out = self.transform_output(results_predictor, tcrs, epitopes, pairwise, joining_list)
        return df_out


class ATM_TCR(ARepoTCRSpecificityPrediction):
    """
    Author: Cai
    Paper: https://www.frontiersin.org/articles/10.3389/fimmu.2022.893247/full
    Repo: https://github.com/Lee-CBG/ATM-TCR
    """
    __name = "ATM-TCR"
    __version = ""
    __trc_length = (0, 20)
    __epitope_length = (0, 22)
    __organism = "H"
    __repo = "https://github.com/Lee-CBG/ATM-TCR"

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

    @property
    def repo(self):
        return self.__repo

    @property
    def organism(self):
        return self.__organism

    def format_tcr_data(self, tcrs, epitopes, pairwise, **kwargs):
        rename_columns = {
            "VDJ_cdr3": "TCR",
        }
        df_tcrs = tcrs.to_pandas(rename_columns=rename_columns)
        if pairwise:
            df_tcrs = self.combine_tcrs_epitopes_pairwise(df_tcrs, epitopes)
        else:
            df_tcrs = self.combine_tcrs_epitopes_list(df_tcrs, epitopes)
        df_tcrs["Binding Affinity"] = 1
        df_tcrs = df_tcrs[["Epitope", "TCR", "Binding Affinity"]]
        df_tcrs = self.filter_by_length(df_tcrs, None, "TCR", "Epitope")
        df_tcrs = df_tcrs[(~df_tcrs["TCR"].isna()) & (df_tcrs["TCR"] != "")]
        df_tcrs = df_tcrs.drop_duplicates()
        return df_tcrs

    def save_tmp_files(self, data, **kwargs):
        paths, tmp_folder = super().save_tmp_files(data)
        paths[1] = os.path.join(self.repository_path, "result", "pred_original_ATM-TCR_input.csv")
        data.to_csv(paths[0], header=False, index=False)
        return paths, tmp_folder

    def get_base_cmd(self, filenames, tmp_folder, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        if "cuda" in kwargs:
            cuda = kwargs["cuda"]
        else:
            import tensorflow as tf
            cuda = tf.test.is_gpu_available(cuda_only=True)
        cmd = f"main.py --infile data/combined_dataset.csv --indepfile {filenames[0]} --mode test --cuda {cuda}"
        return cmd

    def format_results(self, filenames, tmp_folder, tcrs, epitopes, pairwise, **kwargs):
        results_predictor = pd.read_csv(filenames[1], sep="\t", header=None)
        results_predictor.columns = ["Epitope", "VDJ_cdr3", "Label", "Binary", "Score"]
        results_predictor = results_predictor[["Epitope", "VDJ_cdr3", "Score"]]

        joining_list = ["VDJ_cdr3", "Epitope"]
        df_out = self.transform_output(results_predictor, tcrs, epitopes, pairwise, joining_list)
        return df_out


class AttnTAP(ARepoTCRSpecificityPrediction):
    """
    Author: Xu et al.
    Paper: https://www.frontiersin.org/articles/10.3389/fgene.2022.942491/full
    Repo: https://github.com/Bioinformatics7181/AttnTAP
    """
    __name = "AttnTAP"
    __version = ""
    __trc_length = (6, 30)
    __epitope_length = (0, 9)  # TODO check one more time
    __organism = "H"
    __repo = "https://github.com/Bioinformatics7181/AttnTAP.git"

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

    @property
    def repo(self):
        return self.__repo

    @property
    def organism(self):
        return self.__organism

    def format_tcr_data(self, tcrs, epitopes, pairwise, **kwargs):
        rename_columns = {
            "VDJ_cdr3": "tcr"
        }
        required_columns = list(rename_columns.values()) + ["antigen"]
        df_tcrs = tcrs.to_pandas(rename_columns=rename_columns)
        if pairwise:
            df_tcrs = self.combine_tcrs_epitopes_pairwise(df_tcrs, epitopes)
        else:
            df_tcrs = self.combine_tcrs_epitopes_list(df_tcrs, epitopes)
        df_tcrs = df_tcrs.rename(columns={"Epitope": "antigen"})
        df_tcrs = self.filter_by_length(df_tcrs, None, "tcr", "antigen")
        df_tcrs = df_tcrs[(~df_tcrs["tcr"].isna()) & (df_tcrs["tcr"] != "")]
        df_tcrs = df_tcrs[required_columns]
        df_tcrs.drop_duplicates(inplace=True, keep="first")
        df_tcrs["label"] = 1
        if df_tcrs.shape[0] == 1:
            df_tcrs = pd.concat([df_tcrs] * 2).sort_index().reset_index(drop=True)
        df_tcrs.iat[0, df_tcrs.columns.get_loc("label")] = 0
        return df_tcrs

    def get_base_cmd(self, filenames, tmp_folder, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        model = "cv_model_0_vdjdb_0" if "model" not in kwargs else kwargs["model"]
        repository = self.repository_path
        model_filepath = os.path.join(repository, "Models", f"{model}.pt")
        path_script = os.path.join("Codes", "AttnTAP_test.py")
        cmd = f"{path_script} --input_file {filenames[0]} --output_file {filenames[1]} "
        cmd += f"--load_model_file {model_filepath}"
        return cmd

    def format_results(self, filenames, tmp_folder, tcrs, epitopes, pairwise, **kwargs):
        results_predictor = pd.read_csv(filenames[1])
        results_predictor = results_predictor.rename(columns={"tcr": "VDJ_cdr3",
                                                              "antigen": "Epitope",
                                                              "prediction": "Score"})
        required_columns = ["VDJ_cdr3", "Epitope", "Score"]
        joining_list = ["VDJ_cdr3", "Epitope"]
        results_predictor = results_predictor[required_columns]
        results_predictor = results_predictor.drop_duplicates(subset=joining_list)
        df_out = self.transform_output(results_predictor, tcrs, epitopes, pairwise, joining_list)
        return df_out


class TEIM(ARepoTCRSpecificityPrediction):
    """
    Author: Peng et al.
    Paper: https://www.nature.com/articles/s42256-023-00634-4
    Repo: https://github.com/pengxingang/TEIM
    """
    __name = "TEIM"
    __version = ""
    __trc_length = (10, 20)
    __epitope_length = (8, 12)
    __organism = "H"
    __repo = "https://github.com/pengxingang/TEIM.git"

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

    @property
    def repo(self):
        return self.__repo

    @property
    def organism(self):
        return self.__organism

    def format_tcr_data(self, tcrs, epitopes, pairwise, **kwargs):
        rename_columns = {
            "VDJ_cdr3": "cdr3"
        }
        required_columns = list(rename_columns.values()) + ["epitope"]
        df_tcrs = tcrs.to_pandas(rename_columns=rename_columns)
        if pairwise:
            df_tcrs = self.combine_tcrs_epitopes_pairwise(df_tcrs, epitopes)
        else:
            df_tcrs = self.combine_tcrs_epitopes_list(df_tcrs, epitopes)
        df_tcrs = df_tcrs.rename(columns={"Epitope": "epitope"})
        df_tcrs = self.filter_by_length(df_tcrs, None, "cdr3", "epitope")
        df_tcrs = df_tcrs[(~df_tcrs["cdr3"].isna()) & (df_tcrs["cdr3"] != "")]
        df_tcrs = df_tcrs[required_columns]
        df_tcrs.drop_duplicates(inplace=True, keep="first")
        return df_tcrs

    def get_base_cmd(self, filenames, tmp_folder, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        path_script = os.path.join("scripts", "inference_seq.py")
        return f"{path_script}"

    def save_tmp_files(self, data, **kwargs):
        repository = self.repository_path
        path_in = os.path.join(repository, "inputs", "inputs_bd.csv")
        path_out = os.path.join(repository, "outputs", "sequence_level_binding.csv")
        data.to_csv(path_in)
        return [path_in, path_out], None

    def clean_up(self, tmp_folder, files=None):
        for file in files:
            if os.path.exists(file):
                os.remove(file)

    def format_results(self, filenames, tmp_folder, tcrs, epitopes, pairwise, **kwargs):
        results_predictor = pd.read_csv(filenames[1])
        results_predictor = results_predictor.rename(columns={"cdr3": "VDJ_cdr3",
                                                              "epitope": "Epitope",
                                                              "binding": "Score"})
        required_columns = ["VDJ_cdr3", "Epitope", "Score"]
        joining_list = ["VDJ_cdr3", "Epitope"]
        results_predictor = results_predictor[required_columns]
        results_predictor = results_predictor.drop_duplicates()
        df_out = self.transform_output(results_predictor, tcrs, epitopes, pairwise, joining_list)
        return df_out


class BERTrand(ARepoTCRSpecificityPrediction):
    """
    Author: Myronov et al.
    Paper: https://www.biorxiv.org/content/biorxiv/early/2023/06/13/2023.06.12.544613.full.pdf?%3Fcollection=
    Repo: https://github.com/SFGLab/bertrand
    """
    __name = "BERTrand"
    __version = ""
    __tcr_length = (10, 20)
    __epitope_length = (8, 11)
    __organism = "H"
    __repo = "https://github.com/SFGLab/bertrand.git"

    _rename_columns = {
        "VDJ_cdr3": "CDR3b"
    }

    @property
    def name(self):
        return self.__name

    @property
    def version(self):
        return self.__version

    @property
    def tcr_length(self):
        return self.__tcr_length

    @property
    def epitope_length(self):
        return self.__epitope_length

    @property
    def repo(self):
        return self.__repo

    @property
    def organism(self):
        return self.__organism

    def format_tcr_data(self, tcrs, epitopes, pairwise, **kwargs):
        required_columns = list(self._rename_columns.values()) + ["peptide_seq"]
        df_tcrs = tcrs.to_pandas(rename_columns=self._rename_columns)
        if pairwise:
            df_tcrs = self.combine_tcrs_epitopes_pairwise(df_tcrs, epitopes)
        else:
            df_tcrs = self.combine_tcrs_epitopes_list(df_tcrs, epitopes)
        df_tcrs = df_tcrs.rename(columns={"Epitope": "peptide_seq"})
        df_tcrs = self.filter_by_length(df_tcrs, None, "CDR3b", "peptide_seq")
        df_tcrs = df_tcrs[(~df_tcrs["CDR3b"].isna()) & (df_tcrs["CDR3b"] != "")]
        df_tcrs = df_tcrs[required_columns]
        df_tcrs = df_tcrs.drop_duplicates()
        df_tcrs["y"] = 1
        return df_tcrs

    def get_base_cmd(self, filenames, tmp_folder, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        model = f"{self.repository_path}/models/best_checkpoint" if "model" not in kwargs else kwargs["model"]
        if not os.path.isdir(model):
            raise ValueError(
                f"Please download and unzip model from git repository or"
                f" https://drive.google.com/file/d/1FywbDbzhhYbwf99MdZrpYQEbXmwX9Zxm/view?usp=sharing"
                f" to {kwargs['repository']}/models or specify the path via 'model=<path>'")
        return f"bertrand.model.inference -i={filenames[0]} -m={model} -o={filenames[1]}"

    def run_exec_cmd(self, cmd, filenames, interpreter=None, conda=None, cmd_prefix=None, repository="", **kwargs):
        old_dir = os.getcwd()
        os.chdir(repository)
        cmds = []
        if cmd_prefix is not None:
            cmds.append(cmd_prefix)

        cmd_conda = ""
        if conda:
            if sys.platform.startswith("win"):
                cmd_conda = f"conda activate {conda} &&"
            else:
                cmd_conda = f"conda run -n {conda}"
        cmds.append(f"{cmd_conda} python -m {cmd}")
        self.exec_cmd(" && ".join(cmds), filenames[1])
        os.chdir(old_dir)

    def format_results(self, filenames, tmp_folder, tcrs, epitopes, pairwise, **kwargs):
        results_predictor = pd.read_csv(filenames[1])
        input_predictor = pd.read_csv(filenames[0])
        joining_list = ["VDJ_cdr3", "Epitope"]
        results_predictor[joining_list] = input_predictor[["CDR3b", "peptide_seq"]]
        results_predictor = results_predictor.rename(columns={"0": "Score"})
        required_columns = ["VDJ_cdr3", "Epitope", "Score"]
        results_predictor = results_predictor[required_columns]

        df_out = self.transform_output(results_predictor, tcrs, epitopes, pairwise, joining_list)
        return df_out


class Ergo1(ARepoTCRSpecificityPrediction):
    """
    Author: Springer et al.
    Paper: https://www.frontiersin.org/articles/10.3389/fimmu.2020.01803/full
    Repo: https://github.com/louzounlab/ERGO
    """
    __name = "ERGO-I"
    __version = ""
    __tcr_length = (0, 9999)
    __epitope_length = (0, 9999)
    __organism = "HM"
    __repo = "https://github.com/louzounlab/ERGO.git"

    _rename_columns = {
        "VDJ_cdr3": "CDR3b"
    }

    @property
    def name(self):
        return self.__name

    @property
    def version(self):
        return self.__version

    @property
    def tcr_length(self):
        return self.__tcr_length

    @property
    def epitope_length(self):
        return self.__epitope_length

    @property
    def repo(self):
        return self.__repo

    @property
    def organism(self):
        return self.__organism

    def format_tcr_data(self, tcrs, epitopes, pairwise, **kwargs):
        required_columns = list(self._rename_columns.values()) + ["epitope"]
        df_tcrs = tcrs.to_pandas(rename_columns=self._rename_columns)
        if pairwise:
            df_tcrs = self.combine_tcrs_epitopes_pairwise(df_tcrs, epitopes)
        else:
            df_tcrs = self.combine_tcrs_epitopes_list(df_tcrs, epitopes)
        df_tcrs = df_tcrs.rename(columns={"Epitope": "epitope"})
        df_tcrs = df_tcrs[required_columns]
        df_tcrs = self.filter_by_length(df_tcrs, None, "CDR3b", "epitope")
        df_tcrs = df_tcrs.drop_duplicates()
        df_tcrs = df_tcrs[(~df_tcrs["CDR3b"].isna()) & (df_tcrs["CDR3b"] != "")]

        if len(df_tcrs) % 50 == 0:
            # If the length of samples is devisible by the batch size, something weird happens
            df_tcrs.loc[df_tcrs.index.max()+1] = df_tcrs.iloc[-1]

        return df_tcrs

    def save_tmp_files(self, data, **kwargs):
        tmp_folder = self.get_tmp_folder_path()
        path_in = os.path.join(tmp_folder.name, f"{self.name}_input.csv")
        path_out = os.path.join(tmp_folder.name, f"{self.name}_output.csv")
        data.to_csv(path_in, index=False, header=False)
        return [path_in, path_out], tmp_folder

    def get_base_cmd(self, filenames, tmp_folder, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        model_type = "lstm" if "model" not in kwargs else kwargs["model"].split("_")[0]
        cuda = "cpu" if "cuda" not in kwargs else kwargs["cuda"]
        repository = self.repository_path
        model = "lstm_vdjdb1" if "model" not in kwargs else kwargs["model"]
        model_filepath = os.path.join(repository, "models", f"{model}.pt")
        database = "vdjdb" if "model" not in kwargs else kwargs["model"].split("_")[1][:-1]
        return f"ERGO.py predict {model_type} {database} specific {cuda} --model_file={model_filepath} " \
               f"--train_data_file=auto --test_data_file={filenames[0]} >> {filenames[1]}"

    def format_results(self, filenames, tmp_folder, tcrs, epitopes, pairwise, **kwargs):
        results_predictor = pd.read_csv(filenames[1], sep='\t', names=["VDJ_cdr3", "Epitope", "Score"], header=None)
        results_predictor = results_predictor.drop_duplicates()
        joining_list = ["Epitope", "VDJ_cdr3"]
        results_predictor = results_predictor[joining_list + ["Score"]]
        df_out = self.transform_output(results_predictor, tcrs, epitopes, pairwise, joining_list)
        return df_out


class TEINet(ARepoTCRSpecificityPrediction):
    """
    Author: Jiang et al.
    Paper: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008814
    Repo: https://github.com/jiangdada1221/TEINet
    """
    __name = "TEINet"
    __version = ""
    __tcr_length = (5, 30)
    __epitope_length = (7, 15)
    __organism = "H"
    __repo = "https://github.com/jiangdada1221/TEINet.git"

    _rename_columns = {
        "VDJ_cdr3": "CDR3.beta"
    }

    @property
    def name(self):
        return self.__name

    @property
    def version(self):
        return self.__version

    @property
    def tcr_length(self):
        return self.__tcr_length

    @property
    def epitope_length(self):
        return self.__epitope_length

    @property
    def repo(self):
        return self.__repo

    @property
    def organism(self):
        return self.__organism

    def format_tcr_data(self, tcrs, epitopes, pairwise, **kwargs):
        required_columns = list(self._rename_columns.values()) + ["Epitope"]
        df_tcrs = tcrs.to_pandas(rename_columns=self._rename_columns)
        if pairwise:
            df_tcrs = self.combine_tcrs_epitopes_pairwise(df_tcrs, epitopes)
        else:
            df_tcrs = self.combine_tcrs_epitopes_list(df_tcrs, epitopes)
        df_tcrs = self.filter_by_length(df_tcrs, None, "CDR3.beta", "Epitope")
        df_tcrs = df_tcrs[(~df_tcrs["CDR3.beta"].isna()) & (df_tcrs["CDR3.beta"] != "")]
        df_tcrs = df_tcrs[required_columns]
        df_tcrs = df_tcrs.drop_duplicates()
        df_tcrs["Label"] = 1
        if df_tcrs.shape[0] == 1:
            df_tcrs = pd.concat([df_tcrs] * 2).sort_index().reset_index(drop=True)
        df_tcrs.iat[0, df_tcrs.columns.get_loc("Label")] = 0
        return df_tcrs

    def get_base_cmd(self, filenames, tmp_folder, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        device = "cuda:0" if "cuda" not in kwargs else kwargs["cuda"]
        model = f"teinet_data"
        model = f"{self.repository_path}/models/{model}.pth"
        if not os.path.isfile(model):  # how to check better?
            raise ValueError(
                f"Please download model from git repository or "
                f"https://drive.google.com/file/d/12pVozHhRcGyMBgMlhcjgcclE3wlrVO32/view?usp=sharing."
                f" or specify the path to a model in 'model={kwargs['repository']}/models/'")
        cmd = f"predict.py --dset_path {filenames[0]} --save_prediction_path {filenames[1]} "
        cmd += f"--use_column CDR3.beta --model_path {model} --device {device}"
        return cmd

    def format_results(self, filenames, tmp_folder, tcrs, epitopes, pairwise, **kwargs):
        results_predictor = pd.read_csv(filenames[1], header=None, names=["Score", "Label"])
        input_predictor = pd.read_csv(filenames[0])
        joining_list = ["VDJ_cdr3", "Epitope"]
        results_predictor[joining_list] = input_predictor[["CDR3.beta", "Epitope"]]
        required_columns = ["VDJ_cdr3", "Epitope", "Score"]
        results_predictor = results_predictor[required_columns]
        results_predictor = results_predictor.drop_duplicates(subset=joining_list)
        df_out = self.transform_output(results_predictor, tcrs, epitopes, pairwise, joining_list)
        return df_out


class PanPep(ARepoTCRSpecificityPrediction):
    """
    Author: Gao et al.
    Paper: https://www.nature.com/articles/s42256-023-00619-3
    Repo: https://github.com/bm2-lab/PanPep
    """
    __name = "PanPep"
    __version = ""
    __tcr_length = (0, 9999)  # TODO no info in paper found
    __epitope_length = (0, 9999)  # TODO no info in paper found
    __organism = "H"
    __repo = "https://github.com/bm2-lab/PanPep.git"

    _rename_columns = {
        "VDJ_cdr3": "CDR3"
    }

    @property
    def name(self):
        return self.__name

    @property
    def version(self):
        return self.__version

    @property
    def tcr_length(self):
        return self.__tcr_length

    @property
    def epitope_length(self):
        return self.__epitope_length

    @property
    def repo(self):
        return self.__repo

    @property
    def organism(self):
        return self.__organism

    def format_tcr_data(self, tcrs, epitopes, pairwise, **kwargs):
        required_columns = list(self._rename_columns.values()) + ["Peptide"]
        df_tcrs = tcrs.to_pandas(rename_columns=self._rename_columns)
        if pairwise:
            df_tcrs = self.combine_tcrs_epitopes_pairwise(df_tcrs, epitopes)
        else:
            df_tcrs = self.combine_tcrs_epitopes_list(df_tcrs, epitopes)
        df_tcrs = df_tcrs.rename(columns={"Epitope": "Peptide"})
        df_tcrs = df_tcrs[(~df_tcrs["CDR3"].isna()) & (df_tcrs["CDR3"] != "")]
        df_tcrs = df_tcrs[required_columns]
        df_tcrs = df_tcrs.drop_duplicates()
        return df_tcrs

    def get_base_cmd(self, filenames, tmp_folder, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        return f"PanPep.py --learning_setting zero-shot --input {filenames[0]} --output {filenames[1]}"

    def format_results(self, filenames, tmp_folder, tcrs, epitopes, pairwise, **kwargs):
        results_predictor = pd.read_csv(filenames[1])
        joining_list = ["VDJ_cdr3", "Epitope"]
        results_predictor = results_predictor.rename(columns={"CDR3": "VDJ_cdr3",
                                                              "Peptide": "Epitope"})
        required_columns = ["VDJ_cdr3", "Epitope", "Score"]
        results_predictor = results_predictor[required_columns]
        df_out = self.transform_output(results_predictor, tcrs, epitopes, pairwise, joining_list)
        return df_out


class DLpTCR(ARepoTCRSpecificityPrediction):
    """
    Author: Xu, Luo, Lin et al.
    Paper: https://pubmed.ncbi.nlm.nih.gov/34415016/
    Repo: https://github.com/JiangBioLab/DLpTCR
    """
    __name = "DLpTCR"
    __version = ""
    __trc_length = (8, 20)
    __epitope_length = (9, 9)
    __organism = "H"
    __repo = "https://github.com/JiangBioLab/DLpTCR"
    _oldwdir = ""

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

    @property
    def repo(self):
        return self.__repo

    @property
    def organism(self):
        return self.__organism

    def format_tcr_data(self, tcrs, epitopes, pairwise, **kwargs):
        rename_columns = {
            "VJ_cdr3": "TCRA_CDR3",
            "VDJ_cdr3": "TCRB_CDR3"
        }
        required_columns = list(rename_columns.values()) + ["Epitope"]
        df_tcrs = tcrs.to_pandas(rename_columns=rename_columns)
        if pairwise:
            df_tcrs = self.combine_tcrs_epitopes_pairwise(df_tcrs, epitopes)
        else:
            df_tcrs = self.combine_tcrs_epitopes_list(df_tcrs, epitopes)
        model_type = "B" if "model_type" not in kwargs else kwargs["model_type"]
        if model_type == "B":
            df_tcrs = self.filter_by_length(df_tcrs, None, "TCRB_CDR3", "Epitope")
            df_tcrs = df_tcrs[(~df_tcrs["TCRB_CDR3"].isna()) & (df_tcrs["TCRB_CDR3"] != "")]
            prediction_columns = ["TCRB_CDR3", "Epitope"]
        elif model_type == "A":
            df_tcrs = self.filter_by_length(df_tcrs, "TCRA_CDR3", None, "Epitope")
            df_tcrs = df_tcrs[(~df_tcrs["TCRA_CDR3"].isna()) & (df_tcrs["TCRA_CDR3"] != "")]
            prediction_columns = ["TCRA_CDR3", "Epitope"]
        elif model_type == "AB":
            df_tcrs = self.filter_by_length(df_tcrs, "TCRA_CDR3", "TCRB_CDR3", "Epitope")
            df_tcrs = df_tcrs[(~df_tcrs["TCRB_CDR3"].isna()) & (df_tcrs["TCRB_CDR3"] != "")]
            df_tcrs = df_tcrs[(~df_tcrs["TCRA_CDR3"].isna()) & (df_tcrs["TCRA_CDR3"] != "")]
            prediction_columns = ["TCRB_CDR3", "TCRA_CDR3", "Epitope"]
        else:
            raise ValueError(f"Incorrect {model_type}. Please specify a correct model_type: A, B or AB")
        df_tcrs = df_tcrs[required_columns]
        df_tcrs.drop_duplicates(subset=prediction_columns, inplace=True, keep="first")
        df_tcrs = df_tcrs.reset_index(drop=True)
        return df_tcrs

    def save_tmp_files(self, data, **kwargs):
        tmp_folder = self.get_tmp_folder_path()
        model_type = "B" if "model_type" not in kwargs else kwargs["model_type"]
        path_in = os.path.join(tmp_folder.name, f"{self.name}_input.xlsx")
        path_out = os.path.join(tmp_folder.name, f"TCR{model_type}_pred.csv")
        data.to_excel(path_in)
        return [path_in, path_out], tmp_folder

    def get_base_cmd(self, filenames, tmp_folder, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        model_type = "B" if "model_type" not in kwargs else kwargs["model_type"]
        cmd_epitope = ["import sys",
                       f"sys.path.append('{self.repository_path}/code')",
                       "from Model_Predict_Feature_Extraction import *",
                       "from DLpTCR_server import *",
                       f"error_info,TCRA_cdr3,TCRB_cdr3,Epitope = deal_file('{filenames[0]}', "
                       f"'{tmp_folder.name}/', '{model_type}')",
                       f"output_file_path = save_outputfile('{tmp_folder.name}/', '{model_type}', '{filenames[0]}',"
                       f"TCRA_cdr3,TCRB_cdr3,Epitope)"
                       ]
        cmd_epitope = f'python -c "{"; ".join(cmd_epitope)}"'
        return cmd_epitope

    def run_exec_cmd(self, cmd, filenames, interpreter=None, conda=None, cmd_prefix=None, repository="", **kwargs):
        old_dir = os.getcwd()
        os.chdir(repository)

        cmds = []
        if cmd_prefix is not None:
            cmds.append(cmd_prefix)
        cmd_conda = ""
        if conda:
            if sys.platform.startswith("win"):
                cmd_conda = f"conda activate {conda} &&"
            else:
                cmd_conda = f"conda run -n {conda}"
        cmds.append(f"{cmd_conda} {cmd}")
        self.exec_cmd(" && ".join(cmds), filenames[1])
        os.chdir(old_dir)

    def format_results(self, filenames, tmp_folder, tcrs, epitopes, pairwise, **kwargs):
        model_type = "B" if "model_type" not in kwargs else kwargs["model_type"]
        if model_type == "B":
            results_predictor = pd.read_csv(filenames[1], header=0,
                                            names=["Index", "CDR3", "Epitope", "Predict", "Score"])
            joining_list = ["VDJ_cdr3", "Epitope"]
            required_columns = ["VDJ_cdr3", "Epitope", "Score"]
            results_predictor = results_predictor.rename(columns={"CDR3": "VDJ_cdr3"})
        elif model_type == "A":
            results_predictor = pd.read_csv(filenames[1], header=0,
                                            names=["Index", "CDR3", "Epitope", "Predict", "Score"])
            joining_list = ["VJ_cdr3", "Epitope"]
            required_columns = ["VJ_cdr3", "Epitope", "Score"]
            results_predictor = results_predictor.rename(columns={"CDR3": "VJ_cdr3"})
        else:
            results_predictor = pd.read_csv(filenames[1], header=0,
                                            names=["Index", "VJ_cdr3", "VDJ_cdr3", "Epitope", "Predict", "ScoreA",
                                                   "ScoreB"])
            results_predictor["Score"] = results_predictor["Predict"].str.split(" ").str[0]
            results_predictor["Score"].replace({"False": 0, "True": 1}, inplace=True)
            joining_list = ["VDJ_cdr3", "VJ_cdr3", "Epitope"]
            required_columns = ["VDJ_cdr3", "VJ_cdr3", "Epitope", "Score"]
        results_predictor = results_predictor[required_columns]
        results_predictor = results_predictor.drop_duplicates()
        df_out = self.transform_output(results_predictor, tcrs, epitopes, pairwise, joining_list)
        return df_out


class TULIP(ARepoTCRSpecificityPrediction):
    """
    Author: Meynard-Piganeau et al.
    Paper: https://www.biorxiv.org/content/10.1101/2023.07.19.549669v1.full.pdf
    Repo: https://github.com/barthelemymp/TULIP-TCR
    """
    __name = "TULIP-TCR"
    __version = ""
    __tcr_length = (0, 48)
    __epitope_length = (0, 48)
    __organism = "HM"
    __repo = "https://github.com/barthelemymp/TULIP-TCR.git"

    _rename_columns = {
        "VDJ_cdr3": "CDR3b",
        "VJ_cdr3": "CDR3a"
    }

    @property
    def name(self):
        return self.__name

    @property
    def version(self):
        return self.__version

    @property
    def tcr_length(self):
        return self.__tcr_length

    @property
    def epitope_length(self):
        return self.__epitope_length

    @property
    def repo(self):
        return self.__repo

    @property
    def organism(self):
        return self.__organism

    def format_tcr_data(self, tcrs, epitopes, pairwise, **kwargs):
        required_columns = list(self._rename_columns.values()) + ["peptide", "MHC"]
        df_tcrs = tcrs.to_pandas(rename_columns=self._rename_columns)
        if pairwise:
            df_tcrs = self.combine_tcrs_epitopes_pairwise(df_tcrs, epitopes)
        else:
            df_tcrs = self.combine_tcrs_epitopes_list(df_tcrs, epitopes)
        df_tcrs = df_tcrs.rename(columns={"Epitope": "peptide"})
        df_tcrs = df_tcrs[required_columns]
        df_tcrs["MHC"] = df_tcrs["MHC"].fillna("<MIS>")
        df_tcrs["MHC"] = df_tcrs["MHC"].replace("", "<MIS>")
        df_tcrs["CDR3a"] = df_tcrs["CDR3a"].fillna("<MIS>")
        df_tcrs["CDR3a"] = df_tcrs["CDR3a"].replace("", "<MIS>")
        df_tcrs["CDR3b"] = df_tcrs["CDR3b"].fillna("<MIS>")
        df_tcrs["CDR3b"] = df_tcrs["CDR3b"].replace("", "<MIS>")
        df_tcrs = df_tcrs.drop_duplicates().reset_index()
        df_tcrs["binder"] = 1
        return df_tcrs

    def get_base_cmd(self, filenames, tmp_folder, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        model = f"{self.repository_path}/model_weights/pytorch_model.bin" if "model" not in kwargs else {
            kwargs["model"]}
        config = f"{self.repository_path}/configs/shallow.config.json"
        return f"predict.py --test_dir {filenames[0]} --modelconfig {config} --load {model} --output {tmp_folder.name}/ >> {filenames[1]}"

    def format_results(self, filenames, tmp_folder, tcrs, epitopes, pairwise, **kwargs):
        csv_files = list(filter(lambda f: f.endswith(".csv"), os.listdir(tmp_folder.name)))
        csv_files.remove(f"{self.name}_input.csv")
        csv_files.remove(f"{self.name}_output.csv")
        result_list = []
        for file in csv_files:
            result_list.append(pd.read_csv(os.path.join(tmp_folder.name, file)))
        results_predictor = pd.concat(result_list, ignore_index=True)
        results_predictor = results_predictor.fillna("")
        joining_list = ["VJ_cdr3", "VDJ_cdr3", "Epitope", "MHC"]
        results_predictor = results_predictor.rename(columns={"CDR3b": "VDJ_cdr3",
                                                              "CDR3a": "VJ_cdr3",
                                                              "peptide": "Epitope",
                                                              "score": "Score"})
        results_predictor["MHC"] = results_predictor["MHC"].replace("<MIS>", None)
        results_predictor["VJ_cdr3"] = results_predictor["VJ_cdr3"].replace("<MIS>", "")
        results_predictor["VDJ_cdr3"] = results_predictor["VDJ_cdr3"].replace("<MIS>", "")
        required_columns = joining_list + ["Score"]
        results_predictor = results_predictor[required_columns]
        results_predictor = results_predictor.drop_duplicates(subset=joining_list)
        df_out = self.transform_output(results_predictor, tcrs, epitopes, pairwise, joining_list)
        return df_out

    def run_exec_cmd(self, cmd, filenames, interpreter=None, conda=None, cmd_prefix=None, repository="", **kwargs):
        if repository is not None and repository != "" and os.path.isdir(repository):
            self.correct_code(repository)
        super().run_exec_cmd(cmd, filenames, interpreter, conda, cmd_prefix, repository)

    def correct_code(self, path_repo):
        """
        The github repo does not provide full functionality, it will be corrected here.
        """
        script = []
        changed = 0
        with open(os.path.join(path_repo, "predict.py"), "r") as f:
            script.extend(f.readlines())
        # change output
        if '        results["rank"] = ranks\n' in script:
            idx = script.index('        results["rank"] = ranks\n')
            script[idx] = '        results["MHC"] = datasetPetideSpecific.MHC\n'
            changed = 1
        # remove auc calculation
        if "        auce = roc_auc_score(datasetPetideSpecific.binder, ranks)\n" in script:
            script.remove("        auce = roc_auc_score(datasetPetideSpecific.binder, ranks)\n")
            changed = 1
        if "        print(auce)\n" in script:
            script.remove("        print(auce)\n")
            changed = 1
        if changed == 1:
            with open(os.path.join(path_repo, "predict.py"), "w") as f:
                f.writelines(script)


class iTCep(ARepoTCRSpecificityPrediction):
    """
    Author: Zhang et al.
    Paper: https://pubmed.ncbi.nlm.nih.gov/37229205/
    Repo: https://github.com/kbvstmd/iTCep
    """
    __name = "iTCep"
    __version = ""
    __tcr_length = (8, 21)
    __epitope_length = (8, 11)
    __organism = "H"
    __repo = "https://github.com/kbvstmd/iTCep.git"

    _rename_columns = {
        "VDJ_cdr3": "CDR3"
    }

    @property
    def name(self):
        return self.__name

    @property
    def version(self):
        return self.__version

    @property
    def tcr_length(self):
        return self.__tcr_length

    @property
    def epitope_length(self):
        return self.__epitope_length

    @property
    def repo(self):
        return self.__repo

    @property
    def organism(self):
        return self.__organism

    def format_tcr_data(self, tcrs, epitopes, pairwise, **kwargs):
        required_columns = list(self._rename_columns.values()) + ["peptide"]
        df_tcrs = tcrs.to_pandas(rename_columns=self._rename_columns)
        if pairwise:
            df_tcrs = self.combine_tcrs_epitopes_pairwise(df_tcrs, epitopes)
        else:
            df_tcrs = self.combine_tcrs_epitopes_list(df_tcrs, epitopes)
        df_tcrs = df_tcrs.rename(columns={"Epitope": "peptide"})
        df_tcrs = self.filter_by_length(df_tcrs, None, "CDR3", "peptide")
        df_tcrs = df_tcrs[(~df_tcrs["CDR3"].isna()) & (df_tcrs["CDR3"] != "")]
        df_tcrs = df_tcrs[required_columns]
        df_tcrs = df_tcrs.drop_duplicates()
        return df_tcrs

    def get_base_cmd(self, filenames, tmp_folder, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        return f"predict.py --input {filenames[0]} --output {filenames[1]}"

    def format_results(self, filenames, tmp_folder, tcrs, epitopes, pairwise, **kwargs):
        results_predictor = pd.read_csv(filenames[1])
        joining_list = ["VDJ_cdr3", "Epitope"]
        results_predictor = results_predictor.rename(columns={"CDR3": "VDJ_cdr3",
                                                              "peptide": "Epitope",
                                                              "Probability": "Score"})
        required_columns = ["VDJ_cdr3", "Epitope", "Score"]
        results_predictor = results_predictor[required_columns]
        df_out = self.transform_output(results_predictor, tcrs, epitopes, pairwise, joining_list)
        return df_out


class NetTCR22(ARepoTCRSpecificityPrediction):
    """
    Author: Fynbo Jensen, Nielsen
    Paper: https://www.biorxiv.org/content/10.1101/2023.10.12.562001v1.full
    Repo: https://github.com/mnielLab/NetTCR-2.2
    """
    __name = "NetTCR"
    __version = "2.2"
    __tcr_length = (1, 9999)
    __epitope_length = (1, 12)
    __organism = "H"
    __repo = "https://github.com/mnielLab/NetTCR-2.2.git"

    _rename_columns = {
        "VDJ_cdr3": "cdr3_beta_aa",
        "VDJ_v_gene": "TRBV_IMGT",
        "VDJ_j_gene": "TRBJ_IMGT",
        "VJ_cdr3": "cdr3_alpha_aa",
        "VJ_v_gene": "TRAV_IMGT",
        "VJ_j_gene": "TRAJ_IMGT",
    }

    @property
    def name(self):
        return self.__name

    @property
    def version(self):
        return self.__version

    @property
    def tcr_length(self):
        return self.__tcr_length

    @property
    def epitope_length(self):
        return self.__epitope_length

    @property
    def repo(self):
        return self.__repo

    @property
    def organism(self):
        return self.__organism

    def format_tcr_data(self, tcrs, epitopes, pairwise, **kwargs):
        df_tcrs = tcrs.to_pandas(rename_columns=self._rename_columns)
        if pairwise:
            df_tcrs = self.combine_tcrs_epitopes_pairwise(df_tcrs, epitopes)
        else:
            df_tcrs = self.combine_tcrs_epitopes_list(df_tcrs, epitopes)
        df_tcrs = df_tcrs.rename(columns={"Epitope": "peptide"})
        for col in self._rename_columns.values():
            df_tcrs = df_tcrs[(~df_tcrs[col].isna()) & (df_tcrs[col] != 'nan') & (df_tcrs[col] != "")]
        df_tcrs = df_tcrs[(~df_tcrs["organism"].isna()) & (df_tcrs["organism"] != 'nan') & (df_tcrs["organism"] != "")]
        df_tcrs = self.filter_epitopes(df_tcrs, **kwargs)
        df_tcrs = df_tcrs.drop_duplicates()
        df_tcrs["binder"] = 0
        return df_tcrs

    def filter_epitopes(self, df, repository=None, **kwargs):
        models = os.listdir(os.path.join(repository, "models", "nettcr_2_2_pretrained"))
        models = [el for el in models if el not in ["negative_controls", "cv_pred_df.csv"]]
        mask_epitope = df["peptide"].isin(models).values
        delta = len(df) - sum(mask_epitope)
        if delta > 0:
            warnings.warn(f"Filtering {delta} rows as Epitope not available for categorical model")
        df = df[mask_epitope].copy()
        return df

    def save_tmp_files(self, data, **kwargs):
        tmp_folder = self.get_tmp_folder_path()
        path_in_raw = os.path.join(tmp_folder.name, f"{self.name}_raw_input.csv")
        path_in_intermediate = os.path.join(tmp_folder.name, f"{self.name}_intermediate_input.csv")
        path_in = os.path.join(tmp_folder.name, f"{self.name}_input.csv")
        model = "t.0.v.1" if "model" not in kwargs else kwargs["model"]
        model_type = kwargs.get("model_type", "pan")
        path_out = os.path.join(f"{self.repository_path}", f"models/nettcr_2_2_{model_type}", f"{model}_prediction.csv")
        data.to_csv(path_in_raw)
        return [path_in_raw, path_in_intermediate, path_in, path_out], tmp_folder

    def get_base_cmd(self, filenames, tmp_folder, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        path_utils = os.path.dirname(__file__)
        model = kwargs.get("model", "t.0.v.1")
        model_type = kwargs.get("model_type", "pan")
        modeldir = f"{self.repository_path}/models/nettcr_2_2_{model_type}"
        cmd_reconstruct = f"{path_utils}/Utils.py nettcr {filenames[0]} {filenames[1]} {filenames[2]}"
        cmd_predict = f"{self.repository_path}/src/predict.py --test_data {filenames[2]} --outdir {modeldir} --model_name {model} --model_type {model_type}"
        return [cmd_reconstruct, cmd_predict]

    def run_exec_cmd(self, cmd, filenames, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        super().run_exec_cmd(cmd[0], [None, filenames[2]], interpreter, conda, cmd_prefix, m_cmd=False, **kwargs)
        super().run_exec_cmd(cmd[1], [None, filenames[3]], interpreter, conda, cmd_prefix, m_cmd=False, **kwargs)

    def format_results(self, filenames, tmp_folder, tcrs, epitopes, pairwise, **kwargs):
        results_predictor = pd.read_csv(filenames[3])
        results_predictor = results_predictor.fillna("")
        joining_list = ["VJ_cdr3", "VDJ_cdr3", "VDJ_v_gene", "VDJ_j_gene", "VJ_v_gene", "VJ_j_gene", "Epitope"]
        results_predictor = results_predictor.rename(columns={"cdr3_beta_aa": "VDJ_cdr3",
                                                              "TRBV_IMGT": "VDJ_v_gene",
                                                              "TRBJ_IMGT": "VDJ_j_gene",
                                                              "cdr3_alpha_aa": "VJ_cdr3",
                                                              "TRAV_IMGT": "VJ_v_gene",
                                                              "TRAJ_IMGT": "VJ_j_gene",
                                                              "peptide": "Epitope",
                                                              "prediction": "Score"})
        required_columns = joining_list + ["Score"]
        results_predictor = results_predictor[required_columns]
        df_out = self.transform_output(results_predictor, tcrs, epitopes, pairwise, joining_list)
        return df_out


class MixTCRpred(ARepoTCRSpecificityPrediction):
    """
    Author: Croce et al.
    Paper: https://www.frontiersin.org/articles/10.3389/fimmu.2022.893247/full
    Repo: https://github.com/GfellerLab/MixTCRpred
    """
    __name = "MixTCRpred"
    __version = ""
    __trc_length = (0, 20)  # TODO
    __epitope_length = (0, 999999)
    __organism = "HM"
    __repo = "https://www.nature.com/articles/s41467-024-47461-8"

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

    @property
    def repo(self):
        return self.__repo

    @property
    def organism(self):
        return self.__organism

    def format_tcr_data(self, tcrs, epitopes, pairwise, **kwargs):
        self.download_models(**kwargs)
        rename_columns = {
            "VJ_cdr3": "cdr3_TRA",
            "VDJ_cdr3": "cdr3_TRB",
            "VJ_v_gene": "TRAV",
            "VJ_j_gene": "TRAJ",
            "VDJ_v_gene": "TRBV",
            "VDJ_j_gene": "TRBJ"
        }
        df_tcrs = tcrs.to_pandas(rename_columns=rename_columns)
        if pairwise:
            df_tcrs = self.combine_tcrs_epitopes_pairwise(df_tcrs, epitopes)
        else:
            df_tcrs = self.combine_tcrs_epitopes_list(df_tcrs, epitopes)
        df_tcrs = df_tcrs[list(rename_columns.values()) + ["Epitope", "MHC"]]
        df_tcrs = self.filter_by_length(df_tcrs, "cdr3_TRA", "cdr3_TRB", None)
        for el in rename_columns.values():
            df_tcrs = df_tcrs[(~df_tcrs[el].isna()) & (df_tcrs[el] != "")]
        df_tcrs = self.filter_epitopes(df_tcrs, **kwargs)
        df_tcrs = df_tcrs.drop_duplicates()
        return df_tcrs

    def download_models(self, **kwargs):
        path_models = os.path.join(kwargs["repository"], "pretrained_models")
        n_files = len([f for f in os.listdir(path_models)])
        if n_files >= 148:
            return
        cmd = f"MixTCRpred.py --download_all"
        path_file = os.path.join(path_models, "model_A0101_ATDALMTGF.ckpt")
        try:
            super().run_exec_cmd(cmd, path_file, **kwargs)
        except Exception as e:
            if "urllib.error.HTTPError: HTTP Error 404: NOT FOUND" not in str(e):
                raise 

    def filter_epitopes(self, df, repository=None, **kwargs):
        def epitope_contained(row):
            epitope = row["Epitope"]
            mhc = row["MHC"].name
            mhc = mhc.replace("*", "")
            mhc = mhc.replace(":", "")
            name = f"{mhc}_{epitope}"
            return name in models

        models = os.listdir(os.path.join(repository, "pretrained_models"))
        models = [el.split("model_")[1].split(".ckpt")[0] for el in models if el.endswith(".ckpt")]
        mask_epitope = df.apply(epitope_contained, axis=1).values
        delta = len(df) - sum(mask_epitope)
        if delta > 0:
            warnings.warn(f"Filtering {delta} rows as Epitope not available for categorical model")
        df = df[mask_epitope].copy()
        return df
        
                

    def save_tmp_files(self, data, **kwargs):
        tmp_folder = self.get_tmp_folder_path()
        
        paths = []
        for _, row in data[["Epitope", "MHC"]].drop_duplicates().iterrows():
            epitope = row["Epitope"]
            mhc = row["MHC"]

            data_epitope = data[(data["Epitope"] == epitope) & (data["MHC"] == mhc)]
            cols = ["cdr3_TRA", "cdr3_TRB", "TRAV", "TRAJ", "TRBV", "TRBJ"]
            data_epitope = data_epitope[cols].copy()

            mhc = mhc.name.replace("*", "").replace(":", "").replace("H-2", "H2")

            path_in = os.path.join(tmp_folder.name, f"input_{mhc}_{epitope}.csv")
            path_out = os.path.join(tmp_folder.name, f"output_{mhc}_{epitope}.csv")

            paths.append(path_in)
            paths.append(path_out)
            data_epitope.to_csv(path_in)
        return paths, tmp_folder

    def get_base_cmd(self, filenames, tmp_folder, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        batch_size = kwargs.get("batch_size", 1)
        cmds = []
        for i in range(0, len(filenames), 2):
            model = filenames[i].split("input_")[1].split(".csv")[0]
            cmd = f"MixTCRpred.py --model {model} --input {filenames[i]} --output {filenames[i + 1]} " \
                  f"--batch_size {batch_size}"
            cmds.append(cmd)
        return cmds

    def run_exec_cmd(self, cmd, filenames, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        for i in range(0, len(filenames), 2):
            super().run_exec_cmd(cmd[i // 2], [filenames[i], filenames[i + 1]], interpreter,
                                 conda, cmd_prefix, m_cmd=False, **kwargs)

    def format_results(self, filenames, tmp_folder, tcrs, epitopes, pairwise, **kwargs):
        import re
        results_joined = []
        for i in range(0, len(filenames), 2):
            path_out = filenames[i+1]
            results_predictor = pd.read_csv(path_out, index_col=0, comment="#")
            results_predictor.index.name = None
            results_predictor = results_predictor.fillna("")
            results_predictor = results_predictor.rename(columns={"cdr3_TRB": "VDJ_cdr3",
                                                                  "TRBV": "VDJ_v_gene",
                                                                  "TRBJ": "VDJ_j_gene",
                                                                  "cdr3_TRA": "VJ_cdr3",
                                                                  "TRAV": "VJ_v_gene",
                                                                  "TRAJ": "VJ_j_gene",
                                                                  "perc_rank": "Score"})
            results_predictor["Score"] = 1 - results_predictor["Score"]/100
            epitope = path_out.split("_")[-1].split(".csv")[0] 
            mhc = path_out.split("_")[-2] 
            mhc = mhc.replace("H2", "H-2")
            if mhc[0] != "H":
                match = re.match(r"([A-Za-z]+)(\d+)", mhc)
                letters, numbers = match.groups()
                mhc = f"HLA-{letters}*{numbers[:-2]}:{numbers[-2:]}"
            results_predictor["Epitope"] = epitope
            results_predictor["MHC"] = mhc
            results_joined.append(results_predictor)
        results_joined = pd.concat(results_joined, axis=0)

        joining_list = ["VJ_cdr3", "VDJ_cdr3", "VDJ_v_gene", "VDJ_j_gene", "VJ_v_gene", "VJ_j_gene", "Epitope", "MHC"]
        required_columns = joining_list + ["Score"]
        results_joined = results_joined[required_columns]
        df_out = self.transform_output(results_joined, tcrs, epitopes, pairwise, joining_list)
        return df_out


class TCRGP(ARepoTCRSpecificityPrediction):
    """
    Author: Jokinen et al.
    Paper: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008814
    Repo: https://github.com/emmijokinen/TCRGP
    """
    __name = "TCRGP"
    __version = ""
    __trc_length = (0, 99999999)  # TODO
    __epitope_length = (0, 999999)
    __organism = "HM"
    __repo = "https://github.com/emmijokinen/TCRGP"

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

    @property
    def repo(self):
        return self.__repo

    @property
    def organism(self):
        return self.__organism

    def format_tcr_data(self, tcrs, epitopes, pairwise, **kwargs):
        rename_columns = {
            "VDJ_cdr3": "cdr3b",
        }
        df_tcrs = tcrs.to_pandas(rename_columns=rename_columns)
        if pairwise:
            df_tcrs = self.combine_tcrs_epitopes_pairwise(df_tcrs, epitopes)
        else:
            df_tcrs = self.combine_tcrs_epitopes_list(df_tcrs, epitopes)
        df_tcrs = df_tcrs[list(rename_columns.values()) + ["Epitope", "MHC"]]
        df_tcrs = self.filter_by_length(df_tcrs, None, "cdr3b", None)

        df_tcrs = df_tcrs[(~df_tcrs["cdr3b"].isna()) & (df_tcrs["cdr3b"] != "")]

        df_tcrs = self.filter_epitopes(df_tcrs, **kwargs)
        df_tcrs = df_tcrs.drop_duplicates()
        return df_tcrs


    def filter_epitopes(self, df, repository=None, **kwargs):
        models = os.listdir(os.path.join(repository, "models", "paper"))
        models = [el.split("_")[2] for el in models if "model" in el]
        mask_epitope = df["Epitope"].isin(models).values 
        delta = len(df) - sum(mask_epitope)
        if delta > 0:
            warnings.warn(f"Filtering {delta} rows as Epitope not available for categorical model")
        df = df[mask_epitope].copy()
        return df

    def save_tmp_files(self, data, **kwargs):
        tmp_folder = self.get_tmp_folder_path()
        
        paths = []
        for _, row in data[["Epitope", "MHC"]].drop_duplicates().iterrows():
            epitope = row["Epitope"]

            data_epitope = data[data["Epitope"] == epitope]
            cols = ["cdr3b"]
            data_epitope = data_epitope[cols].copy()

            path_in = os.path.join(tmp_folder.name, f"input_{epitope}.csv")
            path_out = os.path.join(tmp_folder.name, f"output_{epitope}.csv")

            paths.append(path_in)
            paths.append(path_out)
            data_epitope.to_csv(path_in)
        return paths, tmp_folder

    def get_base_cmd(self, filenames, tmp_folder, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        path_utils = os.path.dirname(__file__)
        repository = kwargs["repository"]
        cmds = []
        for i in range(0, len(filenames), 2):
            model = filenames[i].split("input_")[1].split(".csv")[0]
            cmd = f"{path_utils}/Utils.py tcrgp {repository} {filenames[i]} {filenames[i+1]}"
            cmds.append(cmd)
        return cmds

    def run_exec_cmd(self, cmd, filenames, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        for i in range(0, len(filenames), 2):
            super().run_exec_cmd(cmd[i // 2], [filenames[i], filenames[i + 1]], interpreter,
                                 conda, cmd_prefix, m_cmd=False, **kwargs)

    def format_results(self, filenames, tmp_folder, tcrs, epitopes, pairwise, **kwargs):
        results_joined = []
        for i in range(0, len(filenames), 2):
            path_out = filenames[i+1]
            results_predictor = pd.read_csv(path_out, index_col=0)
            results_predictor = results_predictor.fillna("")
            results_predictor = results_predictor.rename(columns={"cdr3b": "VDJ_cdr3"})
            results_joined.append(results_predictor)
        results_joined = pd.concat(results_joined, axis=0)

        joining_list = ["VDJ_cdr3", "Epitope"]
        required_columns = joining_list + ["Score"]
        results_joined = results_joined[required_columns]
        df_out = self.transform_output(results_joined, tcrs, epitopes, pairwise, joining_list)
        return df_out
