# This code is part of the epytope distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
.. module:: TCRSpecificityPrediction
   :synopsis: This module contains all classes for external TCR specificity prediction methods.
.. moduleauthor:: albahah, drost
"""

import abc
import os
import shutil

import pandas as pd

from epytope.TCRSpecificityPrediction.ML import ACmdTCRSpecificityPrediction
from epytope.Core.Result import TCRSpecificityPredictionResult


class ARepoTCRSpecificityPrediction(ACmdTCRSpecificityPrediction):
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

    def run_exec_cmd(self, cmd, filenames, interpreter=None, conda=None, cmd_prefix=None, repository="", **kwargs):
        os.chdir(repository)

        interpreter = "python" if interpreter is None else interpreter
        cmds = []
        if cmd_prefix is not None:
            cmds.append(cmd_prefix)
        if conda is not None:
            cmds.append(f"conda activate {conda}")
        cmds.append(f"{interpreter} {repository}/{cmd}")
        self.exec_cmd(" && ".join(cmds), filenames[1])


class Ergo2(ARepoTCRSpecificityPrediction):
    """
    Author: Springer et al.
    Paper: https://www.frontiersin.org/articles/10.3389/fimmu.2021.664514/full
    Repo: https://github.com/IdoSpringer/ERGO-II
    """
    __name = "ERGO-II"
    __version = ""
    __tcr_length = (0, 30)  # TODO
    __epitope_length = (0, 30)  # TODO
    __repo = "https://github.com/IdoSpringer/ERGO-II.git"

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

    def format_tcr_data(self, tcrs, epitopes, pairwise):
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
            os.chdir(repository)
            self.correct_code(repository)
        super().run_exec_cmd(cmd, filenames, interpreter, conda, cmd_prefix, repository)

    def format_results(self, filenames, tcrs, epitopes, pairwise):
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
            with open("Predict.py", "w") as f:
                f.writelines(script)

        # Cpu + gpu usable
        script = []
        with open(os.path.join(path_repo, "Models.py"), "r") as f:
            script.extend(f.readlines())
        if "        checkpoint = torch.load(ae_file)\n" in script:
            idx = script.index("        checkpoint = torch.load(ae_file)\n")
            script[idx] = "        checkpoint = torch.load(ae_file, " \
                          "map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))\n"
            with open("Models.py", "w") as f:
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
    __trc_length = (0, 40) # todo
    __epitope_length = (0, 40) # todo
    __cmd = "pMTnet.py"
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

    def input_check(self, tcrs, epitopes, pairwise, **kwargs):
        super().input_check(tcrs, epitopes, pairwise, **kwargs)
        allowed_alleles = ["A*", "B*", "C*", "E*"]
        for epitope in epitopes:
            if epitope.allele is None:
                raise ValueError("Missing MHC-Annotation: pMTnet requires MHC information")
            if epitope.allele.name[:2] not in allowed_alleles:
                raise ValueError(f"Invalid MHC {epitope.allele}: pMTnet requires MHC out of {allowed_alleles}")

    def format_tcr_data(self, tcrs, epitopes, pairwise):
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
        repository = kwargs["repository"]
        cmd = f"pMTnet.py -input {filenames[0]} -library {repository}/library " \
              f"-output {filenames[1]} -output_log {filenames[2]}"
        return cmd

    def format_results(self, filenames, tcrs, epitopes, pairwise):
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
    __repo = "https://github.com/ddiem-ri-4D/epiTCR.git"

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

    def format_tcr_data(self, tcrs, epitopes, pairwise):
        rename_columns = {
            "VDJ_cdr3": "CDR3b",
        }
        required_columns = list(rename_columns.values()) + ["epitope", "HLA", "binder"]
        df_tcrs = tcrs.to_pandas(rename_columns=rename_columns)
        if pairwise:
            df_tcrs = self.combine_tcrs_epitopes_pairwise(df_tcrs, epitopes)
        else:
            df_tcrs = self.combine_tcrs_epitopes_list(df_tcrs, epitopes)
        df_tcrs = df_tcrs.rename(columns={"Epitope": "epitope", "MHC": "HLA"})  # add MHC sequence?
        df_tcrs = self.filter_by_length(df_tcrs, None, "CDR3b", "epitope")
        df_tcrs = df_tcrs[(~df_tcrs["CDR3b"].isna()) & (df_tcrs["CDR3b"] != "")]
        df_tcrs["binder"] = 1
        df_tcrs.iat[0, df_tcrs.columns.get_loc("binder")] = 0
        df_tcrs["HLA"] = df_tcrs["HLA"].str[4:]  # TODO test what happens if no HLA is provided / do we need HLA?
        df_tcrs = df_tcrs[required_columns]
        df_tcrs.drop_duplicates(inplace=True, keep="first")
        return df_tcrs

    def get_base_cmd(self, filenames, tmp_folder, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        model = "rdforestWithoutMHCModel"
        if "model" in kwargs:
            model = kwargs["model"]
        repository = kwargs["repository"]
        model_filepath = os.path.join(repository, "models", f"{model}.pickle")
        if not os.path.exists(model_filepath):
            raise TypeError(f"Please unzip the models stored at {model_filepath}.zip to {model_filepath}")
        return f"predict.py --testfile {filenames[0]} --modelfile {model_filepath} --chain ce >> {filenames[1]}"

    def run_exec_cmd(self, cmd, filenames, interpreter=None, conda=None, cmd_prefix=None, repository="", **kwargs):
        super().run_exec_cmd(cmd, filenames, interpreter, conda, cmd_prefix, repository)

    def format_results(self, filenames, tcrs, epitopes, pairwise):
        results_predictor = pd.read_csv(filenames[1], skiprows=15, index_col=False)
        results_predictor = results_predictor[:-1]
        results_predictor = results_predictor.rename(columns={"CDR3b": "VDJ_cdr3",
                                                              "epitope": "Epitope",
                                                              "predict_proba": "Score"})
        required_columns = ["VDJ_cdr3", "Epitope", "Score"]  # TODO: Does the model use MHC, if so add here!
        joining_list = ["VDJ_cdr3", "Epitope"]
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
    __trc_length = (0, 40) # todo
    __epitope_length = (0, 40) # todo
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

    def format_tcr_data(self, tcrs, epitopes, pairwise):
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
        paths[1] = os.path.join(kwargs["repository"], "result", "pred_original_ATM-TCR_input.csv")
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

    def format_results(self, filenames, tcrs, epitopes, pairwise):
        results_predictor = pd.read_csv(filenames[1], sep="\t", header=None)
        results_predictor.columns = ["Epitope", "VDJ_cdr3", "Label", "Binary", "Score"]
        results_predictor = results_predictor[["Epitope", "VDJ_cdr3", "Score"]]

        joining_list = ["VDJ_cdr3", "Epitope"]
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
    __tcr_length = (0, 30)  # TODO
    __epitope_length = (0, 30)  # TODO
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

    def format_tcr_data(self, tcrs, epitopes, pairwise):
        required_columns = list(self._rename_columns.values()) + ["epitope"]
        df_tcrs = tcrs.to_pandas(rename_columns=self._rename_columns)

        if pairwise:
            df_tcrs = self.combine_tcrs_epitopes_pairwise(df_tcrs, epitopes)
        else:
            df_tcrs = self.combine_tcrs_epitopes_list(df_tcrs, epitopes)
        df_tcrs = df_tcrs.rename(columns={"Epitope": "epitope"})

        df_tcrs = df_tcrs[required_columns]
        df_tcrs = self.filter_by_length(df_tcrs, None, "TRB", "epitope")
        df_tcrs = df_tcrs.drop_duplicates()
        df_tcrs = df_tcrs[(~df_tcrs["CDR3b"].isna()) & (df_tcrs["CDR3b"] != "")]
        return df_tcrs

    def get_base_cmd(self, filenames, tmp_folder, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        dataset = "vdjdb" if "dataset" not in kwargs else kwargs["dataset"]
        return f"Predict.py ERGO.py predict lstm dataset specific cpu --model_file=models/lstm_vdjdb1.pt --train_data_file=train_data --test_data_file={filenames[0]} >> {filenames[1]}"

    def run_exec_cmd(self, cmd, filenames, interpreter=None, conda=None, cmd_prefix=None, repository="", **kwargs):
        super().run_exec_cmd(cmd, filenames, interpreter, conda, cmd_prefix, repository)

    def format_results(self, filenames, tcrs, epitopes, pairwise):
        results_predictor = pd.read_csv(filenames[1], index_col=0)
        print(results_predictor)
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
            with open("Predict.py", "w") as f:
                f.writelines(script)

        # Cpu + gpu usable
        script = []
        with open(os.path.join(path_repo, "Models.py"), "r") as f:
            script.extend(f.readlines())
        if "        checkpoint = torch.load(ae_file)\n" in script:
            idx = script.index("        checkpoint = torch.load(ae_file)\n")
            script[idx] = "        checkpoint = torch.load(ae_file, " \
                          "map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))\n"
            with open("Models.py", "w") as f:
                f.writelines(script)

        # rename folders
        if os.path.isdir(os.path.join(path_repo, "Models", "AE")):
            shutil.move(os.path.join(path_repo, "Models", "AE"), os.path.join(path_repo, "TCR_Autoencoder"))

