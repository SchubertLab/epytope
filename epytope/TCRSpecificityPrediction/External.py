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
        rename_columns = {
            "VJ_cdr3": "TRA",
            "VDJ_cdr3": "TRB",
            "VJ_v_gene": "TRAV",
            "VJ_j_gene": "TRAJ",
            "VDJ_v_gene": "TRBV",
            "VDJ_j_gene": "TRBJ",
            "celltype": "T-Cell-Type",
        }
        required_columns = list(rename_columns.values()) + ["Peptide", "MHC"]
        df_tcrs = tcrs.to_pandas(rename_columns=rename_columns)

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

    def format_results(self, filenames, tcrs, pairwise):
        results_predictor = pd.read_csv(filenames[1])
        results_predictor["MHC"] = results_predictor["MHC"].fillna("")
        df_out = TCRSpecificityPredictionResult.from_output(results_predictor, tcrs, pairwise, self.name)
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
        return df_tcrs

    def save_tmp_files(self, data):
        filenames, tmp_dir = super().save_tmp_files(data)
        filenames[1] = os.sep.join(filenames[1].split(os.sep)[:-1])
        filenames.append(f"{tmp_dir.name}/{self.name}_logs.log")
        return filenames, tmp_dir

    def get_base_cmd(self, filenames, tmp_folder, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        repository = kwargs["repository"]
        cmd = f"pMTnet.py -input {filenames[0]} -library {repository}/library " \
              f"-output {filenames[1]} -output_log {filenames[2]}"
        return cmd

    def format_results(self, filenames, tcrs, pairwise):
        results_predictor = pd.read_csv(f"{filenames[1]}/prediction.csv")
        results_predictor = results_predictor.rename(columns={"Rank": "Score", "HLA": "MHC", "Antigen": "Peptide"})
        results_predictor["Score"] = 1 - results_predictor["Score"]  # in github: lower rank = good prediction => invert
        df_out = TCRSpecificityPredictionResult.from_output(results_predictor, tcrs, pairwise, self.name)
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
        return df_tcrs

    def save_tmp_files(self, data):
        filenames, tmp_dir = super().save_tmp_files(data)
        filenames.append(f"{tmp_dir.name}/{self.name}_logs.log")
        return filenames, tmp_dir

    def get_base_cmd(self, filenames, tmp_folder, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        repository = kwargs["repository"]
        cmd = f"pMTnet.py -input {filenames[0]} -library {repository}/library " \
              f"-output {filenames[1]} -output_log {filenames[2]}"
        return cmd

    def format_results(self, filenames, tcrs, pairwise):
        results_predictor = pd.read_csv(filenames[1])
        df_out = TCRSpecificityPredictionResult.from_output(results_predictor, tcrs, pairwise, self.name)
        return df_out

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

    def prepare_dataset(self, df: pd.DataFrame, filename: str = None, chain: str = "b"):
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