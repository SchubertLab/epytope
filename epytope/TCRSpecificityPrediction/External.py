# This code is part of the epytope distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
.. module:: TCRSpecificityPrediction
   :synopsis: This module contains all classes for external TCR specificity prediction methods.
.. moduleauthor:: albahah
"""

import os
import shutil
import tempfile
import subprocess

import pandas as pd

from epytope.Core.Base import ATCRSpecificityPrediction
from epytope.Core.Result import TCRSpecificityPredictionResult
from epytope.Core.ImmuneReceptor import ImmuneReceptor
from epytope.Core.TCREpitope import TCREpitope
from epytope.IO.IRDatasetAdapter import IRDataset


class AExternalTCRSpecificityPrediction(ATCRSpecificityPrediction):
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
        self.run_exec_cmd(filenames, interpreter, conda, cmd_prefix, **kwargs)
        df_results = self.format_results(filenames, tcrs, pairwise)
        self.clean_up(tmp_folder, filenames)
        return df_results

    def format_tcr_data(self, tcrs, epitopes, pairwise):
        raise NotImplementedError

    def save_tmp_files(self, data):
        raise NotImplementedError

    def run_exec_cmd(self, filenames, interpreter=None, conda=None, cmd_prefix=None, **kwargs):
        raise NotImplementedError

    def format_results(self, filenames, tcrs, pairwise):
        raise NotImplementedError

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
            p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
            stdo, stde = p.communicate()
            stdr = p.returncode
            if stdr > 0:
                raise RuntimeError("Unsuccessful execution of " + cmd + " (EXIT!=0) with output:\n" + stdo.decode())
            if os.path.getsize(tmp_path_out) == 0:
                raise RuntimeError(
                    "Unsuccessful execution of " + cmd + " (empty output file) with output:\n" +
                    stdo.decode())
        except Exception as e:
            raise RuntimeError(e)


class Ergo2(AExternalTCRSpecificityPrediction):
    """
    Implements ERGO-II, a deep learning based method for predicting TCR and epitope peptide binding by Springer et al.
    Paper: https://www.frontiersin.org/articles/10.3389/fimmu.2021.664514/full
    Repo: https://github.com/IdoSpringer/ERGO-II
    """
    __name = "ERGO-II"
    __version = ""

    @property
    def name(self):
        return self.__name

    @property
    def version(self):
        return self.__version

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
        return df_tcrs

    def save_tmp_files(self, data):
        tmp_folder = self.get_tmp_folder_path()
        path_in = os.path.join(tmp_folder.name, "ergo_ii_input.csv")
        path_out = os.path.join(tmp_folder.name, "ergo_ii_output.csv")
        data.to_csv(path_in, index=False)
        return [path_in, path_out], tmp_folder

    def run_exec_cmd(self, filenames, interpreter=None, conda=None, cmd_prefix=None, repository="", **kwargs):
        if repository == "" or not os.path.isdir(repository):
            raise NotADirectoryError(f"Repository: '{repository}' does not exist."
                                     f"Please provide a keyword argument repository with the path to the ERGO-II.\n"
                                     f"You can obtain the repo via: "
                                     f"'git clone https://github.com/IdoSpringer/ERGO-II.git'")
        os.chdir(repository)
        self.correct_code(repository)

        dataset = "vdjdb" if "dataset" not in kwargs else kwargs["dataset"]
        interpreter = "python" if interpreter is None else interpreter
        repository = "" if repository is None else f"{repository}/"
        cmds = []
        if cmd_prefix is not None:
            cmds.append(cmd_prefix)
        if conda is not None:
            cmds.append(f"conda activate {conda}")
        cmds.append(f"{interpreter} {repository}Predict.py {dataset} {filenames[0]} {filenames[1]}")
        self.exec_cmd("\n".join(cmds), filenames[1])

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


class ImRex(AExternalTCRSpecificityPrediction):
    """
    Implements ImRex (Interaction Map Recognition).
    Paper: https://doi.org/10.1093/bib/bbaa318
    Repo: https://github.com/pmoris/ImRex

    """
    __name = "ImRex"
    __version = ""

    @property
    def version(self):
        return self.__version

    @property
    def name(self):
        return self.__name

    def prepare_dataset_IMRex(self, df: pd.DataFrame, filename: str = None):
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