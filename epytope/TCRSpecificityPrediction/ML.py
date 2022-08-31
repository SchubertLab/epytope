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
import pandas as pd
from tempfile import NamedTemporaryFile
from epytope.Core.Base import AExternal
from epytope.TCRSpecificityPrediction.External import AExternalTCRSpecificityPrediction
from epytope.IO.FileReader import process_dataset_TCR
from epytope.Core.Result import TCRSpecificityPredictionResult
import re


class ERGO(AExternalTCRSpecificityPrediction, AExternal):
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

    def predict(self, peptides, TCRs, repository: str, all: bool, trained_on: str = None):
        """
        Overwrites ATCRSpecificityPrediction.predict

        Predicts binding probability between a T-cell receptor CDR3 protein sequence and a peptide
        If alleles is not given, predictions for all valid alleles of the predictor is performed. If, however,
        a list of alleles is given, predictions for the valid allele subset is performed.
        :param peptides: The TCREpitope objects for which predictions should be performed
        :type peptides: :class:`~epytope.Core.TCREpitope.TCREpitope` or list(:class:`~epytope.Core.TCREpitope.TCREpitope`)
        :param TCRs: T cell receptor objects
        :type  :class:'~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor' or
        list(:class:'~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor')
        :param str repository: a path to a local github repository of ERGO-II predictor
        :param bool all: if true each TCR object will be joined with each peptide to perform the prediction, otherwise
        the prediction will be preformed in the same order of the passed peptides and TCRs objects
        :param str trained_on: specifying the dataset the model trained on
        :return: A :class:`~epytope.Core.TCRSpecificityPredictionResult` object
        :rtype: :class:`~epytope.Core.TCRSpecificityPredictionResult`
        """
        df = super().predict(peptides=peptides, TCRs=TCRs, repository=repository, all=all)
        df_result = self.predict_from_dataset(repository=repository, df=df, score=-1, trained_on=trained_on)
        return df_result

    def predict_from_dataset(self, repository: str, path: str = None, df: pd.DataFrame = None, source: str = "",
                             score: int = 1, trained_on: str=None):
        """
        Predicts binding probability between a T-cell receptor CDR3 protein sequence and a peptide.
        The path should lead to csv file with fixed column names dataset.columns = ['TRA', 'TRB', "TRAV", "TRAJ",
        "TRBV", "TRBJ", "T-Cell-Type", "Peptide", "MHC", "Species", "Antigen.species", "Tissue"]. If some values for
        one or more variables are unavailable, leave them as blank cells.
        :param str repository: a path to a local github repository of ERGO-II predictor
        :param str path: a string representing a path to the dataset(csv file), which will be precessed. Default value
        is None, when the dataframe object is given
        :param `pd.DataFrame` df: a dataframe object. Default value is None, when the path is given
        :param str source: the source of the dataset [vdjdb, mcpas, scirpy, IEDB]. If this parameter does not be passed,
         the dataset should be a csv file with the column names mentioned above
        :param int score: An integer representing a confidence score between 0 and 3 (0: critical information missing,
        1: medium confidence, 2: high confidence, 3: very high confidence). By processing all entries with a confidence
        score >= the passed parameter score will be kept. Default value is 1
        :param str trained_on: specifying the dataset the model trained on
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
        # get only gene allele annotation form family name of v, j regions respectively
        df[["TRAV", "TRAJ", "TRBV", "TRBJ"]] = df[["TRAV", "TRAJ", "TRBV", "TRBJ"]].apply(substring)
        ergo_df = df[['TRA', 'TRB', "TRAV", "TRAJ", "TRBV", "TRBJ", "T-Cell-Type", "Peptide", "MHC"]]
        ergo_df = ergo_df[(ergo_df["Peptide"] != "") & (ergo_df["TRB"] != "")]
        if not os.path.exists(os.path.join(repository, "Predict.py")):
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
        try:
            stdo = None
            stde = None
            cmd = self.__command
            if trained_on and trained_on.lower() in ["vdjdb", "mcpas"]:
                cmd += f" {trained_on.lower()} {tmp_file.name} {tmp_out.name}"
            else:
                if source and source.lower() in ["vdjdb", "mcpas"]:
                    cmd += f" {source.lower()} {tmp_file.name} {tmp_out.name}"
                else:
                    cmd += f" vdjdb {tmp_file.name} {tmp_out.name}"
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


