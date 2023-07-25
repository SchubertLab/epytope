# This code is part of the epytope distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
.. module:: IO.TCRDatasetAdapter
   :synopsis: Module handles reading of common TCR dataset file formats.
.. moduleauthor:: albahah, drost

"""

import os
import re
import warnings
import pandas as pd

from abc import ABCMeta, abstractmethod, ABC

from epytope.Core.Peptide import Peptide
from epytope.Core.TCREpitope import TCREpitope
from epytope.Core.ImmuneReceptorChain import ImmuneReceptorChain
from epytope.Core.ImmuneReceptor import ImmuneReceptor

from epytope.Core.Base import ATCRDatasetAdapter


class IRDataset(metaclass=ABCMeta):
    __name = "dataset"
    __version = "0.0.0.1"

    def __init__(self, receptors=None):
        self.receptors = receptors

    def from_dataframe(self, df_irs, column_celltype="celltype", column_organism="organism",
                       prefix_vj_chain="VJ_", prefix_vdj_chain="VDJ_",
                       suffix_chain_type="chain_type", suffix_cdr3="cdr3",
                       suffix_v_gene="v_gene", suffix_d_gene="d_gene", suffix_j_gene="j_gene"):
        df_irs = df_irs[df_irs[f"{prefix_vj_chain}{suffix_cdr3}"].isna() | df_irs[f"{prefix_vdj_chain}{suffix_cdr3}"]]
        repertoire = []
        for i, row in df_irs.iterrows():
            organism = row[column_organism]
            cell_type = row[column_celltype]
            chains = []
            for prefix_chain in [prefix_vj_chain, prefix_vdj_chain]:
                new_chain = ImmuneReceptorChain(
                    chain_type=row[f"{prefix_chain}{suffix_chain_type}"],
                    cdr3=row[f"{prefix_chain}{suffix_cdr3}"],
                    v_gene=row[f"{prefix_chain}{suffix_v_gene}"],
                    d_gene=row[f"{prefix_chain}{suffix_d_gene}"] if f"{prefix_chain}{suffix_d_gene}" in row else None,
                    j_gene=row[f"{prefix_chain}{suffix_j_gene}"],
                )
                chains.append(new_chain)
            new_ir = ImmuneReceptor(
                receptor_chains=chains,
                cell_type=cell_type,
                organism=organism
            )
            repertoire.append(new_ir)
        self.receptors = repertoire

    def from_path(self, path_csv, **kwargs):
        """
        Creates a IR repertoire from a csv file.
        :param str path_csv: Location of the csv file containing the repertoire
        :param kwargs: follows the definition of IRDataset.from_dataframe()
        """
        df_irs = pd.read_csv(path_csv)
        self.from_dataframe(df_irs, **kwargs)

    def to_pandas(self, rename_columns=None):
        """
        Creates a pandas.DataFrame of the receptor repertoire.
        :param rename_columns: Dictionary for renaming the csv columns different from the default
        :type rename_columns: dict(str, str)
        :return: a data frame containing the repertoire
        :rtype: :class: `pandas.DataFrame`
        """
        content = {
            "celltype": [],
            "organism": [],
        }

        for ir in self.receptors:
            content["celltype"].append(ir.cell_type if ir.cell_type else "")
            content["organism"].append(ir.organism if ir.organism else "")

            for chain_type in ["VDJ", "VJ"]:
                for attribute in ["chain_type", "cdr3", "v_gene", "d_gene", "j_gene"]:
                    if chain_type == "VJ" and attribute == "d_gene":
                        continue
                    column = f"{chain_type}_{attribute}"
                    if column not in content:
                        content[column] = []
                    content[column].append(str(ir.get_chain_attribute(attribute, chain_type)))

        df_irs = pd.DataFrame(content)
        if rename_columns is not None:
            df_irs = df_irs.rename(columns=rename_columns)
        return df_irs

    def to_csv(self, path_out, rename_columns=None):
        """
        Creates a csv file of the receptor repertoire.
        :param str path_out: specifies the output file
        :param str rename_columns: Dictionary for renaming the csv columns different from the default
        :type rename_columns: dict(str, str)
        """
        df_irs = self.to_pandas(rename_columns)
        path_out = path_out if path_out.endswith('.csv') else f"{path_out}.csv"
        df_irs.to_csv(path_out)

    @property
    def name(self):
        """ Name of the format used in this TCR-Dataset."""
        return self.__name

    @property
    def version(self):
        """ Version of the format used in this TCR-Dataset. """
        return self.__name


class MetaclassTCRAdapter(type):
    def __init__(cls, name, bases, nmspc):
        type.__init__(cls, name, bases, nmspc)

    def __call__(self, source, *args, **kwargs):
        """
        If a third person wants to write a new utility to load TCR repertoires. One has to name the file fred_plugin and
        inherit from CleavageFragmentPredictorFactory. That's it nothing more.
        """
        try:
            return ATCRDatasetAdapter[str(source).lower(), None]()
        except KeyError as e:
            raise ValueError(f"Predictor {source} is not known. Please verify that such an Predictor is " +
                             "supported by epytope and inherits ATCRDatasetAdapter")


class IRDatasetAdapterFactory(metaclass=MetaclassTCRAdapter):
    @staticmethod
    def available_methods():
        """
        Returns a dictionary of available epitope predictors and their supported versions

        :return: dict(str,list(str) - dictionary of epitope predictors represented as string and a list of supported
                                      versions
        """
        return {k: sorted(versions.keys()) for k, versions in ATCRDatasetAdapter.registry.items()}


"""
class VDJdbAdapter(IRDataset):
    def __init__(self):
        super().__init__()
        if source and source.lower() == "vdjdb":
            if df is None:
                df = pd.read_csv(path, sep='\t', low_memory=False)
                pd.options.mode.chained_assignment = None

            # select only the columns mentioned in the description above
            df = df[["meta.clone.id", "cdr3.alpha", "cdr3.beta", "v.alpha", "j.alpha", "v.beta", "j.beta",
                     "meta.cell.subset", "antigen.epitope", "mhc.a", "species", "antigen.species", "meta.tissue"]]
            # rename the selected columns
            df.columns = ["Receptor_ID", 'TRA', 'TRB', "TRAV", "TRAJ", "TRBV", "TRBJ", "T-Cell-Type", "Peptide", "MHC",
                          "Species",
                          "Antigen.species", "Tissue"]
            # replace not available values with empty cells
            df = df.fillna('')
            df.loc[:, "T-Cell-Type"] = df["T-Cell-Type"].apply(lambda x: x[:3])
            # get only gene allele annotation form family name of v, j regions respectively
            df[["TRAV", "TRAJ", "TRBV", "TRBJ"]] = df[["TRAV", "TRAJ", "TRBV", "TRBJ"]].apply(substring)
            return process(df)
"""


class IEDBAdapter(IRDataset):
    __name = "iedb"
    __version = "1.0.0"

    def __init__(self):
        """
        Extract TCR information from the IEDB Recetpor-Eptiope database-Format:
        https://www.iedb.org/
        Please download the database or related subset.
        """
        super().__init__()
        self.epitopes = None

    def from_path(self, path_csv, **kwargs):
        df_irs = pd.read_csv(path_csv, encoding='cp1252', low_memory=False)
        df_irs = df_irs.rename(columns={"CDR3.alpha.aa": "TRACDR3", "CDR3.beta.aa": "TRBCDR3"})
        rename_dict = {
            "prefix_vj_chain": "TRA",
            "prefix_vdj_chain": "TRB",
            "suffix_cdr3": "CDR3",
            "suffix_v_gene": "V",
            "suffix_d_gene": "D",
            "suffix_j_gene": "J",
            "column_celltype": "T.Cell.Type",
            "column_organism": "Species",
        }

        keep_cols = ["T.Cell.Type", "Species", "TRBCDR3", "TRBV", "TRBD", "TRBJ",  "TRACDR3", "TRAV", "TRAJ",
                     "Epitope.peptide", "MHC"]
        df_irs = df_irs[keep_cols]
        df_irs = df_irs.fillna("")

        for col in ["TRACDR3", "TRBCDR3", "Epitope.peptide"]:
            df_irs = df_irs[df_irs[col].str.match("^[ACDEFGHIKLMNPQRSTVWY]*$")]

        df_irs.drop_duplicates(keep='first', inplace=True)
        df_irs = df_irs[(df_irs["TRACDR3"] != "") | (df_irs["TRBCDR3"] != "")]
        df_irs = df_irs[df_irs["TRACDR3"].isna() | df_irs["TRBCDR3"]]

        df_irs["TRAchain_type"] = "TRA"
        df_irs["TRBchain_type"] = "TRB"
        df_irs["T.Cell.Type"] = df_irs["T.Cell.Type"] + "T cell"

        df_irs[["TRAcdr1", "TRAcdr2", "TRBcdr1", "TRBcdr2"]] = ""

        self.save_eptiopes(df_irs)
        self.from_dataframe(df_irs, **rename_dict)

    def save_eptiopes(self, df_epitopes):
        epitopes = []
        for i, row in df_epitopes[["Epitope.peptide", "MHC"]].iterrows():
            if re.match(r"^[ACDEFGHIKLMNPQRSTVWY]*$", row["Epitope.peptide"]):
                new_epitope = TCREpitope(peptide=Peptide(row["Epitope.peptide"]),
                                         alleles=row["MHC"] if row["MHC"] != "" else None)
            else:
                new_epitope = None
            epitopes.append(new_epitope)
        self.epitopes = epitopes

    @property
    def name(self):
        return self.__name

    @property
    def version(self):
        return self.__version

    def from_path(self, path_csv, **kwargs):
        df_irs = pd.read_csv(path_csv, sep=",", low_memory=False)

        rename_dict = {
            "column_celltype": None, # todo
            "column_organism": "Organism",
            "prefix_vj_chain": "Calculated Chain 1", # check whether VJ and VDJ right todo
            "prefix_vdj_chain":  "Calculated Chain 2",
            "suffix_chain_type": 'chain_type',
            "suffix_cdr3": "CDR3",
            "suffix_v_gene": "V Gene",
            "suffic_d_gene": None,  # todo
            "suffix_j_gene": "J gene"
        }
        self.from_dataframe(df_irs, **rename_dict)
        df.insert(6, "T-Cell-Type", "")
        df.insert(9, "Species", "")
        df[["TRA", "TRB", "TRAV", "TRAJ", "TRBV", "TRBJ"]] = \
            df[["TRA", "TRB", "TRAV", "TRAJ", "TRBV", "TRBJ"]].replace("nan", "")
        df.fillna("", inplace=True)
        df.drop_duplicates(subset=["TRA", "TRB", "Peptide"], keep='first', inplace=True)
        df = df[df["TRB"] != ""]


class McPasAdapter(ATCRDatasetAdapter, IRDataset):
    __name = "mcpas-tcr"
    __version = "1.0.0"

    def __init__(self):
        """
        Extract TCR information from the McPas-TCR database-Format: http://friedmanlab.weizmann.ac.il/McPAS-TCR/
        Please download the database or related subset.
        """
        super().__init__()
        self.epitopes = None

    def from_path(self, path_csv, **kwargs):
        df_irs = pd.read_csv(path_csv, encoding='cp1252', low_memory=False)
        df_irs = df_irs.rename(columns={"CDR3.alpha.aa": "TRACDR3", "CDR3.beta.aa": "TRBCDR3"})
        rename_dict = {
            "prefix_vj_chain": "TRA",
            "prefix_vdj_chain": "TRB",
            "suffix_cdr3": "CDR3",
            "suffix_v_gene": "V",
            "suffix_d_gene": "D",
            "suffix_j_gene": "J",
            "column_celltype": "T.Cell.Type",
            "column_organism": "Species",
        }

        keep_cols = ["T.Cell.Type", "Species", "TRBCDR3", "TRBV", "TRBD", "TRBJ",  "TRACDR3", "TRAV", "TRAJ",
                     "Epitope.peptide", "MHC"]
        df_irs = df_irs[keep_cols]
        df_irs = df_irs.fillna("")

        for col in ["TRACDR3", "TRBCDR3", "Epitope.peptide"]:
            df_irs = df_irs[df_irs[col].str.match("^[ACDEFGHIKLMNPQRSTVWY]*$")]

        df_irs.drop_duplicates(keep='first', inplace=True)
        df_irs = df_irs[(df_irs["TRACDR3"] != "") | (df_irs["TRBCDR3"] != "")]
        df_irs = df_irs[df_irs["TRACDR3"].isna() | df_irs["TRBCDR3"]]

        df_irs["TRAchain_type"] = "TRA"
        df_irs["TRBchain_type"] = "TRB"
        df_irs["T.Cell.Type"] = df_irs["T.Cell.Type"] + "T cell"

        self.save_eptiopes(df_irs)
        self.from_dataframe(df_irs, **rename_dict)

    def save_eptiopes(self, df_epitopes):
        epitopes = []
        for i, row in df_epitopes[["Epitope.peptide", "MHC"]].iterrows():
            if re.match(r"^[ACDEFGHIKLMNPQRSTVWY]*$", row["Epitope.peptide"]):
                new_epitope = TCREpitope(peptide=Peptide(row["Epitope.peptide"]),
                                         allele=row["MHC"] if row["MHC"] != "" else None)
            else:
                new_epitope = None
            epitopes.append(new_epitope)
        self.epitopes = epitopes

    @property
    def name(self):
        return self.__name

    @property
    def version(self):
        return self.__version


class ScirpyAdapter(ATCRDatasetAdapter, IRDataset):
    __name = "scirpy"
    __version = "0.10.1"

    def __init__(self):
        """
        Extract TCR information from a .h5ad file following scirpy naming convention.
        Note 1: only the primary IR chains are considered.
        Note 2:
        """
        super().__init__()

    def from_path(self, path, **kwargs):
        import scanpy as sc
        adata = sc.read(path)
        self.from_object(adata, **kwargs)

    def from_object(self, data, **kwargs):
        rename_dict = {
            "prefix_vj_chain": "IR_VJ_1_",
            "prefix_vdj_chain": "IR_VDJ_1_",
            "suffix_chain_type": "locus",
            "suffix_cdr3": "junction_aa",
            "suffix_v_gene": "v_call",
            "suffix_d_gene": "d_call",
            "suffix_j_gene": "j_call"
        }

        df_irs = data.obs
        for col in ["column_celltype", "column_organism"]:
            if col in kwargs:
                rename_dict[col] = kwargs[col]
            else:
                df_irs[col] = ""

        # Assign celltype if not provided by chain annotation
        if "column_celltype" not in kwargs:
            def get_celltype(row):
                if row["IR_VJ_1_locus"] and isinstance(row["IR_VJ_1_locus"], str):
                    return f"{row['IR_VJ_1_locus'][0]} cell"
                if row["IR_VDJ_1_locus"] and isinstance(row["IR_VDJ_1_locus"], str):
                    return f"{row['IR_VDJ_1_locus'][0]} cell"
                return ""
            df_irs["celltype"] = df_irs.apply(get_celltype, axis=1)

        required_columns = ["column_celltype", "column_organism"]
        for chain in ["IR_VJ_1_", "IR_VDJ_1_"]:
            for entry in ["locus", "junction_aa", "v_call", "d_call", "j_call"]:
                required_columns.append(f"{chain}{entry}")
        df_irs = df_irs[required_columns].copy()

        df_irs.drop_duplicates(inplace=True)
        df_irs.reset_index(drop=True, inplace=True)

        # Fill missing values
        for col in df_irs.columns:
            df_irs[col] = df_irs[col].astype(str)
        df_irs.replace("None", "", inplace=True)
        df_irs.replace("nan", "", inplace=True)
        df_irs.fillna("", inplace=True)

        self.from_dataframe(df_irs, **rename_dict)

    @property
    def name(self):
        return self.__name

    @property
    def version(self):
        return self.__version


'''
class AIRRAdapter(IRDataset):
    def __init__(self):
        super().__init__()



def process_dataset_TCR(path: str = None, df: pd.DataFrame = None, source: str = None, score: int = 1) \
        -> pd.DataFrame:

    def substring(column):
        """
        helper function to get gene allele annotation from family name of v,j regions
        :param column: pd.Series, where entries are the family name of v,j regions
        """
        return column.apply(lambda x: re.search(r"^\w*(-\d+)*", str(x)).group() if x != "" else x)


    def process(df: pd.DataFrame, source: str = None) -> pd.DataFrame:
        """
        helper function to check for invalid protein sequences in upper case.
        All rows with invalid cdr3 beta seqs will be removed, whereas invalid cdr3 alpha seqs will be replaced with an
        empty string
        :param df: a dataframe, which will be processed
        :param str source: the source of the dataset [vdjdb, McPAS, scirpy, IEDB].
        :return: returns the processed dataframe
        :rtype: `pd.DataFrame`
        """
        df.loc[:, "MHC"] = df["MHC"].apply(lambda x: re.search(r".*?(?=:)|.*", str(x)).group())
        df["TRA"] = df["TRA"].str.upper()
        df["TRB"] = df["TRB"].str.upper()
'''