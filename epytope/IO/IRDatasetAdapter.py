# This code is part of the epytope distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
.. module:: IO.TCRDatasetAdapter
   :synopsis: Module handles reading of common TCR dataset file formats.
.. moduleauthor:: albahah, drost, chernysheva

"""
import pandas as pd
import numpy as np

from abc import ABCMeta

from epytope.Core.Peptide import Peptide
from epytope.Core.TCREpitope import TCREpitope
from epytope.Core.ImmuneReceptorChain import ImmuneReceptorChain
from epytope.Core.ImmuneReceptor import ImmuneReceptor

from epytope.Core.Base import ATCRDatasetAdapter


class IRDataset(metaclass=ABCMeta):
    __name = "dataset"
    __version = "1.0.0.0"

    def __init__(self, receptors=None):
        self.receptors = receptors

    def from_dataframe(self, df_irs, column_celltype="celltype", column_organism="organism",
                       prefix_vj_chain="VJ_", prefix_vdj_chain="VDJ_",
                       suffix_chain_type="chain_type", suffix_cdr3="cdr3",
                       suffix_v_gene="v_gene", suffix_d_gene="d_gene", suffix_j_gene="j_gene"):
        df_irs = df_irs[~df_irs[f"{prefix_vj_chain}{suffix_cdr3}"].isna() |
                        ~df_irs[f"{prefix_vdj_chain}{suffix_cdr3}"].isna()]
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
        df_irs.fillna("")
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
        return self.__name

    @property
    def version(self):
        return self.__version


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
        except KeyError:
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


class VDJdbAdapter(ATCRDatasetAdapter, IRDataset):
    __name = "vdjdb"
    __version = "2023.06.01"

    def __init__(self):
        """
        Extract TCR information from the VDJdb database format:
        https://vdjdb.cdr3.net/
        Please download the database or related subset.
        """
        super().__init__()
        self.epitopes = None

    def from_path(self, path_csv, **kwargs):
        df_irs = pd.read_csv(path_csv, sep="\t", comment="#")
        max_index = df_irs["complex.id"].max() + 1
        n_unpaired = sum(df_irs["complex.id"] == 0)
        df_irs.loc[df_irs["complex.id"] == 0, "complex.id"] = list(range(max_index, max_index + n_unpaired))

        dfs = {"TRA": df_irs[df_irs["gene"] == "TRA"].copy(),
               "TRB": df_irs[df_irs["gene"] == "TRB"].copy()}
        for name, df in dfs.items():
            dfs[name].index = dfs[name]["complex.id"]
            dfs[name] = dfs[name][["gene", "cdr3", "v.segm", "j.segm", "antigen.epitope", "species"]]
            dfs[name].columns = [f"{name} {col}" for col in dfs[name].columns]
        df_irs = dfs["TRA"].join(dfs["TRB"], how="outer")

        for col in ["antigen.epitope", "species"]:
            df_irs[col] = df_irs.apply(lambda x: x[f"TRA {col}"] if not x[f"TRA {col}"] != np.nan else x[f"TRB {col}"],
                                       axis=1)

        rename_dict = {
            "column_organism": "species",
            "prefix_vj_chain": "TRA ",
            "prefix_vdj_chain": "TRB ",
            "suffix_chain_type": "gene",
            "suffix_cdr3": "cdr3",
            "suffix_v_gene": "v.segm",
            "suffix_d_gene": "d.segm",
            "suffix_j_gene": "j.segm"
        }

        df_irs["TRB d.segm"] = None
        df_irs["celltype"] = "T cell"

        df_irs = df_irs.fillna("")
        df_irs = df_irs.replace("nan", "")

        df_irs.drop_duplicates(keep="first", inplace=True)
        df_irs = df_irs[(df_irs["TRA cdr3"] != "") | (df_irs["TRB cdr3"] != "")]
        df_irs = df_irs[df_irs["TRA cdr3"].notna() | df_irs["TRB cdr3"].notna()]

        self.save_eptiopes(df_irs)
        self.from_dataframe(df_irs, **rename_dict)

    def save_eptiopes(self, df_epitopes):
        epitopes = [TCREpitope(peptide=Peptide(ep)) for ep in df_epitopes["antigen.epitope"]]
        self.epitopes = epitopes

    @property
    def name(self):
        return self.__name

    @property
    def version(self):
        return self.__version


class IEDBAdapter(ATCRDatasetAdapter, IRDataset):
    __name = "iedb"
    __version = "1.0.0"

    def __init__(self):
        """
        Extract TCR information from the IEDB Recetpor-Eptiope database format:
        https://www.iedb.org/
        Please download the database or related subset.
        """
        super().__init__()
        self.epitopes = None

    def from_path(self, path_csv, **kwargs):
        df_irs = pd.read_csv(path_csv, low_memory=False, header=[0,1], comment="#")
        df_irs.columns = df_irs.columns.map('|'.join)
        df_irs["Chain 1|Type"] = df_irs["Chain 1|Type"].replace({"alpha": "TRA", "gamma": "TRG"})
        df_irs["Chain 2|Type"] = df_irs["Chain 2|Type"].replace({"beta": "TRB", "delta": "TRD"})
        df_irs = df_irs.rename(columns={"Chain 1|Type": "Chain 1|Calculated Type",
                                        "Chain 2|Type": "Chain 2|Calculated Type",
                                        "Chain 1|CDR3 Calculated": "Chain 1|Calculated CDR3",
                                        "Chain 2|CDR3 Calculated": "Chain 2|Calculated CDR3",
                                        })

        for col in ["Chain 1|Calculated CDR3", "Chain 2|Calculated CDR3"]:
            df_irs[col] = df_irs[col].apply(lambda x: x if x == "" else f"C{x}F")

        rename_dict = {
            "column_celltype": "Assay|Type",
            "prefix_vj_chain": "Chain 1|Calculated ",
            "prefix_vdj_chain":  "Chain 2|Calculated ",
            "suffix_chain_type": "Type",
            "suffix_cdr3": "CDR3",
            "suffix_v_gene": "V Gene",
            "suffix_d_gene": "D Gene",
            "suffix_j_gene": "J Gene"
        }
        df_irs = df_irs.replace("nan", "")
        df_irs["organism"] = ""

        keep_cols = ["Assay|Type", "organism", "Chain 1|Calculated V Gene", "Chain 1|Calculated J Gene",
                     "Chain 1|Calculated CDR3", "Chain 1|Calculated Type", "Chain 2|Calculated V Gene",
                     "Chain 2|Calculated D Gene", "Chain 2|Calculated J Gene",
                     "Chain 2|Calculated CDR3", "Chain 2|Calculated Type", "Epitope|Name", "Assay|MHC Allele Names"]
        df_irs = df_irs[keep_cols]
        df_irs = df_irs.fillna("")

        for col in ["Chain 1|Calculated CDR3", "Chain 2|Calculated CDR3", "Epitope|Name"]:
            df_irs = df_irs[df_irs[col].str.match("^[ACDEFGHIKLMNPQRSTVWY]*$")]

        df_irs.drop_duplicates(keep="first", inplace=True)
        df_irs = df_irs[(df_irs["Chain 1|Calculated CDR3"] != "") | (df_irs["Chain 2|Calculated CDR3"] != "")]
        df_irs = df_irs[df_irs["Chain 1|Calculated CDR3"].isna() | df_irs["Chain 2|Calculated CDR3"]]

        self.save_eptiopes(df_irs)
        self.from_dataframe(df_irs, **rename_dict)

    def save_eptiopes(self, df_epitopes):
        epitopes = []
        for i, row in df_epitopes[["Epitope|Name", "Assay|MHC Allele Names"]].iterrows():
            if row["Epitope|Name"] != "":
                new_epitope = TCREpitope(peptide=Peptide(row["Epitope|Name"]),
                                         allele=row["Assay|MHC Allele Names"].split(",")[0]
                                         if row["Assay|MHC Allele Names"] != "" else None)
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
        df_irs = pd.read_csv(path_csv, encoding='cp1252', low_memory=False, comment="#")
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

        df_irs.drop_duplicates(keep="first", inplace=True)
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
            if row["Epitope.peptide"] != "":
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
                df_irs[col.split("_")[1]] = ""

        # Assign celltype if not provided by chain annotation
        if "column_celltype" not in kwargs:
            def get_celltype(row):
                if row["IR_VJ_1_locus"] and isinstance(row["IR_VJ_1_locus"], str):
                    return f"{row['IR_VJ_1_locus'][0]} cell"
                if row["IR_VDJ_1_locus"] and isinstance(row["IR_VDJ_1_locus"], str):
                    return f"{row['IR_VDJ_1_locus'][0]} cell"
                return ""
            df_irs["celltype"] = df_irs.apply(get_celltype, axis=1)

        required_columns = ["celltype", "organism"]
        for chain in ["IR_VJ_1_", "IR_VDJ_1_"]:
            for entry in ["locus", "junction_aa", "v_call", "d_call", "j_call"]:
                required_columns.append(f"{chain}{entry}")
        df_irs = df_irs[required_columns].copy()

        df_irs.drop_duplicates(keep="first", inplace=True)
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


class AIRRAdapter(ScirpyAdapter, IRDataset):
    __name = "airr"
    __version = "scirpy:0.10.1"

    def __init__(self):
        """
        The sample data epytope/Data/examples/test_airr_example_{alpha, beta}.tsv utilized for the AIRR data loader
        was sourced from https://github.com/scverse/scirpy/tree/main/docs/tutorials/example_data/immunesim_airr, created by immuneSIM [Weber et al. 2020]
        """
        super().__init__()

    @property
    def name(self):
        return self.__name

    @property
    def version(self):
        return self.__version

    def from_path(self, path, **kwargs):
        import scirpy as ir
        adata = ir.io.read_airr(path)
        self.from_object(adata, **kwargs)


class DfDataset(ATCRDatasetAdapter, IRDataset):
    __name = "DataFrame"
    __version = "0.0.0.1"

    def __init__(self):
        super().__init__()

    def from_dataframe_by_prefix(self, df_irs, column_celltype="celltype", column_organism="organism",
                                 prefix_vj_chain="VJ_", prefix_vdj_chain="VDJ_",
                                 suffix_chain_type="chain_type", suffix_cdr3="cdr3",
                                 suffix_v_gene="v_gene", suffix_d_gene="d_gene", suffix_j_gene="j_gene"):
        """
        Creates a IR reperoite from a DataFrame by providing column names, prefixes for VJ / VDJ chain,
        and suffixes for the chain information.
        """
        super(IRDataset).from_dataframe(df_irs, column_celltype, column_organism, prefix_vj_chain, prefix_vdj_chain,
                                        suffix_chain_type, suffix_cdr3, suffix_v_gene, suffix_d_gene, suffix_j_gene)

    def from_dataframe_by_columns(self, df_irs, column_celltype="celltype", column_organism="organism",
                                  column_vj_chain_type="VJ_chain_type", column_vj_cdr3="VJ_cdr3",
                                  column_vj_v_gene="VJ_v_gene", column_vj_j_gene="VJ_j_gene",
                                  column_vdj_chain_type="VDJ_chain_type", column_vdj_cdr3="VDJ_cdr3",
                                  column_vdj_v_gene="VDJ_v_gene", column_vdj_d_gene="VDJ_d_gene",
                                  column_vdj_j_gene="VDJ_j_gene"):
        """
        Creates a IR repertoire from a DataFrame by indicating the column names
        """
        rename_dict = {
            "celltype": column_celltype,
            "organism": column_organism,
            "VJ_chain_type": column_vj_chain_type,
            "VJ_cdr3": column_vj_cdr3,
            "VJ_v_gene": column_vj_v_gene,
            "VJ_j_gene": column_vj_j_gene,
            "VDJ_chain_type": column_vdj_chain_type,
            "VDJ_cdr3": column_vdj_cdr3,
            "VDJ_v_gene": column_vdj_v_gene,
            "VDJ_d_gene": column_vdj_d_gene,
            "VDJ_j_gene": column_vdj_j_gene
        }
        for k, v in rename_dict.items():
            if v is None:
                df_irs[k] = ""
        df_irs = df_irs.rename(columns=dict((v, k) for k, v in rename_dict.items()))
        super(IRDataset).from_dataframe(df_irs)

    def from_path_by_prefix(self, path_csv, **kwargs):
        """
        Creates a IR repertoire from a csv file providing column names, prefixes for VJ and VDJ chain,
        and suffixes for the chain information.
        :param str path_csv: Location of the csv file containing the repertoire
        :param kwargs: follows the definition of DfDataset.from_dataframe_by_prefix()
        """
        super(IRDataset).from_path(path_csv, **kwargs)

    def from_path_by_columns(self, path_csv, **kwargs):
        """
        Creates a IR repertoire from a csv file providing the column names.
        :param str path_csv: Location of the csv file containing the repertoire
        :param kwargs: follows the definition of DfDataset.from_dataframe_by_prefix()
        """

    @property
    def name(self):
        return self.__name

    @property
    def version(self):
        return self.__version
