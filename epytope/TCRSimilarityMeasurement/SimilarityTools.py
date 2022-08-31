# This code is part of the epytope distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
.. module:: TCRSimilarityMeasurement
   :synopsis: This module contains all classes for TCR similarity measurement methods.
.. moduleauthor:: albahah
"""

import pandas as pd
from epytope.Core.Base import ATCRSimilarityMeasurement, AExternal
from epytope.Core.AntigenImmuneReceptor import AntigenImmuneReceptor
from epytope.IO.FileReader import process_dataset_TCR
from epytope.Core.Result import TCRSimilarityMeasurementResult
from tcrdist.repertoire import TCRrep
import os
from tcrdist.rep_funcs import _pws
import pwseqdist as pw


class TCRSimilarityMeasurement(ATCRSimilarityMeasurement):
    """
        Abstract base class for TCR similarity measurement methods.
        Implements measure distance functionality.
    """

    def compute_distance(self, rep1, rep2=None):
        """
        computes pairwise similarity for all TCR seqs in one repertoire or in two repertoires
        :param rep1: :class:`~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor` or
        list(:class:`~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor`) representing the first repertoire
        :type rep1: :class:`~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor` or
        list(:class:`~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor`
        :param rep2: :class:`~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor` or
        list(:class:`~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor`) representing the second repertoire.
        Default value is None, in case the distance measurement should be done for TCR seqs in the first repertoire.
        :type rep2: :class:`~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor` or
        list(:class:`~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor`)
        :return: Returns a :class:`~epytope.Core.Result.TCRSimilarityMeasurementResult`
        :rtype: :class:`~epytope.Core.Result.TCRSimilarityMeasurementResult`
        """
        if rep1 is None:
            raise ValueError("you should pass a list of at least tow TCR seqs of type AntigenImmuneReceptor")
        if isinstance(rep1, AntigenImmuneReceptor):
            rep1 = [rep1]
        if rep2 and isinstance(rep2, AntigenImmuneReceptor):
            rep2 = [rep2]
        elif rep2 and not isinstance(rep2, list):
            raise ValueError(f"rep2 {rep2} should be an AntigenImmuneReceptor object or a list of AntigenImmuneReceptor "
                             f"objects")
        if isinstance(rep1, list):
            if len(rep1) == 0:
                raise ValueError(f"rep1 {rep1} should be an AntigenImmuneReceptor object or a list of "
                                 f"AntigenImmuneReceptor objects")
            for tcr in rep1:
                if not isinstance(tcr, AntigenImmuneReceptor):
                    raise ValueError(f"A {tcr} in rep1 should be an AntigenImmuneReceptor object")
            if rep2 is not None:
                if len(rep2) == 0:
                    raise ValueError(
                        f"rep2 {rep2} should be an AntigenImmuneReceptor object or a list of AntigenImmuneReceptor "
                        f"objects")
                for tcr in rep2:
                    if not isinstance(tcr, AntigenImmuneReceptor):
                        raise ValueError(f"A {tcr} in rep2 should be an AntigenImmuneReceptor object")
                data1 = {i: tcr.tcr for i, tcr in enumerate(rep1)}
                data2 = {i: tcr.tcr for i, tcr in enumerate(rep2)}
                return pd.DataFrame(data1).transpose(), pd.DataFrame(data2).transpose()
            else:
                data = {i: tcr.tcr for i, tcr in enumerate(rep1)}
                return pd.DataFrame(data).transpose(), None
        else:
            raise ValueError(f"rep1 {rep1} should be a list of AntigenImmuneReceptor objects")


class TCRDist3(TCRSimilarityMeasurement):
    """
        Implements tcrdist3, an open-source python package that enables a broad array of T cell receptor sequence
        analyses.
    """
    __name = "tcrdist3"
    __version = ""

    @property
    def version(self) -> str:
        """
        The version of the Method
        """
        return self.__version

    @property
    def name(self) -> str:
        """The name of the Method"""
        return self.__name

    def compute_distance(self, rep1, rep2=None, default: bool = True, organism: str = "human",
                         seq_type: str = None, metric: str = None, **kwargs):
        """
        computes pairwise similarity for all TCR seqs in one repertoire or in two repertoires
        :param rep1: :class:`~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor` or
        list(:class:`~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor`) representing the first repertoire
        :type rep1: :class:`~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor` or
        list(:class:`~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor`
        :param rep2: :class:`~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor` or
        list(:class:`~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor`) representing the second repertoire.
        Default value is None, in case the distance measurement should be done for TCR seqs in the first repertoire.
        :type rep2: :class:`~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor` or
        list(:class:`~epytope.Core.AntigenImmuneReceptor.AntigenImmuneReceptor`)
        :param bool default: the default value is True. In this case metric parameter will can not customized and the
        similarity will be computed by default parameters for all seqs in the repertoire. Otherwise, the metric used
        to compute the similarity can be customized.
        :param str organism: tcrdist3 by default supports only human and mouse organisms. Default value is human. Be
        aware this parameter will not be considered, if the default parameter is set to false.
        :param str seq_type: a string representing the sequence type, for which the tcrdist will be computed.
        For ALPHA the similarity will be computed for all alpha seqs, if they are present.
        For BETA the similarity will be computed for all alpha seqs, if they are present.
        For ALPHABETA the similarity will be computed for all alpha and beta seqs separately, if they are present. In
        this case there will be tow total distance one for all alpha seqs and one for all beta seqs.
        For BOTH the similarity will be computed for all alpha and beta seqs separately, if they are present. In
        this case there will be only one total distance for all alpha and beta seqs combined. Be aware this case will be
        only considered, if the default parameter is set to false.
        :param str metric: a string representing the metric used to compute the distance. It can be one of the following
        strings ['nb_vector_editdistance', 'nb_vector_hamming_distance', 'nb_vector_tcrdist', 'nw_hamming_metric',
        'nw_metric']. This parameter will not be considered, if the default parameter is set to ture.
        :return: Returns a :class:`~epytope.Core.Result.TCRSimilarityMeasurementResult`
        :rtype: :class:`~epytope.Core.Result.TCRSimilarityMeasurementResult`
        """
        df1, df2 = super(TCRDist3, self).compute_distance(rep1=rep1, rep2=rep2)
        return self.compute_distance_from_dataset(df1=df1, df2=df2, default=default, organism=organism,
                                                  seq_type=seq_type, metric=metric, **kwargs)

    def compute_distance_from_dataset(self, df1: pd.DataFrame = None, path1: str = None, df2: pd.DataFrame = None,
                                      path2: str = None, default: bool = True, organism: str = "human",
                                      seq_type: str = None, metric: str = None, source: str = None, **kwargs):
        """
        computes distance metric between all TCR seqs in the passed dataframe or a path to it
        :param str path1: a string representing a path to the first dataset(csv file), which will be precessed. Default
        value is None, when the dataframe object is given
        :param `pd.DataFrame` df1: first dataset(`pd.DataFrame). Default value is None, if the path1 is given
        :param str path2: a string representing a path to the second dataset(csv file), which will be precessed. Default
        value is None, when the dataframe object is given.
        :param `pd.DataFrame` df2: second dataset(`pd.DataFrame). Default value is None, if the path2 is given.
        :param bool default: the default value is True. In this case metric parameter will can not customized and the
        similarity will be computed by default parameters for all seqs in the repertoire. Otherwise, the metric used
        to compute the similarity can be customized.
        :param str organism: tcrdist3 by default supports only human and mouse organisms. Default value is human. Be
        aware this parameter will not be considered, if the default parameter is set to false.
        :param str seq_type: a string representing the sequence type, for which the tcrdist will be computed.
        For ALPHA the similarity will be computed for all alpha seqs, if they are present.
        For BETA the similarity will be computed for all alpha seqs, if they are present.
        For ALPHABETA the similarity will be computed for all alpha and beta seqs separately, if they are present. In
        this case there will be tow total distance one for all alpha seqs and one for all beta seqs.
        For BOTH the similarity will be computed for all alpha and beta seqs separately, if they are present. In
        this case there will be only one total distance for all alpha and beta seqs combined. Be aware this case will be
        only considered, if the default parameter is set to false.
        :param str metric: a string representing the metric used to compute the distance. It can be one of the following
        strings ['nb_vector_editdistance', 'nb_vector_hamming_distance', 'nb_vector_tcrdist', 'nw_hamming_metric',
        'nw_metric']. This parameter will not be considered, if the default parameter is set to ture.
        :param str source: the source of the dataset
        :return: A :class:`~epytope.Core.TCRSimilarityMeasurementResult` object
        :rtype: :class:`~epytope.Core.TCRSimilarityMeasurementResult`

        """
        kargs_vector_tcrdist = {
            "cdr3_a_aa": {'use_numba': True, 'distance_matrix': pw.matrices.tcr_nb_distance_matrix, 'dist_weight': 1,
                          'gap_penalty': 4, 'ntrim': 3, 'ctrim': 2, 'fixed_gappos': False},
            "pmhc_a_aa": {'use_numba': True, 'distance_matrix': pw.matrices.tcr_nb_distance_matrix, 'dist_weight': 1,
                          'gap_penalty': 4, 'ntrim': 0, 'ctrim': 0, 'fixed_gappos': True},
            "cdr2_a_aa": {'use_numba': True, 'distance_matrix': pw.matrices.tcr_nb_distance_matrix, 'dist_weight': 1,
                          'gap_penalty': 4, 'ntrim': 0, 'ctrim': 0, 'fixed_gappos': True},
            "cdr1_a_aa": {'use_numba': True, 'distance_matrix': pw.matrices.tcr_nb_distance_matrix, 'dist_weight': 1,
                          'gap_penalty': 4, 'ntrim': 0, 'ctrim': 0, 'fixed_gappos': True},
            "cdr3_b_aa": {'use_numba': True, 'distance_matrix': pw.matrices.tcr_nb_distance_matrix, 'dist_weight': 1,
                          'gap_penalty': 4, 'ntrim': 3, 'ctrim': 2, 'fixed_gappos': False},
            "pmhc_b_aa": {'use_numba': True, 'distance_matrix': pw.matrices.tcr_nb_distance_matrix, 'dist_weight': 1,
                          'gap_penalty': 4, 'ntrim': 0, 'ctrim': 0, 'fixed_gappos': True},
            "cdr2_b_aa": {'use_numba': True, 'distance_matrix': pw.matrices.tcr_nb_distance_matrix, 'dist_weight': 1,
                          'gap_penalty': 4, 'ntrim': 0, 'ctrim': 0, 'fixed_gappos': True},
            "cdr1_b_aa": {'use_numba': True, 'distance_matrix': pw.matrices.tcr_nb_distance_matrix, 'dist_weight': 1,
                          'gap_penalty': 4, 'ntrim': 0, 'ctrim': 0, 'fixed_gappos': True}
        }
        weights = {
            "cdr3_a_aa": 3,
            "pmhc_a_aa": 1,
            "cdr2_a_aa": 1,
            "cdr1_a_aa": 1,
            "cdr3_b_aa": 3,
            "pmhc_b_aa": 1,
            "cdr2_b_aa": 1,
            "cdr1_b_aa": 1
        }

        def prepare_input(df: pd.DataFrame, seq_type: str):
            pro_df = df[["Receptor_ID", 'TRA', 'TRB', "TRAV", "TRAJ", "TRBV", "TRBJ"]].copy(deep=True)
            pro_df.columns = ["Receptor_ID", "cdr3_a_aa", "cdr3_b_aa", "v_a_gene", "j_a_gene", "v_b_gene", "j_b_gene"]
            # For v_x_gene, include the full IMGT gene name and allele (e.g., TRBV1*01).
            pro_df["v_b_gene"] = pro_df["v_b_gene"].apply(lambda x: x if "*" in str(x) else str(x) + "*01")
            pro_df["v_a_gene"] = pro_df["v_a_gene"].apply(lambda x: x if "*" in str(x) else str(x) + "*01")
            if seq_type.lower() == "alphabeta" or seq_type.lower() == "both":
                if "count" not in pro_df.columns:
                    pro_df.loc[:, "count"] = pro_df.groupby(["cdr3_b_aa", "cdr3_b_aa"])["cdr3_b_aa"].transform("count")
                pro_df.drop_duplicates(subset=["cdr3_a_aa", "cdr3_b_aa"], keep="first", inplace=True)
                return pro_df[["cdr3_a_aa", "cdr3_b_aa", "v_a_gene", "j_a_gene", "v_b_gene", "j_b_gene"]], \
                       pro_df[["Receptor_ID", "cdr3_a_aa", "cdr3_b_aa"]]
            elif seq_type.lower() == "alpha":
                if "count" not in pro_df.columns:
                    pro_df.loc[:, "count"] = pro_df.groupby(["cdr3_a_aa"])["cdr3_a_aa"].transform("count")
                pro_df.drop_duplicates(subset=["cdr3_a_aa"], keep="first", inplace=True)
                return pro_df[["cdr3_a_aa", "v_a_gene", "j_a_gene", "count"]], \
                       pro_df[["Receptor_ID", "cdr3_a_aa", "cdr3_b_aa"]]
            elif seq_type.lower() == "beta":
                if "count" not in pro_df.columns:
                    pro_df.loc[:, "count"] = pro_df.groupby(["cdr3_b_aa"])["cdr3_b_aa"].transform("count")
                pro_df.drop_duplicates(subset=["cdr3_b_aa"], keep="first", inplace=True)
                return pro_df[["cdr3_b_aa", "v_b_gene", "j_b_gene", "count"]], \
                       pro_df[["Receptor_ID", "cdr3_a_aa", "cdr3_b_aa"]]
            else:
                raise ValueError(f"seq_type: {seq_type} should be one of the following seq_types: "
                             f"['alpha', 'bata', 'alphabeta', 'both']")

        def default_TCRrep(df1: pd.DataFrame, df2: pd.DataFrame = None, organism: str = "human",
                           chains: list = ["alpha"]) -> TCRrep:
            if organism not in ["mouse", "human"]:
                raise ValueError("tcrdist3 supports only two species: human and mouse")
            if df2 is not None:
                tr1 = TCRrep(cell_df=df1, organism=organism, chains=chains,
                            db_file='alphabeta_gammadelta_db.tsv', compute_distances=False)
                tr2 = TCRrep(cell_df=df2, organism=organism, chains=chains,
                             db_file='alphabeta_gammadelta_db.tsv', compute_distances=False)
                tr1.compute_rect_distances(df=tr1.cell_df, df2=tr2.cell_df)
                return tr1
            else:
                tr = TCRrep(cell_df=df1, organism=organism, chains=chains,
                             db_file='alphabeta_gammadelta_db.tsv', compute_distances=False)
                tr.compute_distances(df=tr.cell_df)
                return tr

        def cols(df: pd.DataFrame) -> dict:
            keys = []
            for col in df.columns:
                if col in ["pmhc_a_aa", "cdr2_a_aa", "cdr1_a_aa", "pmhc_b_aa", "cdr2_b_aa", "cdr1_b_aa"]:
                    keys.append(col)
            return keys

        def get_kargs(keys: list, metric: str = "nb_vector_tcrdist", **kwargs) -> dict:
            kargs = {}
            if metric == "nb_vector_tcrdist":
                for k in keys:
                    kargs[k] = kargs_vector_tcrdist[k]
            elif metric == 'nw_metric':
                open = 3
                extend = 3
                return_similarity = False
                if "open" in kwargs.keys() and isinstance(kwargs["open"], int):
                    open = kwargs["open"]
                if "extend" in kwargs.keys() and isinstance(kwargs["extend"], int):
                    extend = kwargs["extend"]
                if "return_similarity" in kwargs and isinstance(kwargs["return_similarity"], bool):
                    return_similarity = kwargs["return_similarity"]
                value = {'use_numba': False, "matrix": 'blosum62', "open": open,
                         "extend": extend, "return_similarity": return_similarity}
                for k in keys:
                    kargs[k] = value

            elif metric == 'nw_hamming_metric':
                open = 3
                extend = 3
                if "open" in kwargs.keys() and isinstance(kwargs["open"], int):
                    open = kwargs["open"]
                if "extend" in kwargs.keys() and isinstance(kwargs["extend"], int):
                    extend = kwargs["extend"]
                value = {'use_numba': False, "matrix": 'blosum62', "open": open,
                         "extend": extend}
                for k in keys:
                    kargs[k] = value

            elif metric == 'nb_vector_editdistance':
                value = {'use_numba': True, 'distance_matrix': pw.matrices.tcr_nb_distance_matrix, 'gap_penalty': 4}
                for k in keys:
                    kargs[k] = value

            elif metric == 'nb_vector_hamming_distance':
                value = {'use_numba': True, "check_lengths": False}
                for k in keys:
                    kargs[k] = value

            else:
                raise ValueError(f"The metric: {metric} should be one of the following metrics: "
                                 f"['nb_vector_editdistance', 'nb_vector_hamming_distance', 'nb_vector_tcrdist', "
                                 f"'nw_hamming_metric', 'nw_metric']")
            return kargs

        def alpha(df1: pd.DataFrame, df2: pd.DataFrame = None, metric: str = "nb_vector_tcrdist", **kwargs) -> dict:
            if "cdr3_a_aa" not in df1.columns:
                raise ValueError("cdr3_a_aa column is required")
            keys = [col for col in cols(df1) if col[5] == "a"]
            keys.append("cdr3_a_aa")
            columns = []
            columns.extend(keys)
            if "j_a_gene" in df1.columns:
                columns += ["v_a_gene", "j_a_gene"]
            else:
                columns.append("v_a_gene")
            metrics = {}
            weights_a = {}
            kargs = get_kargs(keys=keys, metric=metric, **kwargs)
            for k in keys:
                metrics[k] = getattr(pw.metrics, metric)
                weights_a[k] = weights[k]
            if df2 is not None:
                if "cdr3_a_aa" not in df2.columns:
                    raise ValueError("cdr3_a_aa column is required")
                result = _pws(df=df1, metrics=metrics, weights=weights_a, kargs=kargs, df2=df2, store=True)
                result["alpha"] = result["tcrdist"]
                return result
            else:
                result = _pws(df=df1, metrics=metrics, weights=weights_a, kargs=kargs, store=True)
                result["alpha"] = result["tcrdist"]
                return result

        def beta(df1: pd.DataFrame, df2: pd.DataFrame = None, metric: str = "nb_vector_tcrdist", **kwargs) -> dict:
            if "cdr3_b_aa" not in df1.columns:
                raise ValueError("cdr3_b_aa column is required")
            keys = [col for col in cols(df1) if col[5] == "b"]
            keys.append("cdr3_b_aa")
            columns = []
            columns.extend(keys)
            if "j_b_gene" in df1.columns:
                columns += ["v_b_gene", "j_b_gene"]
            else:
                columns.append("v_b_gene")
            metrics = {}
            weights_a = {}
            kargs = get_kargs(keys=keys, metric=metric, **kwargs)
            for k in keys:
                metrics[k] = getattr(pw.metrics, metric)
                weights_a[k] = weights[k]
            if df2 is not None:
                if "cdr3_b_aa" not in df2.columns:
                    raise ValueError("cdr3_b_aa column is required")
                result = _pws(df=df1, metrics=metrics, weights=weights_a, kargs=kargs, df2=df2, store=True)
                result["beta"] = result["tcrdist"]
                return result
            else:
                result = _pws(df=df1, metrics=metrics, weights=weights_a, kargs=kargs, store=True)
                result["beta"] = result["tcrdist"]
                return result

        def alpha_beta(df1: pd.DataFrame, df2: pd.DataFrame = None, metric: str = "nb_vector_tcrdist",
                       **kwargs) -> dict:
            result = {}
            if df2 is not None:
                result = alpha(df1=df1, df2=df2, metric=metric, **kwargs)
                result.update(beta(df1=df1, df2=df2, metric=metric, **kwargs))
            else:
                result = alpha(df1=df1, metric=metric, **kwargs)
                result.update(beta(df1=df1, metric=metric, **kwargs))
            return result

        def both(df1: pd.DataFrame, df2: pd.DataFrame = None, metric: str = "nb_vector_tcrdist", **kwargs) -> dict:
            if "cdr3_b_aa" not in df1.columns or "cdr3_a_aa" not in df1.columns:
                raise ValueError("cdr3_b_aa and cdr3_a_aa columns are required")
            keys = cols(df1)
            keys += ["cdr3_b_aa", "cdr3_a_aa"]
            columns = []
            columns.extend(keys)
            if "j_b_gene" in df1.columns and "j_a_gene" in df1.columns:
                columns += ["v_b_gene", "j_b_gene", "v_a_gene", "j_a_gene"]
            else:
                columns += ["v_b_gene", "v_a_gene"]
            metrics = {}
            weights_a = {}
            kargs = get_kargs(keys=keys, metric=metric, kwargs=kwargs)
            for k in keys:
                metrics[k] = getattr(pw.metrics, metric)
                weights_a[k] = weights[k]
            if df2 is not None:
                if "cdr3_b_aa" not in df2.columns or "cdr3_a_aa" not in df2.columns:
                    raise ValueError("cdr3_b_aa and cdr3_a_aa columns are required")
                result = _pws(df=df1, metrics=metrics, weights=weights_a, kargs=kargs, df2=df2, store=True)
                result["alpha+beta"] = result["tcrdist"]
                return result
            else:
                result = _pws(df=df1, metrics=metrics, weights=weights_a, kargs=kargs, store=True)
                result["alpha+beta"] = result["tcrdist"]
                return result

        if df1 is None:
            if path1 is None or not os.path.isfile(path1):
                raise FileNotFoundError("A path to a csv file or a dataframe should be passed")
            else:
                df1 = process_dataset_TCR(path=path1, source=source)
        else:
            df1 = process_dataset_TCR(df=df1, source=source)
        df1, tmp_df1 = prepare_input(df1, seq_type)

        if df2 is None:
            if path2 and not os.path.isfile(path2):
                raise FileNotFoundError(f"{path2} is not a path to a csv file")
            elif path2 and os.path.isfile(path2):
                df2 = pd.read_csv(path2)

        receptor_id2 = None
        if df2 is not None:
            df2 = process_dataset_TCR(df=df2, source=source)
            df2, tmp_df2 = prepare_input(df2, seq_type)

        result = None
        if seq_type and seq_type.upper() == "ALPHA":
            if default:
                result = default_TCRrep(df1=df1, df2=df2, organism=organism, chains=["alpha"])
            else:
                result = alpha(df1=df1, df2=df2, metric=metric, **kwargs)

        elif seq_type and seq_type.upper() == "BETA":
            if default:
                result = default_TCRrep(df1=df1, df2=df2, organism=organism, chains=["beta"])
            else:
                result = beta(df1=df1, df2=df2, metric=metric, **kwargs)

        elif seq_type and seq_type.upper() == "ALPHABETA":
            if default:
                result = default_TCRrep(df1=df1, df2=df2, organism=organism, chains=["alpha", "beta"])
            else:
                result = alpha_beta(df1=df1, df2=df2, metric=metric, **kwargs)

        elif seq_type and seq_type.upper() == "BOTH":
            if default:
                raise ValueError(f"if the seq_type is set to BOTH the default parameter should be set to False and the"
                                 f"metric parameter is required")
            result = both(df1=df1, df2=df2, metric=metric, **kwargs)


        # process the results
        seq_scores = {}
        if default:
            if df2 is not None:
                attributes = ["rw_cdr3_a_aa", "rw_cdr3_b_aa", "rw_alpha", "rw_beta"]
            else:
                attributes = ["pw_cdr3_a_aa", "pw_cdr3_b_aa", "pw_alpha", "pw_beta"]
            for attribute in attributes:
                if attribute in result.__dict__:
                    seq_scores[attribute[3:9]] = getattr(result, attribute)
        else:
            attributes = ["cdr3_a_aa", "cdr3_b_aa", "alpha", "beta"]
            for attribute in attributes:
                if attribute in result.keys():
                    seq_scores[attribute[:6]] = result[attribute]
            if seq_type.lower() == "both":
                seq_scores["alpha+beta"] = result["alpha+beta"]

        if df2 is None:
            index = tmp_df1.merge(tmp_df1, how="cross")
            index.columns = [("Rep1", "recep_id"), ("Rep1", "cdr3_a"), ("Rep1", "cdr3_b"), ("Rep2", "recep_id"),
                             ("Rep2", "cdr3_a"), ("Rep2", "cdr3_b")]
            index = index[[("Rep1", "recep_id"), ("Rep2", "recep_id"), ("Rep1", "cdr3_a"), ("Rep1", "cdr3_b"),
                           ("Rep2", "cdr3_a"), ("Rep2", "cdr3_b")]]
            return TCRSimilarityMeasurementResult.from_dict(seq_scores, index, self.__name, filt=True)

        else:
            index = tmp_df1.merge(tmp_df2, how="cross")
            index.columns = [("Rep1", "recep_id"), ("Rep1", "cdr3_a"), ("Rep1", "cdr3_b"), ("Rep2", "recep_id"),
                             ("Rep2", "cdr3_a"), ("Rep2", "cdr3_b")]
            index = index[[("Rep1", "recep_id"), ("Rep2", "recep_id"), ("Rep1", "cdr3_a"), ("Rep1", "cdr3_b"),
                           ("Rep2", "cdr3_a"), ("Rep2", "cdr3_b")]]
            return TCRSimilarityMeasurementResult.from_dict(seq_scores, index, self.__name)
