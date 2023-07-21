# This code is part of the epytope distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
.. module:: Core.AResult
   :synopsis: Contains relevant classes describing results of predictions.
.. moduleauthor:: schubert

"""
__author__ = 'schubert'

import abc
import numpy
from numpy.lib.arraysetops import isin
import pandas
from epytope.Core.Allele import Allele
from epytope.Core.Peptide import Peptide
from copy import deepcopy
from sys import exit
import logging
import math


class AResult(pandas.DataFrame, metaclass=abc.ABCMeta):
    """
        A :class:`~epytope.Core.Result.AResult` object is a :class:`pandas.DataFrame` with with multi-indexing.

        This class is used as interface and can be extended with custom short-cuts for the sometimes often tedious
        calls in pandas
    """

    @abc.abstractmethod
    def filter_result(self, expressions):
        """
        Filter result based on a list of expressions

        :param list((str, comparator, float)) expressions: A list of triples consisting of (method_name, comparator, threshold)
        :return: A new filtered AResult object
        :rtype: :class:`~epytope.Core.Result.AResult`
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def merge_results(self, others):
        """
        Merges results of the same type and returns a merged result

        :param others: A (list of) :class:`~epytope.Core.Result.AResult` object(s) of the same class
        :type others: list(:class:`~epytope.Core.Result.AResult`)/:class:`~epytope.Core.Result.AResult`
        :return: A new merged :class:`~epytope.Core.Result.AResult` object
        :rtype: :class:`~epytope.Core.Result.AResult`
        """
        raise NotImplementedError()


class EpitopePredictionResult(AResult):
    """
        A :class:`~epytope.Core.Result.EpitopePredictionResult` object is a DataFrame with multi-indexing, where column
        Ids are the prediction model (i.e HLA :class:`~epytope.Core.Allele.Allele` for epitope prediction), row ID the
        target of the prediction (i.e. :class:`~epytope.Core.Peptide.Peptide`) and the second row ID the predictor
        (e.g. BIMAS)

        EpitopePredictionResult

        +----------------+-------------------------------+-------------------------------+
        |  Allele        |          Allele Obj 1         |          Allele Obj 2         | 
        +- - - - - - - - +- - - - - - - -+- - - - - - - -+- - - - - - - -+- - - - - - - -+
        |  Method        |    Method 1   |    Method 2   |    Method 1   |    Method 2   |
        +- - - - - - - - +- - - -+- - - -+- - - -+- - - -+- - - -+- - - -+- - - -+- - - -+
        | ScoreType      | Score |  Rank | Score |  Rank | Score |  Rank | Score |  Rank |
        +- - - - - - - - +- - - -+- - - -+- - - -+- - - -+- - - -+- - - -+- - - -+- - - -+
        |  Peptides      |       |       |       |       |       |       |       |       |
        +================+=======+=======+=======+=======+=======+=======+=======+=======+
        | Peptide Obj 1  |  0.03 |  57.4 |  0.05 |  51.1 |  0.08 |  49.4 |  0.73 |  3.12 |
        +----------------+-------+-------+-------+-------+-------+-------+-------+-------+
        | Peptide Obj 2  |  0.32 |  13.2 |  0.31 |  14.1 |  0.25 |  22.1 |  0.11 |  69.1 |
        +----------------+-------+-------+-------+-------+-------+-------+-------+-------+

    """

    def filter_result(self, expressions, scoretype='Score'):
        """
        Filters a result data frame based on a specified expression consisting of a list of triple with
        (method_name, comparator, threshold) and a str of the methods scoretype to be filtered.
        The expression is applied to each row. If any of the columns fulfill the criteria the row remains.

        :param list((str, comparator, float)) expressions: A list of triples consisting of (method_name, comparator, threshold)
        :param str scoretype: Indicates which scoretype of the specified method should be filtered

        :return: Filtered result object
        :rtype: :class:`~epytope.Core.Result.EpitopePredictionResult`
        """
        if isinstance(expressions, tuple):
            expressions = [expressions]

        df = deepcopy(self)
        methods = list(set(df.columns.get_level_values(1)))
        scoretypes = list(set(df.columns.get_level_values(2)))
        if scoretype not in scoretypes:
            raise ValueError(
                "Specified ScoreType {} does not match ScoreTypes of data frame {}.".format(scoretype, scoretypes))

        for expr in expressions:
            method, comp, thr = expr
            if method not in methods:
                raise ValueError("Specified method {} does not match methods of data frame {}.".format(method, methods))
            else:
                filt = comp(df.xs(method, axis=1, level=1).xs(scoretype, axis=1, level=1), thr).values
                # Only keep rows which contain values fulfilling the comparators logic in the specified method
                keep_row = [bool.any() for bool in filt]
                df = df.loc[keep_row]

        return EpitopePredictionResult(df)

    def merge_results(self, others):
        """
        Merges results of type :class:`~epytope.Core.Result.EpitopePredictionResult` and returns the merged result

        :param others: Another (list of) :class:`~epytope.Core.Result.EpitopePredictionResult`(s)
        :type others: list(:class:`~epytope.Core.Result.EpitopePredictionResult`)/:class:`~epytope.Core.Result.EpitopePredictionResult`
        :return: A new merged :class:`~epytope.Core.Result.EpitopePredictionResult` object
        :rtype: :class:`~epytope.Core.Result.EpitopePredictionResult`
        """
        df = self.copy(deep=False)

        if type(others) == type(self):
            others = [others]

        # Concatenates self and to be merged dataframe(s)
        for other in others:
            df = pandas.concat([df, other], axis=1)

        # Merge result of multiple predictors in others per allele
        df_merged = pandas.concat([group[1] for group in df.groupby(level=[0, 1], axis=1)], axis=1)

        return EpitopePredictionResult(df_merged)

    def from_dict(d, peps, method):
        """
        Create :class:`~epytope.Core.Result.EpitopePredictionResult` object from dictionary holding scores for alleles, peptides and a specified method

        :param d: dict with following structure: {allele: {scoretype: {peptide: score}}}
        :param peps: list of :class:`~epytope.Core.Peptide.Peptide`
        :param method: str specifying the prediction method 
        :return: A new :class:`~epytope.Core.Result.EpitopePredictionResult` object
        :rtype: :class:`~epytope.Core.Result.EpitopePredictionResult` 
        """
        scoreType = numpy.asarray([list(m.keys()) for m in [metrics for a, metrics in d.items()]]).flatten()
        alleles = numpy.asarray([numpy.repeat(a, len(set(scoreType))) for a in d]).flatten()

        meth = numpy.repeat(method, len(scoreType))
        multi_cols = pandas.MultiIndex.from_arrays([alleles, meth, scoreType], names=["Allele", "Method", "ScoreType"])
        df = pandas.DataFrame(float(0), index=pandas.Index(peps), columns=multi_cols)
        df.index.name = 'Peptides'
        # Fill DataFrame
        for allele, metrics in d.items():
            for metric, pep_scores in metrics.items():
                for pep, score in pep_scores.items():
                    df[allele][method][metric][pep] = score

        return EpitopePredictionResult(df)


class Distance2SelfResult(AResult):
    """
        Distance2Self prediction result
    """

    def filter_result(self, expressions):
        # TODO: has to be implemented
        pass

    def merge_results(self, others):
        # TODO: has to be implemented
        pass


class CleavageSitePredictionResult(AResult):
    """
        A :class:`~epytope.Core.Result.CleavageSitePredictionResult` object is a :class:`pandas.DataFrame` with
        multi-indexing, where column Ids are the prediction scores fo the different prediction methods, as well as the
        amino acid a a specific position, row ID the :class:`~epytope.Core.Protein.Protein` ID and the position of the
        sequence (starting at 0).



        CleavageSitePredictionResult:

        +--------------+-------------+-------------+-------------+
        | ID           | Pos         |     Seq     | Method_name |
        +==============+=============+=============+=============+
        | protein_ID   |     0       |     S       |     0.56    |
        +              +-------------+-------------+-------------+
        |              |     1       |     Y       |      15     |
        +              +-------------+-------------+-------------+
        |              |     2       |     F       |     0.36    |
        +              +-------------+-------------+-------------+
        |              |     3       |     P       |      10     |
        +--------------+-------------+-------------+-------------+
    """

    def filter_result(self, expressions):
        """
        Filters a result data frame based on a specified expression consisting
        of a list of triple with (method_name, comparator, threshold). The expression is applied to each row. If any of
        the columns fulfill the criteria the row remains.

        :param list((str, comparator, float)) expressions: A list of triples consisting of (method_name, comparator,
                                                         threshold)
        :return: A new filtered result object
        :rtype: :class:`~epytope.Core.Result.CleavageSitePredictionResult`
        """
        if isinstance(expressions, tuple):
            expressions = [expressions]

        # builde logical expression
        masks = [list(comp(self.loc[:, method], thr)) for method, comp, thr in expressions]

        if len(masks) > 1:
            masks = numpy.logical_and(*masks)
        else:
            masks = masks[0]
        # apply to all rows

        return CleavageSitePredictionResult(self.loc[masks, :])

    def merge_results(self, others):
        """
        Merges results of type :class:`~epytope.Core.Result.CleavageSitePredictionResult` and returns the merged result

        :param others: A (list of) :class:`~epytope.Core.Result.CleavageSitePredictionResult` object(s)
        :type others: list(:class:`~epytope.Core.Result.CleavageSitePredictionResult`) or
                      :class:`~epytope.Core.Result.CleavageSitePredictionResult`
        :return: A new merged :class:`~epytope.Core.Result.CleavageSitePredictionResult` object
        :rtype: :class:`~epytope.Core.Result.CleavageSitePredictionResult`
        """
        if type(others) == type(self):
            others = [others]
        df = self

        for i in range(len(others)):
            o = others[i]
            df1a, df2a = df.align(o, )

            o_diff = o.index.difference(df.index)
            d_diff = df.index.difference(o.index)

            if len(d_diff) and len(o_diff):
                df2a.loc[d_diff, "Seq"] = ""
                df1a.loc[o_diff, "Seq"] = ""
            elif len(o_diff):
                df2a.loc[df.index.intersection(o.index), "Seq"] = ""
                df1a.loc[o_diff, "Seq"] = ""
            elif len(d_diff):
                df2a.loc[d_diff, "Seq"] = ""
                df1a.loc[o.index.intersection(df.index), "Seq"] = ""
            else:
                df2a.loc[o.index, "Seq"] = ""

            zero1 = df1a == 0
            zero2 = df2a == 0
            true_zero = zero1 | zero2

            df1 = df1a.fillna(0)
            df2 = df2a.fillna(0)

            df_merged = df1 + df2
            false_zero = df_merged == 0
            zero = true_zero & false_zero

            nans = ~true_zero & false_zero
            df_merged = df_merged.where(~zero, other=0)
            df_merged = df_merged.where(~nans, other=numpy.NaN)
            df = df_merged
        return CleavageSitePredictionResult(df)


class CleavageFragmentPredictionResult(AResult):
    """
        A :class:`~epytope.Core.Result.CleavageFragmentPredictionResult` object is a :class:`pandas.DataFrame` with
        single-indexing, where column  Ids are the prediction scores fo the different prediction methods, and row ID
        the :class:`~epytope.Core.Peptide.Peptide` object.

        CleavageFragmentPredictionResult:

        +--------------+-------------+
        | Peptide Obj  | Method Name |
        +==============+=============+
        | Peptide1     | -15.34      |
        +--------------+-------------+
        | Peptide2     | 23.34       |
        +--------------+-------------+
    """

    def filter_result(self, expressions):
        """
        Filters a result data frame based on a specified expression consisting of a list of triple with
        (method_name, comparator, threshold). The expression is applied to each row. If any of the columns fulfill the
        criteria the row remains.

        :param list((str, comparator, float)) expressions: A list of triples consisting of (method_name, comparator,
                                                         threshold)
        :return: A new filtered result object
        :rtype: :class:`~epytope.Core.Result.CleavageFragmentPredictionResult`
        """

        if isinstance(expressions, tuple):
            expressions = [expressions]

        masks = [list(comp(self.loc[:, method], thr)) for method, comp, thr in expressions]

        if len(masks) > 1:
            masks = numpy.logical_and(*masks)
        else:
            masks = masks[0]
        # apply to all rows
        return CleavageFragmentPredictionResult(self.loc[masks, :])

    def merge_results(self, others):
        """
        Merges results of type :class:`~epytope.Core.Result.CleavageFragmentPredictionResult` and returns the merged
        result

        :param others: A (list of) :class:`~epytope.Core.Result.CleavageFragmentPredictionResult` object(s)
        :type others: list(:class:`~epytope.Core.Result.CleavageFragmentPredictionResult`) or
                      :class:`~epytope.Core.Result.CleavageFragmentPredictionResult`
        :return: new merged :class:`~epytope.Core.Result.CleavageFragmentPredictionResult` object
        :rtype: :class:`~epytope.Core.Result.CleavageFragmentPredictionResult`
        """
        if type(others) == type(self):
            others = [others]

        return CleavageFragmentPredictionResult(pandas.concat([self] + others, axis=1))


class TAPPredictionResult(AResult):
    """
        A :class:`~epytope.Core.Result.TAPPredictionResult` object is a :class:`pandas.DataFrame` with single-indexing,
        where column Ids are the ` prediction names of the different prediction methods, and row ID the
        :class:`~epytope.Core.Peptide.Peptide` object

        TAPPredictionResult:

        +--------------+-------------+
        | Peptide Obj  | Method Name |
        +==============+=============+
        | Peptide1     | -15.34      |
        +--------------+-------------+
        | Peptide2     | 23.34       |
        +--------------+-------------+
    """

    def filter_result(self, expressions):
        """
        Filters a result data frame based on a specified expression consisting
        of a list of triple with (method_name, comparator, threshold). The expression is applied to each row. If any of
        the columns fulfill the criteria the row remains.

        :param list((str, comparator, float)) expressions: A list of triples consisting of (method_name, comparator,
                                                         threshold)
        :return: A new filtered result object
        :rtype: :class:`~epytope.Core.Result.TAPPredictionResult`
        """
        if isinstance(expressions, tuple):
            expressions = [expressions]

        masks = [list(comp(self.loc[:, method], thr)) for method, comp, thr in expressions]

        if len(masks) > 1:
            masks = numpy.logical_and(*masks)
        else:
            masks = masks[0]
        # apply to all rows

        return TAPPredictionResult(self.loc[masks, :])

    def merge_results(self, others):
        """
        Merges results of type :class:`~epytope.Core.Result.TAPPredictionResult and returns the merged result

        :param others: A (list of) :class:`~epytope.Core.Result.TAPPredictionResult` object(s)
        :type others: list(:class:`~epytope.Core.Result.TAPPredictionResult`) or
                      :class:`~epytope.Core.Result.TAPPredictionResult`
        :return: A new merged :class:`~epytope.Core.Result.TAPPredictionResult` object
        :rtype: :class:`~epytope.Core.Result.TAPPredictionResult``
        """
        if type(others) == type(self):
            others = [others]

        return TAPPredictionResult(pandas.concat([self] + others, axis=1))


class TCRSpecificityPredictionResult(AResult):
    """
        A :class:`~epytope.Core.Result.TCRSpecificityPredictionResult` object is a :class:`pandas.DataFrame` with
        single-indexing, where column Ids are the prediction scores of the different prediction methods, and row Ids
        are the alpha and beta chains of an AntigenImmuneReceptor object and class:`~epytope.Core.TCREpitope.TCREpitope`
        object.
        TCRSpecificityPredictionResult:
        +--------------+-------------+-------------+-------------+
        |      TRA     |      TRB    |   Peptide   | Method Name |
        +==============+=============+=============+=============+
        |     TRA1     |     TRB1    |   Peptide1  |    0.34     |
        +--------------+-------------+-------------+-------------+
        |     TRA2     |     TRB2    |   Peptide2  |     0.95    |
        +--------------+-------------+-------------+-------------+
    """

    def filter_result(self, expressions):
        """
        Filters a result data frame based on a specified expression consisting of a list of triple with
        (method_name, comparator, threshold). The expression is applied to each row. If any of the columns fulfill the
        criteria the row remains.
        :param list((str, comparator, float)) expressions: A list of triples consisting of (method_name, comparator,
                                                         threshold)
        :return: A new filtered result object
        :rtype: :class:`~epytope.Core.Result.TCRSpecificityPredictionResult`
        """

        if isinstance(expressions, tuple):
            expressions = [expressions]

        masks = [list(comp(self.loc[:, method], thr)) for method, comp, thr in expressions]

        if len(masks) > 1:
            masks = numpy.logical_and(*masks)
        else:
            masks = masks[0]
        # apply to all rows
        return TCRSpecificityPredictionResult(self.loc[masks, :])

    def merge_results(self, others):
        """
        Merges results of type :class:`~epytope.Core.Result.TCRSpecificityPredictionResult` and returns the merged
        result
        :param others: A (list of) :class:`~epytope.Core.Result.TCRSpecificityPredictionResult` object(s)
        :type others: list(:class:`~epytope.Core.Result.TCRSpecificityPredictionResult`) or
                      :class:`~epytope.Core.Result.TCRSpecificityPredictionResult`
        :return: new merged :class:`~epytope.Core.Result.TCRSpecificityPredictionResult` object
        :rtype: :class:`~epytope.Core.Result.TCRSpecificityPredictionResult`
        """
        if type(others) == type(self):
            others = [others]

        return TCRSpecificityPredictionResult(pandas.concat([self] + others, axis=1))


class TCRSimilarityMeasurementResult(AResult):
    """
        A :class:`~epytope.Core.Result.TCRSimilarityMeasurementResult` object is a :class:`pandas.DataFrame` with
        single-indexing, where column Ids are the similarity scores of the different similarity methods, and row Ids
        are the Receptor_ID , alpha and beta chains of the first and second AntigenImmuneReceptor object respectively.
        TCRSimilarityMeasurementResult:
        +---------------+---------------+--------------+----------------+--------------+-------------+--------------------------------------+
        | Method                                                                                     |    Method Name                       |
        +- - - - - - - -+- - - - - - - -+- - - - - - -+ - - - - - - - - +- - - - - - - + - - - - - - +- - - - -+- - - - - +- - - - +- - - - +
        | Chain                                                                                      |  cdr3_a |  cdr3_b | alpha  |  beta   |
        +- - - - - - - -+- - - - - - - -+- - - - - - -+ - - - - - - - - +- - - - - - - + - - - - - - +- - - - -+- - - - - +- - - - +- - - - +
        |(Rep1,recep_id)| (Rep1,cdr3_a)| (Rep1,cdr3_b)| (Rep2,recep_id)| (Rep2,cdr3_a)| (Rep2,cdr3_b)|                                      |
        +===============+==============+==============+================+==============+==============+=========+=========+========+========+
        |        0      |     TRA1     |     TRB1     |        0       |    TRA2      |     TRB2     |    53   |    35   |   65   |   75   |
        +--------------+---------------+--------------+----------------+--------------+--------------+---------+---------+--------+--------+
        |              |               |              |        1       |    TRA2      |     TRB2     |    53   |    35   |   65   |   75   |
        +--------------+---------------+--------------+----------------+--------------+--------------+---------+---------+--------+--------+
    """

    def filter_result(self, expressions, seq_type):
        """
        Filters a result data frame based on a specified expression consisting of a list of triple with
        (method_name, comparator, threshold). The expression is applied to each row. If any of the columns fulfill the
        criteria the row remains.
        :param list((str, comparator, float)) expressions: A list of triples consisting of (method_name, comparator,
                                                         threshold)
        :return: A new filtered result object
        :rtype: :class:`~epytope.Core.Result.TCRSimilarityMeasurementResult`
        """
        if isinstance(expressions, tuple):
            expressions = [expressions]
        df = deepcopy(self)
        methods = list(set(df.columns.get_level_values(0)))
        seq_types = list(set(df.columns.get_level_values(1)))
        if seq_type not in seq_types:
            raise ValueError(f"Specified seq_type {seq_type} does not match seq_types of data frame {seq_types}.")

        for expr in expressions:
            method, comp, thr = expr
            if method not in methods:
                raise ValueError(f"Specified method {method} does not match methods of data frame {methods}.")
            else:
                filt = comp(df.xs(method, axis=1).xs(seq_type, axis=1), thr).values
                df = df.loc[filt]

        return TCRSimilarityMeasurementResult(df)

    def merge_results(self, others):
        """
        Merges results of type :class:`~epytope.Core.Result.TCRSimilarityMeasurementResult` and returns the merged
        result
        :param others: A (list of) :class:`~epytope.Core.Result.TCRSimilarityMeasurementResult` object(s)
        :type others: list(:class:`~epytope.Core.Result.TCRSimilarityMeasurementResult`) or
                      :class:`~epytope.Core.Result.TCRSimilarityMeasurementResult`
        :return: new merged :class:`~epytope.Core.Result.TCRSimilarityMeasurementResult` object
        :rtype: :class:`~epytope.Core.Result.TCRSimilarityMeasurementResult`
        """
        if type(others) == type(self):
            others = [others]
        return TCRSimilarityMeasurementResult(pandas.concat([self] + others, axis=1))

    def from_dict(d: dict, idx: pandas.DataFrame, method: str, filt: bool = False):
        """
        Create :class:`~epytope.Core.Result.TCRSimilarityMeasurementResult` object from dictionary holding distances for
         cdr3 alpha, cdr3 beta, alpha, beta or alpha and beta combined.
        :param d: dictionary with the following structure: {sequence type: distances `numpy.ndarray`}
        :param pandas.DataFrame idx: a dataframe with the following header:
        [("Rep1", "recep_id"), ("Rep2", "recep_id"), ("Rep1", "cdr3_a"), ("Rep1", "cdr3_b"), ("Rep2", "cdr3_a"),
        ("Rep2", "cdr3_b")] representing the one tcr from each two repertoires  and the corresponding sequences as rows.
        :param method: str specifying the tool used to measure the similarity
        :param filt: boolean value to filter all pairwise computed similarity for one seq compared with itself and avoid
        displaying the similarity score for two seqs more than one time in one repertoire. Default value is False. In
        this case no pairs will be eliminated by computing the scores for two repertoires.
        :return: A new :class:`~epytope.Core.Result.TCRSimilarityMeasurementResult` object
        :rtype: :class:`~epytope.Core.Result.TCRSimilarityMeasurementResult`
        """
        seq_type = numpy.asarray(list(d.keys()))
        meth = numpy.repeat(method, len(seq_type))
        multi_cols = pandas.MultiIndex.from_arrays([meth, seq_type], names=["Method", "Chain"])
        length = math.sqrt(len(idx))
        indices = []
        last_idx = length*(length-1)
        j = 1
        round = 1
        if filt:
            while j < last_idx:
                if j % length == 0:
                    round += 1
                    j += round
                indices.append(j)
                j += 1
        if filt:
            df = pandas.DataFrame(int(0), index=pandas.MultiIndex.from_frame(idx.iloc[indices, :]), columns=multi_cols)
        else:
            df = pandas.DataFrame(int(0), index=pandas.MultiIndex.from_frame(idx), columns=multi_cols)
        # Fill DataFrame
        tuples = list(idx.itertuples(index=False))
        for seq, scores in d.items():
                scores = scores.flatten()
                if filt:
                    for index in indices:
                        df.loc[tuples[index], (method, seq)] = scores[index]
                else:
                    for i, index in enumerate(tuples):
                        df.loc[index, (method, seq)] = scores[i]
        return TCRSimilarityMeasurementResult(df)
