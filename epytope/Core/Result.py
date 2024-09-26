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
<<<<<<< HEAD
import math
=======
>>>>>>> repo-b2/main


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
<<<<<<< HEAD

=======
            
>>>>>>> repo-b2/main
        df = deepcopy(self)
        methods = list(set(df.columns.get_level_values(1)))
        scoretypes = list(set(df.columns.get_level_values(2)))
        if scoretype not in scoretypes:
<<<<<<< HEAD
            raise ValueError(
                "Specified ScoreType {} does not match ScoreTypes of data frame {}.".format(scoretype, scoretypes))

=======
            raise ValueError("Specified ScoreType {} does not match ScoreTypes of data frame {}.".format(scoretype, scoretypes))
        
>>>>>>> repo-b2/main
        for expr in expressions:
            method, comp, thr = expr
            if method not in methods:
                raise ValueError("Specified method {} does not match methods of data frame {}.".format(method, methods))
            else:
<<<<<<< HEAD
                filt = comp(df.xs(method, axis=1, level=1).xs(scoretype, axis=1, level=1), thr).values
=======
                filt = comp(df.xs(method, axis = 1, level = 1).xs(scoretype, axis = 1, level = 1), thr).values
>>>>>>> repo-b2/main
                # Only keep rows which contain values fulfilling the comparators logic in the specified method
                keep_row = [bool.any() for bool in filt]
                df = df.loc[keep_row]

        return EpitopePredictionResult(df)
<<<<<<< HEAD
=======
        
>>>>>>> repo-b2/main

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
<<<<<<< HEAD

=======
        
>>>>>>> repo-b2/main
        # Concatenates self and to be merged dataframe(s)
        for other in others:
            df = pandas.concat([df, other], axis=1)

        # Merge result of multiple predictors in others per allele
<<<<<<< HEAD
        df_merged = pandas.concat([group[1] for group in df.groupby(level=[0, 1], axis=1)], axis=1)

=======
        df_merged = pandas.concat([group[1] for group in df.groupby(level=[0,1], axis=1)], axis=1)
    
>>>>>>> repo-b2/main
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
<<<<<<< HEAD

        meth = numpy.repeat(method, len(scoreType))
        multi_cols = pandas.MultiIndex.from_arrays([alleles, meth, scoreType], names=["Allele", "Method", "ScoreType"])
        df = pandas.DataFrame(float(0), index=pandas.Index(peps), columns=multi_cols)
=======
        
        meth = numpy.repeat(method, len(scoreType))
        multi_cols = pandas.MultiIndex.from_arrays([alleles, meth, scoreType], names=["Allele", "Method", "ScoreType"])
        df = pandas.DataFrame(float(0),index=pandas.Index(peps), columns=multi_cols)
>>>>>>> repo-b2/main
        df.index.name = 'Peptides'
        # Fill DataFrame
        for allele, metrics in d.items():
            for metric, pep_scores in metrics.items():
                for pep, score in pep_scores.items():
                    df[allele][method][metric][pep] = score
<<<<<<< HEAD

=======
        
>>>>>>> repo-b2/main
        return EpitopePredictionResult(df)


class Distance2SelfResult(AResult):
    """
        Distance2Self prediction result
    """

    def filter_result(self, expressions):
<<<<<<< HEAD
        # TODO: has to be implemented
        pass

    def merge_results(self, others):
        # TODO: has to be implemented
=======
        #TODO: has to be implemented
        pass

    def merge_results(self, others):
        #TODO: has to be implemented
>>>>>>> repo-b2/main
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

<<<<<<< HEAD
        # builde logical expression
=======
        #builde logical expression
>>>>>>> repo-b2/main
        masks = [list(comp(self.loc[:, method], thr)) for method, comp, thr in expressions]

        if len(masks) > 1:
            masks = numpy.logical_and(*masks)
        else:
            masks = masks[0]
<<<<<<< HEAD
        # apply to all rows
=======
        #apply to all rows
>>>>>>> repo-b2/main

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
<<<<<<< HEAD
            df1a, df2a = df.align(o, )
=======
            df1a, df2a = df.align(o,)
>>>>>>> repo-b2/main

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

<<<<<<< HEAD
            df_merged = df1 + df2
=======
            df_merged = df1+df2
>>>>>>> repo-b2/main
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
<<<<<<< HEAD
        # apply to all rows
=======
        #apply to all rows
>>>>>>> repo-b2/main
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

<<<<<<< HEAD
        return CleavageFragmentPredictionResult(pandas.concat([self] + others, axis=1))
=======
        return CleavageFragmentPredictionResult(pandas.concat([self]+others, axis=1))
>>>>>>> repo-b2/main


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
<<<<<<< HEAD
        # apply to all rows
=======
        #apply to all rows
>>>>>>> repo-b2/main

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

<<<<<<< HEAD
        return TAPPredictionResult(pandas.concat([self] + others, axis=1))


class TCRSpecificityPredictionResult(AResult):
    """
        A :class:`~epytope.Core.Result.TCRSpecificityPredictionResult` object is a :class:`pandas.DataFrame` with
        multi-indexing, where column Ids are the prediction model epitope (:class:`~epytope.Core.TCREpitope.TCREpitope`),
        row ID the binding recepotr (i.e. :class:`~epytope.Core.ImmunceReceptor.TCRReceptor`)
        and the second row ID the predictor (e.g. ERGO-II)


        TCRSpecificityPredictionResult
        +----------------+---------------------+---------------------+
        |  Epitope       |     Epitope Obj 1   |    Epitope Obj 2    |
        +- - - - - - - - +- - - - - - - - - - -+- - - - - - - - - - -+
        |  Method        | Method 1 | Method 2 | Method 1 | Method 2 |
        +- - - - - - - - +- - - - - +- - - - - +- - - - - +- - - - - +
        |  TCRs          |          |          |          |          |
        +================+==========+==========+==========+==========+
        | TCR     Obj 1  |   0.03   |   0.05   |   0.08   |   0.73   |
        +----------------+----------+----------+----------+----------+
        | TCR     Obj 2  |    0.32  |   0.31   |   0.25   |   0.11   |
        +----------------+----------+----------+----------+----------+

        When non-pairwise prediction is performed:
        +----------------------------+----------+----------+
        |  Method                    | Method 1 | Method 2 |
        +- - - - - - + - - - - - - - +- - - - - +- - - - - +
        |  TCRs      |    Epitopes   |          |          |
        +============+===============+==========+==========+
        | TCR Obj 1  | Epitope Obj 1 |   0.05   |    0.08  |
        +------------+---------------+----------+----------+
        | TCR Obj 2  | Epitope Obj 2 |   0.31   |    0.25  |
        +------------+---------------+----------+----------+
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
        result = self
        result = result.drop_duplicates()
        if not isinstance(others, list):
            others = [others]
        merging_tcr = self["TCR"]
        merging_tuples = [("TCR", el) for el in merging_tcr.columns]
        if "Epitope" in self:
            merging_tuples += [("Epitope", el) for el in ["Peptide", "MHC"]]
        for i in range(len(others)):
            other = others[i].drop_duplicates()
            epitope_cols = []
            result = pandas.merge(result, other, on = merging_tuples, how="outer")
            tcr = result["TCR"]
            result.drop(["TCR"], axis=1, level=0, inplace=True)
            result.sort_index(axis=1, level=0, sort_remaining=False, inplace=True)
            tuples = [("TCR", el) for el in tcr.columns]
            tcr.columns = pandas.MultiIndex.from_tuples(tuples)
            result = pandas.concat([tcr, result], axis=1)
        return result
=======
        return TAPPredictionResult(pandas.concat([self]+others, axis=1))
>>>>>>> repo-b2/main
