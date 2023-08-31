# coding=utf-8
# This code is part of the epytope distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
.. module:: TCRSpecficityPrediction.Utils
   :synopsis: This module contains utility functions, which need to be executed in a seperate environment
   due to model conflicts.
.. moduleauthor:: drost
"""

import argparse
import pandas as pd
import warnings
import sys


def tcellmatch():
    import tcellmatch.api as tm
    import numpy as np

    path_model = str(sys.argv[2])
    path_data = str(sys.argv[3])
    path_out = str(sys.argv[4])
    path_blosum = str(sys.argv[5])

    ffn = tm.models.EstimatorFfn()
    ffn.x_train = np.zeros((1, 1, 65, 26))
    ffn.covariates_train = np.zeros((1, 0))
    ffn.y_train = np.zeros((1, 1))
    ffn.tcr_len = 40
    ffn.load_model(path_model, path_model)

    ffn.clear_test_data()
    do_blosum = 'blosum' in path_model.lower()
    ffn.read_vdjdb(fns=path_data, fn_blosum=path_blosum, blosum_encoding=do_blosum, chains="trb", is_train=False)
    ffn.pad_sequence(target_len=40, sequence="tcr")
    ffn.pad_sequence(target_len=25, sequence="antigen")

    ffn.predict()
    np.save(path_out, ffn.predictions)


if __name__ == "__main__":
    flavor = str(sys.argv[1])

    functions = {
        "tcellmatch": tcellmatch,
    }
    functions[flavor]()

