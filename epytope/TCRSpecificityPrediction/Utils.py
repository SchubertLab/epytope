# coding=utf-8
# This code is part of the epytope distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
.. module:: TCRSpecficityPrediction.Utils
   :synopsis: This module contains utility functions, which need to be executed in a seperate environment
   due to model conflicts.
.. moduleauthor:: drost, chernysheva
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
    do_blosum = "blosum" in path_model.lower()
    chains = "separate" if "SEPARATE" in path_model else "CONCAT"
    ffn.read_iedb(fns=path_data, fn_blosum=path_blosum, blosum_encoding=do_blosum, chains=chains, is_train=False)
    ffn.pad_sequence(target_len=40, sequence="tcr")
    ffn.pad_sequence(target_len=25, sequence="antigen")

    ffn.predict()
    np.save(path_out, ffn.predictions)


def fullseq_reconstruction(todo_cut=True):
    try:
        from Stitchr import stitchrfunctions as fxn
        from Stitchr import stitchr as st
    except:
        raise ImportError("The tool requires full TCR sequences, which are derived from the CDR3 and V+J-genes. For this, please install Stitchr to the environment of the tool.")
    path_in = sys.argv[2]
    path_out = sys.argv[3]
    df_tcrs = pd.read_csv(path_in, index_col=0)
    species_counts = df_tcrs["organism"].value_counts(dropna=False)
    if len(species_counts) == 0:
        raise ValueError("The tools require organism information. However, no data was provided. Please add organism information to your input.")
    species = species_counts.index[0].lower().replace(" ", "")
    species = "human" if species in ["homosapiens"] else "mouse" if species in ["musmusculus", "murine"] else species
    species = species.upper()
    if species_counts[0] != len(df_tcrs):
        warnings.warn(f"Mixed or undefined TCR organism. The prediction will be conducted on majority {species}. Please make sure this is intended")

    codons = fxn.get_optimal_codons('', species)

    def stitch_tcr(row, chain="TRB"):
        prefix = "beta" if chain == "TRB" else "alpha"
        tcr_bits = {"v": row[f"{chain}V_IMGT"], "j": row[f"{chain}J_IMGT"], "cdr3": row[f"cdr3_{prefix}_aa"],
                "l": "", "c": "TRBC1*01" if chain=="TRB" else "TRAC*01",
                "skip_c_checks": False, "species": species, "seamless": False,
                "5_prime_seq": "", "3_prime_seq": "", "name": "TCR"}
        stitched = st.stitch(tcr_bits, tcr_dat, functionality, partial, codons, 3, "")
        seq = fxn.translate_nt("N" * stitched[2] + stitched[1])
        if todo_cut:
            idx_remove = 177 if chain == "TRB" else 141
            seq = seq[:idx_remove]
        return seq

    try:
        tcr_dat, functionality, partial = fxn.get_imgt_data("TRB", st.gene_types, species)
        df_tcrs["full_seq_reconstruct_beta_aa"] = df_tcrs.apply(lambda x: stitch_tcr(x, "TRB"), axis=1)

        tcr_dat, functionality, partial = fxn.get_imgt_data("TRA", st.gene_types, species)
        df_tcrs["full_seq_reconstruct_alpha_aa"] = df_tcrs.apply(lambda x: stitch_tcr(x, "TRA"), axis=1)
    except Exception as e:
        warnings.warn(f"{e}")
        warnings.warn("An error occurred during the full TCR sequence reconstruction with Stitchr. This issue can arise for several reasons. Please ensure that your data is correctly formatted. Please verify that you have downloaded the TCRB and TCRA data required for Stitchrdl. If not, run the following commands in the tool environment: pip install IMGTgeneDL && stitchrdl -s human .")
    df_tcrs.to_csv(path_out)

def nettcr():
    fullseq_reconstruction(todo_cut=False)
    path_in = sys.argv[3]
    path_out = sys.argv[4]
    df_tcrs = pd.read_csv(path_in)
    df_tcrs["ind"] = df_tcrs.index
    df_tcrs[["B1", "B2", "B3"]] = pd.DataFrame(df_tcrs.apply(lambda x: get_cdrs_from_fullseq(x, "TRB"), axis=1).tolist())
    df_tcrs[["A1", "A2", "A3"]] = pd.DataFrame(df_tcrs.apply(lambda x: get_cdrs_from_fullseq(x, "TRA"), axis=1).tolist())
    df_tcrs = df_tcrs.drop_duplicates()
    df_tcrs.to_csv(path_out)

def get_cdrs_from_fullseq(row, chain="TRB"):
    column = "full_seq_reconstruct_beta_aa" if chain == "TRB" else "full_seq_reconstruct_alpha_aa"
    full_seq = list(zip(row[["ind"]], row[[column]]))
    try:
        from anarci import anarci
    except:
        raise ImportError("NetTCR-2.2 requires sequences of all cdr regions, which are derived using ANARCI. Please install ANARCI to NetTCR-2.2 environment from https://github.com/oxpig/ANARCI.git.")
    numbered, al_det, hit = anarci(full_seq, scheme="imgt", output=False)
    cdr1 = ""
    cdr2 = ""
    cdr3 = ""
    for j in range(26, 38):
        cdr1 += numbered[0][0][0][j][1]
    cdr1 = cdr1.replace("-", "")
    for j in range(55, 65):
        cdr2 += numbered[0][0][0][j][1]
    cdr2 = cdr2.replace("-", "")
    for j in range(104, 117):
        cdr3 += numbered[0][0][0][j][1]
    cdr3 = cdr3.replace("-", "")
    return cdr1, cdr2, cdr3


if __name__ == "__main__":
    flavor = str(sys.argv[1])

    functions = {
        "tcellmatch": tcellmatch,
        "stapler": fullseq_reconstruction,
        "nettcr": nettcr
    }
    functions[flavor]()
