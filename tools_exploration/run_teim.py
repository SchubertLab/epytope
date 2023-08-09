import os
import pandas as pd

def transform_data(filename_input: str, filename_output: str):
    df = pd.read_csv(filename_input)
    df_new = pd.DataFrame()
    df_new['cdr3'] = df['IR_VDJ_1_junction_aa']
    df_new['epitope'] = df['epitope']
    df_new = df_new.dropna(subset = df_new.columns.values)
    df_new['cdr3'] = 'C' + df_new['cdr3'].astype(str)+ 'F'
    df_new.to_csv(filename_output, index=False)


transform_data('/home/icb/anna.chernysheva/vdjdb.csv', './inputs/inputs_bd.csv')
os.system('python scripts/inference_seq.py')
