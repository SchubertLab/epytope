import os
import pandas as pd

def transform_data(filename_input: str, filename_output: str):
    df = pd.read_csv(filename_input)
    df_new = pd.DataFrame()
    df_new['Peptide'] = df['epitope']
    df_new['CDR3'] = df['IR_VDJ_1_junction_aa']
    df_new = df_new.dropna(subset = df_new.columns.values)
    df_new['CDR3'] = 'C' + df_new['CDR3'].astype(str)+ 'F'
    df_new.to_csv(filename_output, index=False)


transform_data('/home/icb/anna.chernysheva/vdjdb.csv', './input.csv')
os.system('export LD_LIBRARY_PATH=/home/icb/anna.chernysheva/miniconda3/lib:$LD_LIBRARY_PATH')
os.system('python PanPep.py --learning_setting zero-shot --input ./input.csv --output ./Example_zero-shot_output.csv')
