import pandas as pd
import os

def transform_data(filename_input: str, filename_output: str):
    df = pd.read_csv(filename_input)
    df_new = pd.DataFrame()
    df_new['CDR3b'] = df['IR_VDJ_1_junction_aa']
    df_new['epitope'] = df['epitope']
    df_new = df_new.dropna(subset = df_new.columns.values)
    df_new['CDR3b'] = 'C' + df_new['CDR3b'].astype(str)+ 'F'
    df_new.to_csv(filename_output, index=False, header=False)


os.environ['MKL_THREADING_LAYER'] = 'GNU'
transform_data('/home/icb/anna.chernysheva/vdjdb.csv', './input.csv')
os.system('python ERGO.py predict lstm vdjdb specific cpu --model_file=models/lstm_vdjdb1.pt --train_data_file=train_data --test_data_file=input.csv >> output.csv')
