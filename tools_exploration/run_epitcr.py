import pandas as pd
import os

def transform_data(filename_input: str, filename_output: str):
    df = pd.read_csv(filename_input)
    df_new = pd.DataFrame()
    df_new['CDR3b'] = df['IR_VDJ_1_junction_aa']
    df_new['epitope'] = df['epitope']
    df_new['binder'] = 1
    df_new = df_new[df_new['CDR3b'].notnull()]
    new_row = {'CDR3b':'ASSYLPGQGDHYSNLPLP', 'epitope':'FLKEKGGL', 'binder':0}
    df_new = df_new.append(new_row, ignore_index=True)
    df_new.to_csv(filename_output, index=False)


transform_data('/home/icb/anna.chernysheva/vdjdb.csv', 'input_file.csv')
os.system('python3 predict.py --testfile input_file.csv --modelfile models/rdforestWithoutMHCModel.pickle --chain ce >> output.csv')
