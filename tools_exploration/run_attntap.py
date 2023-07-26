import pandas as pd
import os

def transform_data(filename_input: str, filename_output: str):
    df = pd.read_csv(filename_input)
    df_new = pd.DataFrame()
    df_new['antigen'] = df['epitope']
    df_new['label'] = 1
    df_new['tcr'] = df['IR_VDJ_1_junction_aa']
    df_new = df_new[df_new['tcr'].notnull()]
    df_new['tcr'] = 'C' + df_new['tcr'].astype(str) + 'F'
    new_row = pd.DataFrame.from_records([{'antigen':'FLKEKGGL', 'label':0, 'tcr':'CASSYLPGQGDHYSNLPLPF'}])
    df_new = pd.concat([df_new, new_row])
    df_new.to_csv(filename_output, index=False)

transform_data('/home/icb/anna.chernysheva/vdjdb.csv', './Data/McPAS/McPAS_crossvalid_data/0/input_file.csv')
os.system('python ./Codes/AttnTAP_test.py --input_file=./Data/McPAS/McPAS_crossvalid_data/0/input_file.csv --output_file=./Results/output_file.csv --load_model_file=./Models/cv_model_0_vdjdb_0.pt')
