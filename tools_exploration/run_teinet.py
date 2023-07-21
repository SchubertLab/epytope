import pandas as pd
from predict import predict_only
from utils import load_teinet

def transform_data(filename_input: str):
    df = pd.read_csv(filename_input)
    df_new = pd.DataFrame()
    df_new['CDR3b'] = df['IR_VDJ_1_junction_aa']
    df_new['epytope'] = df['epitope']
    df_new = df_new.dropna(subset = df_new.columns.values)
    df_new['CDR3b'] = 'C' + df_new['CDR3b'].astype(str)+ 'F'
    return df_new

teinet = load_teinet('results/large_dset.pth',device='cuda:0')
df = transform_data('/home/icb/anna.chernysheva/vdjdb.csv')
df = df.loc[df['CDR3b'].str.len() >= 5]
df = df.loc[df['CDR3b'].str.len() <= 29]
ts = df['CDR3b'].tolist()
es = df['epytope'].tolist()
predictions = predict_only(ts,es,model=teinet)
print(predictions)
