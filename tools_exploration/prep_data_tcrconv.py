import preprocessing.prep as prep
import numpy as np

protVb,protJb,_,_ = prep.get_protseqs_ntseqs(chain='B')
protVa,protJa,_,_ = prep.get_protseqs_ntseqs(chain='A')

datafile = './db/vdjdb_2021_1_11.tsv' # Exported from VDJdb
tcrs_vdj,epis_u,num_epis = prep.get_tcr_dict(datafile,chain='B', min_tcrs_per_epi=50)
tcru_dict = prep.get_unique_tcr_dict(tcrs_vdj,'B',vbseqs=protVb,jbseqs=protJb)

print('Number of unique TCRs:', len(tcru_dict))

prep.write_data_to_file(tcru_dict,'./training_data/vdjdb-b-large.csv',chain='B')
filename = './training_data/vdjdb-b-large.csv'

epis = np.loadtxt(filename,usecols=(0),unpack=True,delimiter=',',skiprows=1,comments=None,dtype='str')
epis_u,labels = prep.get_labels(epis)
np.save('./training_data/unique_epitopes_vdjdb-b-large.npy',epis_u)
