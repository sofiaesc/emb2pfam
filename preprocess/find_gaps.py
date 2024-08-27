"""Find gaps in pfam test set labels"""

import pickle
import pandas as pd
import numpy as np 
from tqdm import tqdm 
from Bio import SeqIO

full_seq = SeqIO.to_dict(SeqIO.parse("data/Pfam-A_seed_completeseq_v32.fasta", "fasta"))
dataset = pd.read_csv("data/Pfam-A_Full+Seed_v32.csv")
test_data = pd.read_csv("data/ALL_test.csv")

pfam_gaps = [] 
min_gap_size = 50
for PID in tqdm(test_data.PID.unique()):
    L = len(full_seq[PID].seq)
    domains = dataset[dataset.PID == PID].sort_values(by="Inicio")
    covered_regions = np.zeros(L)
    for d in domains.itertuples():
        covered_regions[d.Inicio:d.Fin+1] = 1

    # Get consecutive regions in covered_regions with zero values
    ups = np.where(np.diff(covered_regions) == 1)[0].tolist() + [L]
    downs =  [0] + np.where(np.diff(covered_regions) == -1)[0].tolist()
    for s, e in zip(downs, ups):
        if e-s >= min_gap_size:
            pfam_gaps.append([PID, s, e])
    
pfam_gaps = pd.DataFrame(pfam_gaps, columns=["PID", "Inicio", "Fin"])
pfam_gaps.to_csv("pfam_gaps.csv")