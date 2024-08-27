"""Genera subsets de pfam"""
import os
import pickle
import pandas as pd
from random import shuffle
from tqdm import tqdm
from Bio import SeqIO
from dataset import ESM


DEVICE = "cuda"
PFAM_FULL_PATH = "data/Pfam-A_Full+Seed_v32.csv"

# complete protein sequences
sequences = {}
for item in SeqIO.parse(f"data/Pfam-A_seed_completeseq_v32.fasta", "fasta"):
    sequences[item.name.split("|")[0]] = str(item.seq)

# seed+full domain tags
dataset = pd.read_csv(PFAM_FULL_PATH)
print(dataset.shape)

# Pfam100: subset of 100 most common families from seed entries
#categories = dataset[dataset.Seed==1].groupby("PF")["PF"].count().sort_values(ascending=False)[:100].index.tolist()
#with open("data/categories_pfam100v32.txt", "w") as fout:
#    for c in categories:
#        fout.write(f"{c}\n") 
# Pfam200: Pfam100 + 100 less common families (with more than samples each)
#categories = [line.strip() for line in open(f"{DATA_DIR}Pfam_subset/categories_Pfam200v32.txt", "r")]
#SET_NAME = "Pfam200v32"

# all categories
categories = dataset.PF.unique()
SET_NAME = "ALL"

dataset = dataset[dataset.PF.isin(categories)]
print(dataset.shape)

# Run ESM on on each protein
OUT_PATH = "PFam_esm_half/"
        
esm = ESM()
if not os.path.isdir(OUT_PATH):
    os.mkdir(OUT_PATH)

included_pid = []
for PID in tqdm(dataset.PID.unique()):
    pickle_file = f"{OUT_PATH}{PID}.pk"
    if os.path.isfile(pickle_file):
        included_pid.append(PID)
        continue
    if PID not in sequences:
        continue
    seq = sequences[PID]
    if len(seq) >= 1023:
        continue
    emb = esm(seq).detach().cpu()   
    pickle.dump(emb.half(), open(pickle_file, "wb"))
    included_pid.append(PID)
        
# Remove samples that point to missing/failed embeddings
dataset = dataset[dataset.PID.isin(included_pid)]

# Split PID randomly  
pids = dataset.PID.unique()
shuffle(pids)
L = len(pids)

train_pids = pids[:int(.8*L)]
dataset_part = dataset[dataset.PID.isin(train_pids)]
dataset_part.to_csv(f"{SET_NAME}_train.csv", index=False)
print("train", dataset_part.shape)

dev_pids = pids[int(.8*L):int(.9*L)]
dataset_part = dataset[dataset.PID.isin(dev_pids)]
dataset_part.to_csv(f"{SET_NAME}_dev.csv", index=False)
print("dev", dataset_part.shape)

test_pids = pids[int(.9*L):]
dataset_part = dataset[dataset.PID.isin(test_pids)]
dataset_part.to_csv(f"{SET_NAME}_test.csv", index=False)
print("test", dataset_part.shape)