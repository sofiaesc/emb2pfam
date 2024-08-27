"""Script para generar los embeddings con ESM. Los archivos anteriores no se borran"""
from tqdm import tqdm
import pandas as pd
import os 
from Bio import SeqIO
from dataset import ESM2
import pickle 
import torch as tr

# TODO!!! Check the paths
DATA_PATH = 'data/minidataset/'
EMB_PATH = f"{DATA_PATH}embeddings/"
SUBSET = "Pfam_v32"
partition = "train" # train, dev, test
FASTA_PATH = f"{DATA_PATH}{SUBSET}_lt1024AA_seed_{partition}_filtered.fasta"
DATFR_PATH = f"{DATA_PATH}{SUBSET}_lt1024AA_seed_{partition}_filtered.csv"

DEVICE = "cuda"
# TODO!!! Path indicates where to search/save the embedder. 
tr.hub.set_dir("/home/sescudero/embedders/")
esm = ESM2(DEVICE)
if not os.path.isdir(EMB_PATH):
    os.mkdir(EMB_PATH)

# complete protein sequences
sequences = {}
for item in SeqIO.parse(FASTA_PATH, "fasta"):
    sequences[item.name.split("|")[0]] = str(item.seq)

dataset = pd.read_csv(DATFR_PATH)

# precalculate embeddings 
for PID in tqdm(dataset.PID.unique()):
    pickle_file = f"{EMB_PATH}{PID}.pk"
    if os.path.isfile(pickle_file):
        continue
    if PID not in sequences:
        continue
    seq = sequences[PID]
    emb = esm(seq).detach().cpu()   
    pickle.dump(emb.half(), open(pickle_file, "wb"))