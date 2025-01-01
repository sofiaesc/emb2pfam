import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle
import torch as tr
import argparse
import json
from scipy.signal import medfilt

from utils import predict
from domCNN import domCNN

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, help="Input path to get model weights and save logs.")
parser.add_argument("-c", "--config", type=str, help="Parameters to use during testing.", default=None)
parser.add_argument("-o", "--output", type=str, help="Output path to save errors.")
args = parser.parse_args()
if args.input is None:
    raise argparse.ArgumentError(None, "An input path must be provided.")
if args.output is None:
    args.output = args.input

if not os.path.exists(args.output):
    os.makedirs(args.output)

config_path = args.config
if args.config is None:
    # using config from input path
    config_path = os.path.join(args.input,'config.json')
#config_path = os.path.join(args.input,'config.json')    # loads configuration file
with open(config_path, 'r') as f:
    config = json.load(f)
    
categories = [line.strip() for line in open(f"{config['data_path']}categories.txt")]
DEVICE = "cuda"

filename = os.path.join(args.input, "weights.pk")
summary_file = os.path.join(args.output, "model_scores_summary.txt")

net = domCNN(nclasses=len(categories), device=DEVICE)
net.load_state_dict(tr.load(filename))
net.eval()

dataset = pd.read_csv(f"{config['data_path']}test.csv")

print(f"Total rows: {len(dataset)}")
med = "_medfilt5_" if config['use_medfilt'] else "_"

all_scores_comp_path = os.path.join(args.output, "all_scores_comp.pkl")
if os.path.exists(all_scores_comp_path):
    with open(all_scores_comp_path, 'rb') as f:
        all_scores_comp = pickle.load(f)
else:
    all_scores_comp = {}

for pid in tqdm(dataset.PID.unique()):
    emb_file = f"{config['emb_path']}{pid}.pk"
    if not os.path.isfile(emb_file):
        print(f"Missing embedding: {pid}")
        continue

    emb = pickle.load(open(emb_file, "rb")).squeeze().float()

    centers, pred = predict(net, emb, config['window_len'], use_softmax=config['soft_max'], step=config['step'])

    # Keep only classes that have the maximum softmax score > TH (0.1)
    ind = tr.where(pred.max(dim=0)[0] > 0.3)[0]
    scores_comp = pred[:, ind]

    if config['use_medfilt']:
        scores_comp = medfilt(scores_comp.float(), [5, 1])

    all_scores_comp[pid] = scores_comp

    with open(all_scores_comp_path, 'wb') as f:
        pickle.dump(all_scores_comp, f)
