import torch as tr
import json
import os
import argparse

from torch.utils.data import DataLoader
from dataset import PFamDataset
from domCNN import domCNN

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, help="Input path to get model weights and save logs.")
parser.add_argument("-c", "--config", type=str, help="Parameters to use during testing.", default=None)
parser.add_argument("-o", "--output", type=str, help="Output summary file.", default=None)

args = parser.parse_args()

if args.input is None:
    raise argparse.ArgumentError(None, "An input path must be provided.")

config_path = args.config
if args.config is None:
    # using config from input path
    config_path = os.path.join(args.input,'config.json')
summary = args.output
if args.output is None:
    summary = os.path.join(args.input, 'test_summary.csv')


with open(config_path, 'r') as f:
    config = json.load(f)

categories = [line.strip() for line in open(f"{config['data_path']}categories.txt")]
DEVICE = "cuda"

filename = f"{args.input}weights.pk"

test_data = PFamDataset(f"{config['data_path']}test.csv",
                        config['emb_path'], categories, win_len=config['window_len'],
                        label_win_len=config['label_win_len'], 
                        only_seeds=config['only_seeds'], is_training=False)
test_loader = DataLoader(test_data, batch_size=config['batch_size'], num_workers=1)

net = domCNN(len(categories), lr=config['lr'], device=DEVICE)
net.load_state_dict(tr.load(filename))
net.eval()
test_loss, test_errate, _, _, _, _, _, _, _ = net.pred(test_loader)
print(f"test_loss {test_loss:.5f} test_errate {test_errate:.5f}")

with open(summary, 'a') as s:
    s.write(f"test_loss,test_errate\n")
    s.write(f"{test_loss:.5f},{test_errate:.5f}")
