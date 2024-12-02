import torch as tr
import os
import json
import shutil
import argparse
import sys
import time  # Step 1: Import the time module

from torch.utils.data import DataLoader
from dataset import PFamDataset
from domCNN import domCNN

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="config/config.json",type=str,help="Config file path (optional, if not specified uses default configuration)")
parser.add_argument("-o", "--output",type=str,help="Output path to save model weights and logs.")

args = parser.parse_args()
if args.output is None:
    raise argparse.ArgumentError(None, "An output path must be provided.")
if not os.path.exists(args.output):
    os.makedirs(args.output)
    
with open(args.config, 'r') as f:
    config = json.load(f)

config_path = os.path.join(args.output, 'config.json')

categories = [line.strip() for line in open(f"{config['data_path']}categories.txt")]
DEVICE = "cuda"

filename = os.path.join(args.output, "weights.pk")
summary = os.path.join(args.output, "train_summary.csv")

train_data = PFamDataset(f"{config['data_path']}train.csv", config['emb_path'],
                         categories, win_len=config['window_len'], label_win_len=config['label_win_len'],
                         only_seeds=config['only_seeds'], is_training=True)
dev_data = PFamDataset(f"{config['data_path']}dev.csv", config['emb_path'],
                       categories, win_len=config['window_len'], label_win_len=config['label_win_len'],
                       only_seeds=config['only_seeds'], is_training=False)

print("train", len(train_data), "dev", len(dev_data))

train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, num_workers=config['nworkers'])
dev_loader = DataLoader(dev_data, batch_size=config['batch_size'], num_workers=config['nworkers'])

net = domCNN(len(categories), lr=config['lr'], device=DEVICE)
if os.path.exists(filename):
    if config['continue_training'] == True:         # if we want to continue training from before.
        print(f"Loading model from {filename}")
        net.load_state_dict(tr.load(filename))
        with open(summary, 'r') as s:
            last_sum = s.readlines()[-1].split(',')
            INIT_EP  = int(last_sum[0])+1
            best_err = float(last_sum[4])
            counter  = int(last_sum[5])
    else:                                          # if it was training before but stopped and we don't want to continue it, it starts again.
        print(f"Previous model found in {filename} and 'continue_training' is set to False.")
        confirmation = input("Are you sure you want to overwrite the existing model? (yes/[no]): ")  #confirmation message
        if confirmation.lower() == "yes":
            os.remove(filename)
            os.remove(summary)
            print("Previous model deleted successfully. Starting training...")
            with open(summary, 'w') as s:               
                s.write("Ep,Train loss,Dev Loss,Dev error,Best error,Counter,Epoch time (s)\n")  # Updated header
                INIT_EP, counter, best_err = 0, 0, 999.0
        else:
            print("Deletion aborted. Set 'continue_traning' parameter to 'True' in config.json if you want to continue training an already existing model.")
            sys.exit()
else:   # if the path didn't exist, there was no previous training so it starts anew.
    with open(summary, 'w') as s:
        s.write("Ep,Train loss,Dev Loss,Dev error,Best error,Counter,Epoch time (s)\n")  # Updated header
        INIT_EP, counter, best_err = 0, 0, 999.0

for epoch in range(INIT_EP, config['nepoch']):
    start_time = time.time()  # Step 2: Record the start time

    train_loss = net.fit(train_loader)
    dev_loss, dev_err, _, _, _, _, _, _, _ = net.pred(dev_loader)

    # early stop
    sv_mod=""
    if dev_err < best_err:
        best_err = dev_err
        tr.save(net.state_dict(), filename)
        counter = 0
        sv_mod=" - MODEL SAVED"
    else:
        counter += 1
        sv_mod=f" - EPOCH {counter} of {config['patience']}"

    epoch_time = time.time() - start_time  # Step 3: Record the end time and calculate duration

    print_msg=f"{epoch}: train loss {train_loss:.3f}, dev loss {dev_loss:.3f}, dev err {dev_err:.3f}"
    print(print_msg + sv_mod + f" - Time: {epoch_time:.2f}s")  # Optional: Print epoch time

    with open(summary, 'a') as s:
        s.write(f"{epoch},{train_loss},{dev_loss},{dev_err},{best_err},{counter},{epoch_time:.2f}\n")  # Step 4: Save the epoch time

        if counter >= config['patience']:
            break
