# run a line of the grid search on the mini dataset
#   https://docs.google.com/spreadsheets/d/1hoO96OTKLa7SKzUX1RKkg3xsThkpj7XQFceAt9gbg1s
#
# run this script from the root directory (not from scripts/)

import sys
import os 
import shutil
import json
from datetime import datetime
import os

# set experiment name and results folder name
experiment = "grid_mini_ESM1b"
out_path_base = f"results/{experiment}_{str(datetime.now()).replace(' ', '-')}/"

# create directories (deletes the folder if it already exists)
if not os.path.exists(out_path_base):
    os.makedirs(out_path_base)

# save a copy of this script 
script_filename = os.path.basename(sys.argv[0])
shutil.copyfile(sys.argv[0], os.path.join(out_path_base, script_filename)) 

# copy the base config file to the output directory
base_config_path = "config/base.json"
train_config_path = os.path.join(out_path_base, "train_config.json")
shutil.copyfile(base_config_path, train_config_path)

# set training parameters
with open(train_config_path,'r') as config_file:     # opens the config file
    config_data = json.load(config_file)

config_data['window_len'] = 32                 # editing parameters
config_data['label_win_len'] = 32
config_data['batch_size'] = 32
config_data['lr'] = 1e-6

config_data['emb_path'] = "/DATA/emb2PFam/clustered_full_sequences_filtered/embeddings/"

with open(train_config_path,'w') as config_file:      # modifies the config file 
    json.dump(config_data,config_file,indent=4)

# train
os.system(f"python train.py -c {train_config_path} -o {out_path_base}")

# test with a window centered on the domain
os.system(f"python test.py -c {train_config_path} -i {out_path_base} ")

# predict errors with a sliding window, for different step sizes and softmax values
steps = [2,4,8,12,24]
softmax_values = [True, False]

for softmax in softmax_values:  # varying softmax
    config_data['soft_max'] = softmax
    print(f'softmax: {softmax}')

    for step in steps:   # varying number of steps
        config_data['step'] = step
        print(f"Step size = {step}")
        
        # create a new directory for the predictions
        pred_path = os.path.join(out_path_base, f"pred_step{step}_softmax{softmax}")
        os.makedirs(pred_path)
        # write the config file
        pred_config_path = os.path.join(pred_path, f"config.json")
        shutil.copyfile(train_config_path, pred_config_path)
        with open(pred_config_path,'w') as config_file: 
            json.dump(config_data,config_file,indent=4)

        os.system(f"python predict_errors.py -i {out_path_base} -c {pred_config_path} -o {pred_path}")
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")