# run a line of the grid search on the mini dataset
#   https://docs.google.com/spreadsheets/d/1hoO96OTKLa7SKzUX1RKkg3xsThkpj7XQFceAt9gbg1s
#
# run this script from the root directory (not from scripts/)

import sys
import os 
import shutil
import json
import os

out_path_base = "/home/sescudero/emb2pfam/results/win128_out128_3e-04_1500filters/"

# save a copy of this script 
script_filename = os.path.basename(sys.argv[0])
shutil.copyfile(sys.argv[0], os.path.join(out_path_base, script_filename)) 

# copy the base config file to the output directory
train_config_path = os.path.join(out_path_base, "config.json")

# set training parameters
with open(train_config_path,'r') as config_file:     # opens the config file
    config_data = json.load(config_file)

steps = [4]
softmax_values = [False]

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

        os.system(f"python3 predict_errors.py -i {out_path_base} -c {pred_config_path} -o {pred_path}")
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")