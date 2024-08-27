# run a line of the grid search on the mini dataset
#   https://docs.google.com/spreadsheets/d/1hoO96OTKLa7SKzUX1RKkg3xsThkpj7XQFceAt9gbg1s
#
# run this script from the root directory (not from scripts/)

import sys
import os 
import shutil
import json
import os

out_path_base = "ensembles/models/grid_mini_mejor4/"

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

# test with a window centered on the domain
os.system(f"python3 test.py -c {train_config_path} -i {out_path_base} ")