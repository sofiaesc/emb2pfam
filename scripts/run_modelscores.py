import sys
import os 
import shutil
import json
import os

out_path_base = "ensembles/models/fulldataset_win32_out32_1e-4"

# save a copy of this script 
script_filename = os.path.basename(sys.argv[0])
shutil.copyfile(sys.argv[0], os.path.join(out_path_base, script_filename)) 

# copy the base config file to the output directory
train_config_path = os.path.join(out_path_base, "config.json")

# set training parameters
with open(train_config_path,'r') as config_file:     # opens the config file
    config_data = json.load(config_file)

pred_path = os.path.join(out_path_base, f"model_scores")
os.makedirs(pred_path)
# write the config file
pred_config_path = os.path.join(pred_path, f"config.json")
shutil.copyfile(train_config_path, pred_config_path)
with open(pred_config_path,'w') as config_file: 
    json.dump(config_data,config_file,indent=4)

os.system(f"python3 model_scores.py -i {out_path_base} -c {pred_config_path} -o {pred_path}")
print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")