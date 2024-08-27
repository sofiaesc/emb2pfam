# run this script from the root directory (not from scripts/)
import sys
import os 
import shutil
import json
from datetime import datetime
import os

# name of the experiment and output directory
experiment = "ws_eq_wls"
out_path_base = f"results/{experiment}_{str(datetime.now()).replace(' ', '-')}/"

# create directories (deletes the folder if it already exists)
shutil.rmtree(out_path_base, ignore_errors=True)
os.makedirs(out_path_base)

# save a copy of this script 
script_filename = os.path.basename(sys.argv[0])
shutil.copyfile(sys.argv[0], os.path.join(out_path_base, script_filename)) 

# copy the base config file to the output directory
base_config_path = "config/base.json"
config_path = os.path.join(out_path_base, "config.json")
shutil.copyfile(base_config_path, config_path)

# set training parameters
with open(config_path,'r') as config_file:     # opens the config file
    config_data = json.load(config_file)

config_data['window_len'] = 32                 # editing parameters
config_data['label_win_len'] = 32
config_data['batch_size'] = 32
config_data['lr'] = 1e-6
config_data['step'] = 4

config_data['data_path'] = "/media/DATOS/emb2pfam_data/minidataset/"
config_data['emb_path'] = "/media/DATOS/emb2pfam_data/minidataset/embeddings/"

with open(config_path,'w') as config_file:      # modifies the config file 
    json.dump(config_data,config_file,indent=4)

# train
os.system(f"python train.py -c {config_path} -o {out_path_base}")

# test
os.system(f"python test.py -i {out_path_base}")

# predict, by default use the same path for input and output
os.system(f"python predict_errors.py -i {out_path_base}")   
print("Done.")