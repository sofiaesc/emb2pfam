import sys
import os 
import shutil
import json
from datetime import datetime

# name of the experiment and output directory
experiment = "win128_out128_4e-04"
out_path_base = f"results/{experiment}_{str(datetime.now()).replace(' ', '-')}/"

# Check if the directory already exists
if os.path.exists(out_path_base):
    print(f"Directory {out_path_base} already exists. Continuing training...")
else:
    # Create the directories since they don't exist
    os.makedirs(out_path_base)

    # Save a copy of this script 
    script_filename = os.path.basename(sys.argv[0])
    shutil.copyfile(sys.argv[0], os.path.join(out_path_base, script_filename)) 

    # Copy the base config file to the output directory
    base_config_path = "config/base_full.json"
    config_path = os.path.join(out_path_base, "config.json")
    shutil.copyfile(base_config_path, config_path)

# Path to the config file
config_path = os.path.join(out_path_base, "config.json")

# Train
os.system(f"python3 train.py -c {config_path} -o {out_path_base}")

# Test
os.system(f"python3 test.py -i {out_path_base}")

print("Done.")
