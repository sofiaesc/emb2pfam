import sys
import os 
import shutil
from datetime import datetime

config_folder = "varios_configs"
config_files = [f for f in os.listdir(config_folder) if f.endswith('.json')]

for config_file in config_files:
    config_name = os.path.splitext(config_file)[0]
    out_path_base = f"results/{config_name}_{str(datetime.now()).replace(' ', '-')}/"
    
    # Create directory (deletes if it already exists)
    shutil.rmtree(out_path_base, ignore_errors=True)
    os.makedirs(out_path_base)

    # Save a copy in the model folder
    script_filename = os.path.basename(sys.argv[0])
    shutil.copyfile(sys.argv[0], os.path.join(out_path_base, script_filename)) 

    base_config_path = os.path.join(config_folder, config_file)
    config_path = os.path.join(out_path_base, "config.json")
    shutil.copyfile(base_config_path, config_path)

    # Training
    os.system(f"python3 train.py -c {config_path} -o {out_path_base}")

    # Testing
    os.system(f"python3 test.py -i {out_path_base}")

print("Done.")