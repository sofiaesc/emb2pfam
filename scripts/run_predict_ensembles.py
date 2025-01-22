import os

voting_methods = ["majority","mean","weighted_mean","weighted_families"]

for voting_method in voting_methods:
    print(f'----------------------------{voting_method}-----------------------------')
    os.system(f"python3 predict_errors_ensemble.py -m {voting_method}")
    
os.system(f"python3 test_ensemble_module_all.py")