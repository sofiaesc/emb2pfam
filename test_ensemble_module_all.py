import sklearn.metrics as mt
import numpy as np
from tqdm import tqdm
import torch as tr

from torch.utils.data import DataLoader
from domCNNe_all import domCNNe
from dataset import PFamDataset

# Parameters
models_path = 'ensembles/models/'
emb_path = "/home/rvitale/pfam32/embeddings/esm2/"
data_path = "/home/rvitale/pfam32/full/"
cat_path = '/home/rvitale/pfam32/full/categories.txt'
voting_method = 'all'  # 'mean', 'weighted_mean', 'weighted_families', 'majority', 'all'

LABEL_WIN_LEN = 32
BATCH_SIZE = 32
WIN_LEN = 32
ONLY_SEEDS = True

# Initializing ensemble instance
ensemble = domCNNe(models_path, emb_path, data_path, cat_path, voting_method)
ensemble.fit()
pred_mean, pred_weighted_mean, pred_weighted_fam, pred_majority = ensemble.pred()

# Obtaining references to compare with predictions for score
with open(cat_path, 'r') as f:
    categories = [item.strip() for item in f]

test_data = PFamDataset(f"{data_path}test.csv",
                        emb_path, categories, win_len=WIN_LEN,
                        label_win_len=LABEL_WIN_LEN, 
                        only_seeds=ONLY_SEEDS, is_training=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=1)

ref = []
for _, y, _, _, _, _, _ in tqdm(test_loader):
    ref.append(y.cpu())
ref = tr.cat(ref)
ref_bin = tr.argmax(ref, dim=1)

# Function to calculate and print metrics
def evaluate_predictions(predictions, method_name):
    accuracy = mt.accuracy_score(ref_bin.cpu().numpy(), predictions.cpu().numpy())
    total_errors = np.sum(1 - (predictions.cpu().numpy() == ref_bin.cpu().numpy()))
    print(f"{method_name} results:")
    print(f"Accuracy:     {accuracy * 100:6.2f}%")
    print(f"Error rate:   {(1 - accuracy) * 100:6.2f}%")
    print(f"Total errors: {total_errors:6d}")
    print()

# Evaluate all voting methods
if voting_method == 'all' or voting_method == 'majority':
    evaluate_predictions(pred_majority, "Majority")
if voting_method == 'all' or voting_method == 'mean':
    evaluate_predictions(pred_mean, "Mean")
if voting_method == 'all' or voting_method == 'weighted_mean':
    evaluate_predictions(pred_weighted_mean, "Weighted Mean")
if voting_method == 'all' or voting_method == 'weighted_families':
    evaluate_predictions(pred_weighted_fam, "Weighted Families")
