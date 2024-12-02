import sklearn.metrics as mt
import numpy as np
from tqdm import tqdm
import torch as tr

from torch.utils.data import DataLoader
from domCNNe_errors import domCNNe
from dataset import PFamDataset

# parameters:
models_path = 'ensembles/models/'
emb_path = "/home/rvitale/pfam32/embeddings/esm2/"
data_path = "/home/rvitale/pfam32/full/"
cat_path = '/home/rvitale/pfam32/full/categories.txt'
voting_method = 'weighted_mean'  # 'mean', 'weighted_mean', 'majority'

LABEL_WIN_LEN = 32
BATCH_SIZE = 32
WIN_LEN = 32
ONLY_SEEDS = True

# DATALOADERS:
with open(cat_path, 'r') as f:
    categories = [item.strip() for item in f]

dev_data = PFamDataset(f"{data_path}dev.csv",
                       emb_path, categories, win_len=WIN_LEN,
                       label_win_len=LABEL_WIN_LEN, 
                       only_seeds=ONLY_SEEDS, is_training=False)
dev_loader = DataLoader(dev_data, batch_size=BATCH_SIZE, num_workers=1)

test_data = PFamDataset(f"{data_path}test.csv",
                        emb_path, categories, win_len=WIN_LEN,
                        label_win_len=LABEL_WIN_LEN, 
                        only_seeds=ONLY_SEEDS, is_training=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=1)

# initializing ensemble instance
ensemble = domCNNe(models_path, emb_path, data_path, cat_path, voting_method)
ensemble.fit(dev_loader,test_loader)
pred_avg_bin = ensemble.pred(test_loader)

# obtaining references to compare with prediction for score
ref = []
for _, y, _, _, _, _, _ in tqdm(test_loader):
    ref.append(y.cpu())
ref = tr.cat(ref)
ref_bin = tr.argmax(ref, dim=1)

# obtaining accuracy score for the ensemble
a = mt.accuracy_score(ref_bin.cpu().numpy(), pred_avg_bin.cpu().numpy())
print("Ensemble results:")
print("Accuracy:     {0:6.2f}".format(a * 100) + "%")
print("Error rate:   {0:6.2f}".format((1 - a) * 100) + "%")
print("Total errors: {0:6d}".format(np.sum(1 - (np.array(pred_avg_bin) == np.array(ref_bin)))))
