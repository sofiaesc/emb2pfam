from torch.utils.data import Dataset
import torch as tr
import numpy as np
import pandas as pd
import pickle
from torch import nn

class PFamDataset(Dataset):
    """
    Sample regions of proteins with multiple family tags (either from seed
    or full PFAM).
    Proteins have precomputed per-residue embeddings.
    """
    def __init__(self, dataset_path, emb_path, categories, win_len=256,
                 label_win_len=32, only_seeds=True, debug=False, is_training=False):
        """
        Dataset contains all valid domains related to complete proteins.
        """

        self.is_training = is_training
        self.emb_path = emb_path
        self.dataset = pd.read_csv(dataset_path)
        if only_seeds:
            self.dataset = self.dataset[self.dataset.Seed==1]
        self.categories = categories

        self.win_len = win_len
        self.label_win_len = label_win_len

        if debug:
            self.dataset = self.dataset.sample(n=100)

    def __len__(self):
        return len(self.dataset)

    def soft_domain_score(self, window_start, window_end, domain_start, domain_end):
        """Compute the percentage of the interval [domain_start, domain_end] in [window_start, window_end]"""
        return max(0, (min(domain_end, window_end) - max(domain_start, window_start))/(window_end-window_start))

    def __getitem__(self, item):
        """Sample one random window from a domain entry"""
        item = self.dataset.iloc[item]

        # Load precomputed embedding
        emb = pickle.load(open(f"{self.emb_path}{item.PID}.pk", "rb")).squeeze()
        
        # center the window in any AA of the domain (with 50% of the window 
        # covering the domain). Sometimes sample from a random point in the protein
        #if np.random.rand() < 0.3:
        #center = np.random.randint(0, emb.shape[1])
        #else:

        # at least 50% of domain in each window
        if self.is_training:
            center = np.random.randint(item.Inicio, item.Fin)
        else:
            center = (item.Inicio + item.Fin)//2
        # at least 10% of domain in each window
        #center = np.random.randint(item.Inicio-int(.9*(self.win_len//2)),
        #                           item.Fin + int(.9*(self.win_len//2)))

        start = max(0, center - self.win_len//2)
        end = min(emb.shape[1], center + self.win_len//2)

        label_start = max(0, center - self.label_win_len//2)
        label_end = min(emb.shape[1], center + self.label_win_len//2)

        # create label tensor
        label = tr.zeros(len(self.categories))
        # calculate coverage of the window on each domain to get a class score
        domains = self.dataset[self.dataset.PID==item.PID]
        for k in range(len(domains)):
            score = self.soft_domain_score(label_start, label_end, domains.iloc[k].Inicio, domains.iloc[k].Fin)
            label_ind = self.categories.index(domains.iloc[k].PF)
            label[label_ind] = max(score, label[label_ind])

        # force labels to sum 1
        s = label.sum()
        if s<1:
            ind = tr.where(label==0)[0]
            label[ind] = (1-s)/len(ind)

        emb_win = tr.zeros((emb.shape[0], self.win_len), dtype=tr.float)
        emb_win[:,:end-start] = emb[:, start:end]

        return emb_win, label, item.PID, start, end, label_start, label_end


def pad_batch(batch):
    """batch is a list of (seq, label, label_name)"""
    seqs = [(k, b[0]) for k, b in enumerate(batch)]
    labels = tr.tensor([b[1] for b in batch])
    names = [b[2] for b in batch]
    return seqs, labels, names


class ESM(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        # ESM init using torch hub
        self.emb_model, self.alphabet = tr.hub.load("facebookresearch/esm:main", "esm1b_t33_650M_UR50S")
        self.emb_model.eval()
        self.emb_model.to(device)
        self.device = device
        self.batch_converter = self.alphabet.get_batch_converter()

    def forward(self, seq):
        x = [(0, seq)]
        with tr.no_grad():
            _, _, tokens = self.batch_converter(x)
            emb = self.emb_model(tokens.to(self.device), repr_layers=[33],
            return_contacts=True)["representations"][33].detach().to(self.device)

        return emb.permute(0,2,1)

class ESM2(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        # ESM2 init using torch hub
        self.emb_model, self.alphabet = tr.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        self.emb_model.eval()
        self.emb_model.to(device)
        self.device = device
        self.batch_converter = self.alphabet.get_batch_converter()

    def forward(self, seq):
        x = [(0, seq)]
        with tr.no_grad():
            _, _, tokens = self.batch_converter(x)
            emb = self.emb_model(tokens.to(self.device), repr_layers=[33],
            return_contacts=True)["representations"][33].detach().to(self.device)

        return emb.permute(0,2,1)