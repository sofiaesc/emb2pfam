import os 
import numpy as np
import pandas as pd
from scipy.signal import medfilt
from matplotlib import pyplot as plt
from Bio import SeqIO

AMINO_ACID_VOCABULARY = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'V', 'W', 'Y'
]

_PFAM_GAP_CHARACTER = '.'

# Other characters representing amino-acids not in AMINO_ACID_VOCABULARY.
_ADDITIONAL_AA_VOCABULARY = [
    # Substitutions
    'U',
    'O',
    # Ambiguous Characters
    'B',
    'Z',
    'X',
    # Gap Character
    _PFAM_GAP_CHARACTER
]

# Vocab of all possible tokens in a valid input sequence
FULL_RESIDUE_VOCAB = AMINO_ACID_VOCABULARY + _ADDITIONAL_AA_VOCABULARY

# Map AA characters to their index in FULL_RESIDUE_VOCAB.
_RESIDUE_TO_INT = {aa: idx for idx, aa in enumerate(FULL_RESIDUE_VOCAB)}

def residues_to_indices(amino_acid_residues):
    return [_RESIDUE_TO_INT[c] for c in amino_acid_residues]

def residues_to_one_hot(amino_acid_residues):
    """Given a sequence of amino acids, return one hot array.
    Supports ambiguous amino acid characters B, Z, and X by distributing evenly
    over possible values, e.g. an 'X' gets mapped to [.05, .05, ... , .05].
    Supports rare amino acids by appropriately substituting. See
    normalize_sequence_to_blosum_characters for more information.
    Supports gaps and pads with the '.' and '-' characters; which are mapped to
    the zero vector.
    Args:
      amino_acid_residues: string. consisting of characters from
        AMINO_ACID_VOCABULARY
    Returns:
      A numpy array of shape (len(amino_acid_residues),
      len(AMINO_ACID_VOCABULARY)).
    Raises:
      KeyError: if amino_acid_residues has a character not in FULL_RESIDUE_VOCAB.
    """
    residue_encodings = _build_one_hot_encodings()
    int_sequence = residues_to_indices(amino_acid_residues)
    return residue_encodings[int_sequence]

def _build_one_hot_encodings():
    """Create array of one-hot embeddings.
    Row `i` of the returned array corresponds to the one-hot embedding of amino
      acid FULL_RESIDUE_VOCAB[i].
    Returns:
      np.array of shape `[len(FULL_RESIDUE_VOCAB), 20]`.
    """
    base_encodings = np.eye(len(AMINO_ACID_VOCABULARY))
    to_aa_index = AMINO_ACID_VOCABULARY.index

    special_mappings = {
        'B':
            .5 *
            (base_encodings[to_aa_index('D')] + base_encodings[to_aa_index('N')]),
        'Z':
            .5 *
            (base_encodings[to_aa_index('E')] + base_encodings[to_aa_index('Q')]),
        'X':
            np.ones(len(AMINO_ACID_VOCABULARY)) / len(AMINO_ACID_VOCABULARY),
        _PFAM_GAP_CHARACTER:
            np.zeros(len(AMINO_ACID_VOCABULARY)),
    }
    special_mappings['U'] = base_encodings[to_aa_index('C')]
    special_mappings['O'] = special_mappings['X']
    special_encodings = np.array(
        [special_mappings[c] for c in _ADDITIONAL_AA_VOCABULARY])
    return np.concatenate((base_encodings, special_encodings), axis=0)


# adapted from https://www.kaggle.com/code/petersarvari/protcnn-fast
def read_original_data(path):
    shards = []
    for fn in os.listdir(path):
        with open(os.path.join(path, fn)) as f:
            shards.append(pd.read_csv(f, index_col=None))
    return pd.concat(shards)

def plot_domains(xind, categories, domains, title="Protein domains", pred=None, th=.1, median_filter=3, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    # Plot scores with at least th probability on any part
    if pred is not None:
        if median_filter is not None:
            pred = medfilt(pred, [median_filter, 1])

        for k in range(pred.shape[1]):
            if (pred[:, k] > th).any(): 
                ax.plot(xind, pred[:, k], "*-", label=categories[k])
        ax.legend()
        
    # add domain regions
    if len(domains.shape) == 1:
        domains = [domains]
    else:
        domains = [domains.sort_values("Inicio").iloc[k] for k in range(len(domains))]

    y = [.95 - k/20 for k in range(len(domains))]
    for k in range(len(domains)):
        domain = domains[k]
        color="red" if domain.Seed else "gray"
        # span from Inicio to Fin, and with y from 0 to 0.2
        
        plt.axvspan(domain.Inicio, domain.Fin, ymin=y[k]-.025, ymax=y[k]+.025, alpha=0.2, color=color)
        
        # add text between Inicio and Fin
        plt.text((domain.Inicio+domain.Fin)/2, y[k], domain.PF, color="black", fontsize=8, ha="center", va="center")
    plt.title(title);
    plt.ylim([0, 1.05])
    plt.xlim([xind.min(), xind.max()])

from torch.nn.functional import softmax
import torch as tr

# prediction with sliding window
def predict(net, emb, window_len, use_softmax=True, step=8):
    L = emb.shape[1]
    pred = []
    centers = np.arange(0, L, step)
    batch = tr.zeros((len(centers), emb.shape[0], window_len), dtype=tr.float)

    for k, center in enumerate(centers):
        start = max(0, center-window_len//2)
        end = min(L, center+window_len//2)
        batch[k,:,:end-start] = emb[:, start:end].unsqueeze(0)
    with tr.no_grad():
        pred = net(batch).cpu().detach()
    if use_softmax:
        pred = softmax(pred, dim=1)

    return centers, pred

def read_fasta(filename):
    with open(filename, "r") as f:
        sequences = {}
        for item in SeqIO.parse(filename, "fasta"):
            sequences[item.name] = str(item.seq)
    return sequences

def f1_scores_tables(res_path, output_path):
    resdf = pd.read_csv(res_path) # ex value for res_path: "../results/filtered_dataset/pfam_v32_medfilt5_ONLY_SEEDS_top1_U05_no_softmax.csv"

    # True positives (said it was 'x' when it was 'x')
    resdf_tp=resdf[(resdf["PF"]==resdf["pred1"])]
    scrdf_tp=resdf_tp.groupby("PF").agg(CountTP=("PID","count"),
                                        MeanScoreTP=("score1","mean"),
                                        StdScoreTP=("score1","std"),
                                        MinScoreTP=("score1","min"),
                                        MaxScoreTP=("score1","max"))

    # Separating the errors
    resdf_neq=resdf[(resdf["PF"]!=resdf["pred1"])]

    # False positives for the errors (said it was 'x' when it wasn't)
    scrdf_fp=resdf_neq.groupby("pred1").agg(CountFP=("PID","count"),
                                            MeanScoreFP=("score1","mean"),
                                            StdScoreFP=("score1","std"),
                                            MinScoreFP=("score1","min"),
                                            MaxScoreFP=("score1","max"))
    scrdf_fp.index.names = ['PF']

    # False negatives for the errors (did not say it was 'x' when it was)
    scrdf_fn=resdf_neq.groupby("PF").agg(CountFN=("PID","count"),
                                        MeanScoreFN=("score1","mean"),
                                        StdScoreFN=("score1","std"),
                                        MinScoreFN=("score1","min"),
                                        MaxScoreFN=("score1","max"))

    # Join all of the previous tables
    scrdf=resdf.groupby("PF").agg(CountInTest=("PID","count"))
    scrdf=scrdf.join(scrdf_tp, on="PF")
    scrdf["CountTP"]=scrdf["CountTP"].fillna(0)
    scrdf=scrdf.join(scrdf_fp, on="PF")
    scrdf["CountFP"]=scrdf["CountFP"].fillna(0)
    scrdf=scrdf.join(scrdf_fn, on="PF")
    scrdf["CountFN"]=scrdf["CountFN"].fillna(0)
    scrdf["FamilyF1"]=2*scrdf["CountTP"]/(2*scrdf["CountTP"]+
                                        scrdf["CountFN"]+scrdf["CountFP"])

    scrdf.to_csv(output_path) # stores results. ex value for output_path: "../results/filtered_dataset/scores_per_family_no_softmax.csv"
