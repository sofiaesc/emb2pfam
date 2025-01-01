import pickle

scores_comp_path = "/home/sescudero/emb2pfam/ensembles/models/fulldataset_win32_out32_1e-4/model_scores/all_scores_comp.pkl"

with open(scores_comp_path, 'rb') as f:
    all_scores_comp = pickle.load(f)

pid = "B1WYH7_CYAA5"
if pid in all_scores_comp:
    scores_comp = all_scores_comp[pid]
    print(f"Scores for PID {pid}:")
    print(f"Shape: {scores_comp.shape}")
    print(f"Entries: {scores_comp}")
else:
    print(f"PID {pid} not found.")