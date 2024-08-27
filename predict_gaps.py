"""Predicción en zonas donde no hay dominios conocidos TODO: revisar con los cambios del 202405"""

import torch as tr 
import pickle 
from tqdm import tqdm 
import pandas as pd 
import numpy as np 
from scipy.signal import medfilt

categories = [item.strip() for item in open("data/categories_ALL.txt")]
preds = pickle.load(open(f"pred_pfamALL_seeds_softlabels_w256_th.01.pk", "rb"))
pfam_gaps = pd.read_csv("pfam_gaps.csv", index_col='PID').drop(columns=["Unnamed: 0"])
dataset = pd.read_csv(f"data/ALL_test.csv")
median_filter = 5 

summary = []
for PID in tqdm(pfam_gaps.index.unique()):
    centers, scores_comp, class_ind = preds[PID]
    ref = dataset[dataset.PID==PID]
    
    scores_comp = medfilt(scores_comp.float(), [median_filter, 1])
    
    # uncompress predictions (easier to follow)
    scores = tr.zeros((centers.shape[0], len(categories)))
    scores[:, class_ind] = tr.tensor(scores_comp)
    
    if len(pfam_gaps.loc[PID].shape) == 1:
        gaps = [pfam_gaps.loc[PID].values]
    else:
        gaps = pfam_gaps.loc[PID].values.tolist()
    
    for gap in gaps:    
        
        ind = (centers>gap[0]) & (centers<gap[1])

        # Get the top predictions for each center
        # TODO esta mal esto, tendria que hacer el argmax si algun score es >.9, sino q ponga -1
        pred_class_raw = np.argmax(scores[ind,:], axis=1)
        
        # wrap it up in a dataframe
        pred_class = np.unique(pred_class_raw, return_counts=True)
        pred_class = sorted(zip(pred_class[0], pred_class[1]), key=lambda x: x[1], reverse=True)
        
        item = [PID, gap[0], gap[1]]
        for k in range(len(pred_class)):
            if k < len(pred_class):
                pc = pred_class[k]
                #score = pc[1]/len(pred_class_raw) score as a fraction of the total number of frames
                # score as maximum score of the frames
                score = tr.max(scores[ind, pc[0]]).item()
                segmentation_error = False
                if score>.9:
                    # check if the class is not already in the neighborhood
                    check_list = ref[ref.PF==categories[pc[0]]]
                    left, right = gap[0]-30, gap[1] + 30
                    for _, row in check_list.iterrows():
                        if (row.Inicio<left<row.Fin) or (row.Inicio<right<row.Fin):
                            segmentation_error = True # probably a segmentation error
                    if not segmentation_error:                     
                        item += [categories[pc[0]], score]
                
                
            if len(item) >= 9: # save top 3 predictions
                break
        
        L = len(item)
        if L == 3:
            continue # no prediction s
        if L < 9:
            item+=["",""]*int(.5*(9-L))
        summary.append(item)
        
    
summary = pd.DataFrame(summary, columns=["PID", "start", "end", "pred1", "score1", "pred2", "score2", "pred3", "score3"])
summary.to_csv("pfamAll_gaps_scores_filtered.csv", index=False)