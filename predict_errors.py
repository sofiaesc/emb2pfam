import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle
import torch as tr
import argparse
import json
from scipy.signal import medfilt

from utils import predict
from domCNN import domCNN

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, help="Input path to get model weights and save logs.")
parser.add_argument("-c", "--config", type=str, help="Parameters to use during testing.", default=None)
parser.add_argument("-o", "--output", type=str, help="Output path to save errors.")
args = parser.parse_args()
if args.input is None:
    raise argparse.ArgumentError(None, "An input path must be provided.")
if args.output is None:
    args.output = args.input

if not os.path.exists(args.output):
    os.makedirs(args.output)

config_path = args.config
if args.config is None:
    # using config from input path
    config_path = os.path.join(args.input,'config.json')
#config_path = os.path.join(args.input,'config.json')    # loads configuration file
with open(config_path, 'r') as f:
    config = json.load(f)
    
categories = [line.strip() for line in open(f"{config['data_path']}categories.txt")]
DEVICE = "cuda"

filename = os.path.join(args.input, "weights.pk")
summary_file = os.path.join(args.output, "test_predict_errors_summary.txt")

res_name = os.path.join(args.output, "errors.csv")

net = domCNN(nclasses=len(categories), device=DEVICE)
net.load_state_dict(tr.load(filename))
net.eval()

dataset = pd.read_csv(f"{config['data_path']}test.csv")

print(f"Total rows: {len(dataset)}")
nTop=1

errors=[]
#missed_reference=[]
#missed_embedding=[]
#missed_index_err=[]
#missed_short_dom=[]
nOKs=0
nOKsArea=0
nOKsCoverage=0

med = "_medfilt5_" if config['use_medfilt'] else "_"

for pid in tqdm(dataset.PID.unique()):
    emb_file = f"{config['emb_path']}{pid}.pk"
    if not os.path.isfile(emb_file):
        print(f"Missing embedding: {pid}")
        #missed_embedding.append(pid)
        continue

    emb = pickle.load(open(emb_file, "rb")).squeeze().float()

    centers, pred = predict(net, emb, config['window_len'], use_softmax=config['soft_max'], step=config['step'])

    # keep only classes that have the maximum softmax score > TH
    ind = tr.where(pred.max(dim=0)[0]>config['th'])[0]
    scores_comp = pred[:, ind]
    if config['use_medfilt']:
        scores_comp = medfilt(scores_comp.float(), [5, 1])

    # uncompress predictions (easier to follow)
    scores = tr.zeros((centers.shape[0], len(categories)))
    scores[:, ind] = tr.tensor(scores_comp)

    ref=dataset[dataset.PID == pid].sort_values(by="Inicio")
    if ref.empty:
        print(f"Missing reference: {pid}")
        #missed_reference.append(pid)
        continue

    for _, row in ref.iterrows(): # for each domain in the reference
        start, end = row.Inicio, row.Fin
        if end-start < 10:
            print(f"Short domain: {pid}")
            #missed_short_dom.append(pid)
            continue

        pred_start = np.argmin(np.abs(centers-start))
        pred_end = np.argmin(np.abs(centers-end))

        if (pred_start<pred_end):
            # take all classes with score > TH
            pred_class = np.where(scores[pred_start:pred_end,:].max(axis=0).values > config['th'])[0] 

            # sort by score
            if len(pred_class) > 1:
                pred_class_ind = np.argsort(scores[pred_start:pred_end, pred_class].max(axis=0).values)#[::-1]
                # save the top-10 classes to csv
                pred_class = pred_class[pred_class_ind][::-1]
            # get the score of the top-10 classes in pred_class
            summary = [pid, start, end, row.PF]
            pred_ok = False
            for k in range(nTop):
                if k < len(pred_class):
                    pc = pred_class[k]
                    score = tr.max(scores[pred_start:pred_end, pc]).item() # maximum score in the interval
    
                    # if the correct class is in the top-n with avg score, error is not tagged
                    if (categories[pc] == row.PF) and (score>config['minscore']):
                        pred_ok = True
                        nOKs += 1
                    summary += [categories[pc], score]
                else:
                    summary += ["", ""]
            
            summary += [pred_ok]

            # Alternative predictions ########################################
                    
            # Prediction using the "area" under the curve
            if config["scoreArea"]:
                predictionArea = categories[np.argmax(pred[pred_start:pred_end].sum(axis=0))]
                # Score as the "area" under the curve of the predicted class
                # "area" is the sum of the scores of the predicted class in the interval (does not consider the whidth of the interval)
                scoreArea = pred[pred_start:pred_end].sum(axis=0).max()
                if predictionArea == row.PF:
                    nOKsArea += 1
                    pred_ok_area = True
                else:
                    pred_ok_area = False
                summary += [predictionArea, scoreArea.item(), pred_ok_area]
            
            # Prediction using the most common class along the domain
            if config["scoreCoverage"]:
                candidatesCoverage=np.argmax(pred[pred_start:pred_end],axis=1)
                # get the most common value
                countsCoverage = np.bincount(candidatesCoverage)
                predictionCoverage = categories[np.argmax(countsCoverage)]
                # Score as the proportion of domain covered by the most common class
                scoreCoverage = countsCoverage.max()/len(candidatesCoverage)
                if predictionCoverage == row.PF:
                    nOKsCoverage += 1
                    pred_ok_coverage = True
                else:
                    pred_ok_coverage = False
                
                summary += [predictionCoverage, scoreCoverage, pred_ok_coverage]
            ###################################################################
            
            errors.append(summary)
                          
        else:
            print(f"Missing index: {pid}")
            #missed_index_err.append(pid)

print("Saving errors")
cols=["PID", "start", "end", "PF"] 

for i in range(nTop):
    cols+=[f"pred{i+1}", f"score{i+1}"]

cols+=["pred_ok"]

if config["scoreArea"]:
    cols+=["pred_area", "score_area", "score_ok_area"]

if config["scoreCoverage"]:
    cols+=["pred_coverage", "score_coverage", "score_ok_coverage"]


errors = pd.DataFrame(errors, columns=cols)
errors.to_csv(f"{res_name}", index=False)

print("\n=====================================")
print(f"Top {nTop}, umbral {config['minscore']}")
print(f"Pred_OK: {nOKs},  {nOKs/len(errors)*100:6.2f}%")
print(f"Error rate: {(1-nOKs/len(errors))*100:6.2f}%")
if config["scoreArea"]:
    print(f"Error rate Area: {(1-nOKsArea/len(errors))*100:6.2f}%")
if config["scoreCoverage"]:
    print(f"Error rate Coverage: {(1-nOKsCoverage/len(errors))*100:6.2f}%")

with open(summary_file, 'a') as s:
    outputText = f"predict_errors.py soft_max {config['soft_max']}, step {config['step']}: pred_ok {nOKs/len(errors)*100:6.2f}%, error_rate {(1-nOKs/len(errors))*100:6.2f}%"
    if config["scoreArea"]:
        outputText += f", Area error: {(1-nOKsArea/len(errors))*100:6.2f}%"
    if config["scoreCoverage"]:
        outputText += f", Coverage error: {(1-nOKsCoverage/len(errors))*100:6.2f}%"
    s.write(outputText + "\n")