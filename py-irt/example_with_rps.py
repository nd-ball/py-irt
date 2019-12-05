'''
given set of observations, fit irt model and write outputs to disk
subject: train_noise
item: imageID
y: response
'''

import argparse
import csv 

import numpy as np
import pyro
import torch
import torch.nn as nn

from models.one_param_logistic import OneParamLog
from scipy.special import expit 


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--num-epochs', default=1000, type=int)
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--priors', help='[vague, hierarchical]', default='hierarchical')
parser.add_argument('--model', help='[1PL,2PL]', default='1PL')  # 2pl not implemented yet
parser.add_argument('--response-patterns', help='file with response pattern data')
args = parser.parse_args()

device = torch.device('cpu')
if args.gpu:
    device = torch.device('cuda')

# 1. load data from file 
# for this example, input file is CSV with columns modelID, itemID, response (1=correct,0=incorrect) 

models = []
items = []
responses = []

itemID2idx = {} 
idx2itemID = {} 
modelID2idx = {} 
idx2modelID = {}

m_idx = 0
i_idx = 0

with open(args.response_patterns, 'r') as infile:
    inreader = csv.reader(infile) 
    for mID, iID, _, _, response in inreader:
        if mID not in modelID2idx:
            modelID2idx[mID] = m_idx 
            idx2modelID[m_idx] = mID 
            m_idx += 1
        if iID not in itemID2idx:
            itemID2idx[iID] = i_idx 
            idx2itemID[i_idx] = iID 
            i_idx += 1
        models.append(modelID2idx[mID])
        items.append(itemID2idx[iID])  
        responses.append(eval(response)) 

num_models = len(set(models))
num_items = len(set(items))
print(num_items, num_models)

models = torch.tensor(models, dtype=torch.long, device=device) 
items = torch.tensor(items, dtype=torch.long, device=device) 
responses = torch.tensor(responses, dtype=torch.float, device=device)

# 3. define model and guide accordingly
if args.model == '1PL':
    m = OneParamLog(args.priors, device, num_items, num_models)

# 4. fit irt model with svi, trace-elbo loss
m.fit(models, items, responses, args.num_epochs) 

# 5. once model is fit, write outputs (diffs and thetas) to disk, 
#       retaining original modelIDs and itemIDs so we can use them 


for name in pyro.get_param_store().get_all_param_names():
    print(name)
    val = pyro.param(name).data.numpy()
    print(val)
    if name == 'loc_diff':
        with open(args.response_patterns + '.diffs', 'w') as outfile:
            outwriter = csv.writer(outfile, delimiter=',') 
            for i in range(len(val)):
                row = [idx2itemID[i], val[i]] 
                outwriter.writerow(row) 
    elif name == 'loc_ability':
        with open(args.response_patterns + '.theta', 'w') as outfile:
            outwriter = csv.writer(outfile, delimiter=',') 
            for i in range(len(val)):
                row = [idx2modelID[i], val[i]] 
                outwriter.writerow(row) 


