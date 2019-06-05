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

from models.irt import OneParamLog
from scipy.special import expit 


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--infile', help='response pattern file')
parser.add_argument('-e', '--num-epochs', default=1000, type=int)
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--priors', help='[vague, hierarchical]', default='hierarchical')
parser.add_argument('--model', help='[1PL,2PL]', default='1PL')
args = parser.parse_args()

device = torch.device('cpu')
if args.gpu:
    device = torch.device('cuda')

# 1. load data from file 
# 2. combine into obs 3-tuples

models = []
items = []
responses = []

real_theta = np.random.normal(size=[50])
real_diff = np.random.normal(size=[50])

obs = []
for i in range(len(real_theta)):
    for j in range(len(real_diff)):
        y = np.random.binomial(1, expit(real_theta[i] - real_diff[j]))
        models.append(i) 
        items.append(j) 
        responses.append(y) 

print(real_theta)
print(real_diff)
print(responses) 

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
        print('mse: {}'.format(np.mean((val - real_diff) ** 2)))
    elif name == 'loc_ability':
        print('mse: {}'.format(np.mean((val - real_theta) ** 2)))


