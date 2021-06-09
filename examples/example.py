"""
given set of observations, fit irt model and write outputs to disk
subject: train_noise
item: imageID
y: response
"""

import argparse
import csv

import numpy as np
import pyro
import torch
import torch.nn as nn

from py_irt.models.one_param_logistic import OneParamLog
from py_irt.models.two_param_logistic import TwoParamLog
from scipy.special import expit


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--num-epochs", default=1000, type=int)
parser.add_argument("--gpu", action="store_true")
parser.add_argument("-v", "--verbose", action="store_true")
args = parser.parse_args()

device = torch.device("cpu")
if args.gpu:
    device = torch.device("cuda")

# 1. load data from file
# 2. combine into obs 3-tuples

models = []
items = []
responses = []

num_subjects = 2000
num_items = 100

real_theta = np.random.normal(size=[num_subjects])
real_diff = np.random.normal(size=[num_items])
real_slope = np.random.normal(size=[num_items])
print(real_slope)

obs = []
for i in range(len(real_theta)):
    for j in range(len(real_diff)):
        y = np.random.binomial(1, expit(real_slope[j] * (real_theta[i] - real_diff[j])))
        models.append(i)
        items.append(j)
        responses.append(y)

with open("test_data_params.csv", "w") as outfile:
    D = np.zeros((len(real_theta), len(real_diff)))
    for i in range(len(responses)):
        model = models[i]
        item = items[i]
        response = responses[i]
        D[model, item] = response
    np.savetxt(outfile, D, delimiter=",", fmt="%.1f")

with open("test_data_params.knowndiffs.csv", "w") as outfile:
    np.savetxt(outfile, real_diff, delimiter="\n", fmt="%.5f")
with open("test_data_params.knownthetas.csv", "w") as outfile:
    np.savetxt(outfile, real_theta, delimiter="\n", fmt="%.5f")
with open("test_data_params.knownslopes.csv", "w") as outfile:
    np.savetxt(outfile, real_slope, delimiter="\n", fmt="%.5f")


# print(real_theta)
# print(real_diff)
# print(real_slope)
# print(responses)

num_subjects = len(set(models))
num_items = len(set(items))
# print(num_items, num_subjects)

models = torch.tensor(models, dtype=torch.long, device=device)
items = torch.tensor(items, dtype=torch.long, device=device)
responses = torch.tensor(responses, dtype=torch.float, device=device)


# 3. define model and guide accordingly
m1v = OneParamLog("vague", device, num_items, num_subjects, args.verbose)
m1h = OneParamLog("hierarchical", device, num_items, num_subjects, args.verbose)
m2v = TwoParamLog("vague", device, num_items, num_subjects, args.verbose)
m2h = TwoParamLog("hierarchical", device, num_items, num_subjects, args.verbose)


for m in [m1v, m2v, m1h, m2h]:
    # 4. fit irt model with svi, trace-elbo loss
    m.fit(models, items, responses, args.num_epochs)
    # 5. once model is fit, write outputs (diffs and thetas) to disk,
    #       retaining original modelIDs and itemIDs so we can use them

    for name in pyro.get_param_store().get_all_param_names():
        if name not in ["loc_diff", "loc_ability", "loc_slope"]:
            continue
        print(name)
        val = pyro.param(name).data.numpy()
        if args.verbose:
            print(val)
        if name == "loc_diff":
            print("rmse: {}".format(np.sqrt(np.mean((val - real_diff) ** 2))))
        elif name == "loc_ability":
            print("rmse: {}".format(np.sqrt(np.mean((val - real_theta) ** 2))))
        elif name == "loc_slope":
            print("rmse: {}".format(np.sqrt(np.mean((val - real_slope) ** 2))))
            if args.verbose:
                print(val)
