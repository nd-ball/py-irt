# MIT License

# Copyright (c) 2019 John Lalor <john.lalor@nd.edu> and Pedro Rodriguez <me@pedro.ai>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from py_irt.models import abstract_model
import pyro
import pyro.distributions as dist
import torch

import torch.distributions.constraints as constraints

from pyro.infer import SVI, Trace_ELBO, EmpiricalMarginal, TraceEnum_ELBO
from pyro.infer.mcmc import MCMC, NUTS
from pyro.optim import Adam, SGD

import pyro.contrib.autoguide as autoguide

import pandas as pd

from functools import partial

import numpy as np

import torch.nn as nn
import torch.nn.functional as F



# building off of ProdLDA model for amortization (text only for now) 
# https://pyro.ai/examples/prodlda.html
class Encoder(nn.Module):
    # Base class for the encoder net, used in the guide
    def __init__(self, vocab_size, num_dimensions, hidden, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)  # to avoid component collapse
        self.fc1 = nn.Linear(vocab_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fcmu = nn.Linear(hidden, num_dimensions)
        self.fclv = nn.Linear(hidden, num_dimensions)
        # NB: here we set `affine=False` to reduce the number of learning parameters
        # See https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
        # for the effect of this flag in BatchNorm1d
        self.bnmu = nn.BatchNorm1d(num_dimensions, affine=False)  # to avoid component collapse
        self.bnlv = nn.BatchNorm1d(num_dimensions, affine=False)  # to avoid component collapse

    def forward(self, inputs):
        h = F.softplus(self.fc1(inputs))
        h = F.softplus(self.fc2(h))
        h = self.drop(h)
        # μ and Σ are the outputs
        logtheta_loc = self.bnmu(self.fcmu(h))
        logtheta_logvar = self.bnlv(self.fclv(h))
        logtheta_scale = (0.5 * logtheta_logvar).exp()  # Enforces positivity
        return logtheta_loc, logtheta_scale


class Decoder(nn.Module):
    # Base class for the decoder net, used in the model
    def __init__(self, vocab_size, num_dimensions, dropout):
        super().__init__()
        self.beta = nn.Linear(num_dimensions, vocab_size, bias=False)
        self.bn = nn.BatchNorm1d(vocab_size, affine=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        inputs = self.drop(inputs)
        # the output is σ(βθ)
        return F.softmax(self.bn(self.beta(inputs)), dim=1)


@abstract_model.IrtModel.register("amortized_1pl")
class Amortized1PL(abstract_model.IrtModel):
    
    def __init__(
        self, *, 
        priors: str, 
        num_items: int, 
        num_subjects: int, 
        verbose: bool = False, 
        device: str = "cpu",
        vocab_size: int,
        dropout: float,
        hidden: int,
        **kwargs
    ):
        super().__init__(
            device=device, num_items=num_items, num_subjects=num_subjects, verbose=verbose
        )

        # initialize the class with all arguments provided to the constructor
        self.num_dimensions = 1
        self.device = torch.device(device)
        self.drop = dropout
        self.hidden = hidden
        self.num_items = num_items

        self.vocab_size = vocab_size
        self.encoder = Encoder(vocab_size, self.num_dimensions, self.hidden, self.drop)
        self.decoder = Decoder(vocab_size, self.num_dimensions, self.drop)
    

    def model_irt(self, models, items, obs):
        num_items = len(items)
        options = dict(dtype=torch.float64, device=self.device)
        #xs = torch.flatten(items, start_dim=1)
        xs = items
        models = torch.tensor(models, dtype=torch.long, device=items.device)
        items = torch.tensor(items, dtype=torch.long, device=items.device)
        obs = torch.tensor(obs, dtype=torch.float, device=items.device)

        with pyro.plate("thetas"):
            ability = pyro.sample('theta', dist.Normal(torch.zeros(self.num_subjects, **options),
                torch.ones(self.num_subjects, **options)))
        with pyro.plate("diffs", num_items):
            # sample the item difficulty from the prior distribution
            diff_prior_loc = torch.zeros(num_items, **options).unsqueeze(1).float()
            diff_prior_scale = torch.ones(num_items, **options).fill_(1.e3).unsqueeze(1).float()
            diff = pyro.sample('b', dist.Normal(diff_prior_loc, diff_prior_scale).to_event(1))
            loc = self.decoder.forward(diff)
            total_count = int(xs.sum(-1).max())
            pyro.sample(
                'items',
                dist.Multinomial(total_count, loc),
                obs=items
            )
            #diff = pyro.sample('b', dist.Normal(torch.zeros(num_items, **options),
            #    torch.tensor(num_items, **options).fill_(1.e-3)))

        with pyro.plate("data", len(obs)):
            pyro.sample("obs", dist.Bernoulli(logits=ability[models] - diff).to_event(1), obs=obs)
        
    def guide_irt(self, models, items, obs):
        num_items = len(items)
        options = dict(dtype=torch.float64, device=self.device)
        #xs = torch.flatten(items, start_dim=1)
        xs = items
        # vectorize
        models = torch.tensor(models, dtype=torch.long, device=self.device)
        items = torch.tensor(items, dtype=torch.float, device=self.device)
        obs = torch.tensor(obs, dtype=torch.float, device=self.device)


        # register learnable params in the param store
        with pyro.plate("thetas"):
            m_theta_param = pyro.param("loc_ability", torch.zeros(self.num_subjects, **options))
            s_theta_param = pyro.param("scale_ability", torch.ones(self.num_subjects, **options),
                            constraint=constraints.positive)
            dist_theta = dist.Normal(m_theta_param, s_theta_param)
            pyro.sample("theta", dist_theta)

        # items 
        with pyro.plate("diffs", num_items):
            irt_batch_size = 256
            loc_diffs_all, scale_diffs_all = [], []
            for i in range(0, len(items), irt_batch_size):
                # pick out the appropriate images from xs based on items idx
                batch_xs = items[i:i+irt_batch_size]
                loc_diffs, scale_diffs = self.encoder.forward(batch_xs)            
                loc_diffs_all.extend(loc_diffs)
                scale_diffs_all.extend(scale_diffs)
            loc_diffs_all = torch.tensor(loc_diffs_all, **options).unsqueeze(1).float()
            scale_diffs_all = torch.tensor(scale_diffs_all, **options).unsqueeze(1).float()
            dist_b = dist.Normal(loc_diffs_all, scale_diffs_all)
            pyro.sample('b', dist_b.to_event(1))

    def get_model(self):
        return self.model_irt

    def get_guide(self):
        return self.guide_irt

    def fit(self, models, items, responses, num_epochs):
        """Fit the IRT model with variational inference"""
        # need to step with IRT loss and with reconstruction loss

        optim = Adam({"lr": 0.1})
        svi = SVI(self.model_irt, self.guide_irt, optim, loss=Trace_ELBO())
        #svi_diff = SVI(self.model, self.guide, optim, loss=Trace_ELBO())

        pyro.clear_param_store()
        for j in range(num_epochs):
            loss = svi.step(models, items, responses)
            #loss_diff = svi_diff.step(items)
            if j % 100 == 0 and self.verbose:
                print("[epoch %04d] irt loss: %.4f" % (j + 1, loss))
                #print("[epoch %04d] recon loss: %.4f" % (j + 1, loss_diff))

        print("[epoch %04d] loss: %.4f" % (j + 1, loss))
        #print("[epoch %04d] loss: %.4f" % (j + 1, loss_diff))
        values = ["loc_diff", "scale_diff", "loc_ability", "scale_ability"]

    def export(self, items):
        items = torch.tensor(items, dtype=torch.float)
        diffs, _ = self.encoder.forward(items)
        diffs = diffs.squeeze().detach().numpy()

        return {
            "ability": pyro.param("loc_ability").data.tolist(),
            "diff": diffs.tolist()
        }

    def fit_MCMC(self, models, items, responses, num_epochs):
        """Fit the IRT model with MCMC"""
        sites = ["theta", "b"]
        nuts_kernel = NUTS(self.model_vague, adapt_step_size=True)
        hmc_posterior = MCMC(nuts_kernel, num_samples=1000, warmup_steps=100).run(
            models, items, responses
        )
        theta_sum = self.summary(hmc_posterior, ["theta"]).items()
        b_sum = self.summary(hmc_posterior, ["b"]).items()
        
    def predict(self, subjects, items, params_from_file=None):
        """predict p(correct | params) for a specified list of model, item pairs"""
        if params_from_file is not None:
            model_params = params_from_file
        else:
            model_params = self.export(items)
        abilities = np.array([model_params["ability"][i] for i in subjects])
        diffs = np.array(model_params["diff"])
        #items = torch.tensor(items, dtype=torch.float)
        #diffs, _ = self.encoder.forward(items)
        #diffs = diffs.squeeze().detach().numpy()
        return 1 / (1 + np.exp(-(abilities - diffs)))

    def summary(self, traces, sites):
        """Aggregate marginals for MCM"""
        marginal = (
            EmpiricalMarginal(traces, sites)._get_samples_and_weights()[0].detach().cpu().numpy()
        )
        print(marginal)
        site_stats = {}
        for i in range(marginal.shape[1]):
            site_name = sites[i]
            marginal_site = pd.DataFrame(marginal[:, i]).transpose()
            describe = partial(pd.Series.describe, percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
            site_stats[site_name] = marginal_site.apply(describe, axis=1)[
                ["mean", "std", "5%", "25%", "50%", "75%", "95%"]
            ]
        return site_stats
