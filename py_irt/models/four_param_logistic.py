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


# pylint: disable=unused-argument,unused-variable,not-callable,no-name-in-module,no-member,protected-access
from functools import partial
from py_irt.models import abstract_model

import pandas as pd
import pyro
import pyro.distributions as dist
import torch
import torch.distributions.constraints as constraints
from pyro.infer import EmpiricalMarginal
from rich.console import Console

import numpy as np

console = Console()


@abstract_model.IrtModel.register("4pl")
class FourParamLog(abstract_model.IrtModel):
    """4PL IRT Model"""

    # pylint: disable=not-callable
    def __init__(
        self, 
        *, 
        device: str, 
        num_items: int, 
        num_subjects: int, 
        verbose: bool = False,
        **kwargs):
        super().__init__(
            num_items=num_items, num_subjects=num_subjects, device=device, verbose=verbose
        )

    def model_hierarchical(self, subjects, items, obs):
        mu_b = pyro.sample(
            "mu_b",
            dist.Normal(
                torch.tensor(0.0, device=self.device),
                torch.tensor(1.0e6, device=self.device),
            ),
        )
        u_b = pyro.sample(
            "u_b",
            dist.Gamma(
                torch.tensor(1.0, device=self.device),
                torch.tensor(1.0, device=self.device),
            ),
        )

        mu_theta = pyro.sample(
            "mu_theta",
            dist.Normal(
                torch.tensor(0.0, device=self.device),
                torch.tensor(1.0e6, device=self.device),
            ),
        )
        u_theta = pyro.sample(
            "u_theta",
            dist.Gamma(
                torch.tensor(1.0, device=self.device),
                torch.tensor(1.0, device=self.device),
            ),
        )

        mu_gamma = pyro.sample(
            "mu_gamma",
            dist.Normal(
                torch.tensor(0.0, device=self.device),
                torch.tensor(1.0e6, device=self.device),
            ),
        )
        u_gamma = pyro.sample(
            "u_gamma",
            dist.Gamma(
                torch.tensor(1.0, device=self.device),
                torch.tensor(1.0, device=self.device),
            ),
        )

        # Fraction of feasible: Simple variable to be fit
        lambdas = pyro.param(
            "lambdas",
            torch.ones(self.num_items, device=self.device),
            constraint=constraints.unit_interval,
        )

        with pyro.plate("thetas", self.num_subjects, device=self.device):
            ability = pyro.sample("theta", dist.Normal(mu_theta, 1.0 / u_theta))

        with pyro.plate("bs", self.num_items, device=self.device):
            diff = pyro.sample("b", dist.Normal(mu_b, 1.0 / u_b))

        with pyro.plate("gammas", self.num_items, device=self.device):
            disc = pyro.sample("gamma", dist.Normal(mu_gamma, 1.0 / u_gamma))

        with pyro.plate("observe_data", obs.size(0)):
            p_star = torch.sigmoid(disc[items] * (ability[subjects] - diff[items]))
            pyro.sample(
                "obs",
                dist.Bernoulli(probs=lambdas[items] * p_star),
                obs=obs,
            )

    def guide_hierarchical(self, subjects, items, obs):
        loc_mu_b_param = pyro.param("loc_mu_b", torch.tensor(0.0, device=self.device))
        scale_mu_b_param = pyro.param(
            "scale_mu_b",
            torch.tensor(1.0e2, device=self.device),
            constraint=constraints.positive,
        )
        loc_mu_gamma_param = pyro.param("loc_mu_gamma", torch.tensor(0.0, device=self.device))
        scale_mu_gamma_param = pyro.param(
            "scale_mu_gamma",
            torch.tensor(1.0e2, device=self.device),
            constraint=constraints.positive,
        )
        loc_mu_theta_param = pyro.param("loc_mu_theta", torch.tensor(0.0, device=self.device))
        scale_mu_theta_param = pyro.param(
            "scale_mu_theta",
            torch.tensor(1.0e2, device=self.device),
            constraint=constraints.positive,
        )
        alpha_b_param = pyro.param(
            "alpha_b",
            torch.tensor(1.0, device=self.device),
            constraint=constraints.positive,
        )
        beta_b_param = pyro.param(
            "beta_b",
            torch.tensor(1.0, device=self.device),
            constraint=constraints.positive,
        )
        alpha_gamma_param = pyro.param(
            "alpha_gamma",
            torch.tensor(1.0, device=self.device),
            constraint=constraints.positive,
        )
        beta_gamma_param = pyro.param(
            "beta_gamma",
            torch.tensor(1.0, device=self.device),
            constraint=constraints.positive,
        )
        alpha_theta_param = pyro.param(
            "alpha_theta",
            torch.tensor(1.0, device=self.device),
            constraint=constraints.positive,
        )
        beta_theta_param = pyro.param(
            "beta_theta",
            torch.tensor(1.0, device=self.device),
            constraint=constraints.positive,
        )
        m_theta_param = pyro.param(
            "loc_ability", torch.zeros(self.num_subjects, device=self.device)
        )
        s_theta_param = pyro.param(
            "scale_ability",
            torch.ones(self.num_subjects, device=self.device),
            constraint=constraints.positive,
        )
        m_b_param = pyro.param("loc_diff", torch.zeros(self.num_items, device=self.device))
        s_b_param = pyro.param(
            "scale_diff",
            torch.ones(self.num_items, device=self.device),
            constraint=constraints.positive,
        )
        m_gamma_param = pyro.param("loc_disc", torch.zeros(self.num_items, device=self.device))
        s_gamma_param = pyro.param(
            "scale_disc",
            torch.ones(self.num_items, device=self.device),
            constraint=constraints.positive,
        )

        # sample statements
        mu_b = pyro.sample("mu_b", dist.Normal(loc_mu_b_param, scale_mu_b_param))
        u_b = pyro.sample("u_b", dist.Gamma(alpha_b_param, beta_b_param))

        mu_gamma = pyro.sample("mu_gamma", dist.Normal(loc_mu_gamma_param, scale_mu_gamma_param))
        u_gamma = pyro.sample("u_gamma", dist.Gamma(alpha_gamma_param, beta_gamma_param))

        mu_theta = pyro.sample("mu_theta", dist.Normal(loc_mu_theta_param, scale_mu_theta_param))
        u_theta = pyro.sample("u_theta", dist.Gamma(alpha_theta_param, beta_theta_param))

        with pyro.plate("thetas", self.num_subjects, device=self.device):
            pyro.sample("theta", dist.Normal(m_theta_param, s_theta_param))

        with pyro.plate("bs", self.num_items, device=self.device):
            pyro.sample("b", dist.Normal(m_b_param, s_b_param))

        with pyro.plate("gammas", self.num_items, device=self.device):
            pyro.sample("gamma", dist.Normal(m_gamma_param, s_gamma_param))

    def export(self):
        return {
            "ability": pyro.param("loc_ability").data.tolist(),
            "diff": pyro.param("loc_diff").data.tolist(),
            "disc": pyro.param("loc_disc").data.tolist(),
            "lambdas": pyro.param("lambdas").data.tolist(),
        }

    def predict(self, subjects, items, params_from_file=None):
        """predict p(correct | params) for a specified list of model, item pairs"""
        if params_from_file is not None:
            model_params = params_from_file
        else:
            model_params = self.export()
        abilities = np.array([model_params["ability"][i] for i in subjects])
        diffs = np.array([model_params["diff"][i] for i in items])
        discs = np.array([model_params["disc"][i] for i in items])
        lambdas = np.array([model_params["lambdas"][i] for i in items])
        return lambdas / (1 + np.exp(-discs * (abilities - diffs)))

    def get_guide(self):
        return self.guide_hierarchical

    def get_model(self):
        return self.model_hierarchical

    def summary(self, traces, sites):
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
