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


import pyro
import pyro.distributions as dist
import torch
import torch.distributions.constraints as constraints
from rich.console import Console

from py_irt.models.abstract_model import IrtModel

console = Console()


@IrtModel.register("multidim_2pl")
class Multidim2PL(IrtModel):
    def __init__(
        self,
        *,
        num_items: int,
        num_subjects: int,
        dims: int = 2,
        verbose=False,
        device: str = "cpu",
        **kwargs
    ):
        super().__init__(
            device=device, num_items=num_items, num_subjects=num_subjects, verbose=verbose
        )
        self.dims = dims

    def export(self):
        return {
            "ability": pyro.param("loc_ability").data.tolist(),
            "diff": pyro.param("loc_diff").data.tolist(),
            "disc": pyro.param("loc_disc").data.tolist(),
        }

    def get_model(self):
        return self.model_hierarchical

    def get_guide(self):
        return self.guide_hierarchical

    def model_hierarchical(self, subjects, items, obs):
        with pyro.plate("mu_b_plate", self.dims):
            mu_b = pyro.sample(
                "mu_b",
                dist.Normal(
                    torch.tensor(0.0, device=self.device),
                    torch.tensor(1.0e6, device=self.device),
                ),
            )

        with pyro.plate("u_b_plate", self.dims):
            u_b = pyro.sample(
                "u_b",
                dist.Gamma(
                    torch.tensor(1.0, device=self.device),
                    torch.tensor(1.0, device=self.device),
                ),
            )

        with pyro.plate("mu_theta_plate", self.dims):
            mu_theta = pyro.sample(
                "mu_theta",
                dist.Normal(
                    torch.tensor(0.0, device=self.device),
                    torch.tensor(1.0e6, device=self.device),
                ),
            )

        with pyro.plate("u_theta_plate", self.dims):
            u_theta = pyro.sample(
                "u_theta",
                dist.Gamma(
                    torch.tensor(1.0, device=self.device),
                    torch.tensor(1.0, device=self.device),
                ),
            )

        with pyro.plate("mu_gamma_plate", self.dims):
            mu_gamma = pyro.sample(
                "mu_gamma",
                dist.Normal(
                    torch.tensor(0.0, device=self.device),
                    torch.tensor(1.0e6, device=self.device),
                ),
            )

        with pyro.plate("u_gamma_plate", self.dims):
            u_gamma = pyro.sample(
                "u_gamma",
                dist.Gamma(
                    torch.tensor(1.0, device=self.device),
                    torch.tensor(1.0, device=self.device),
                ),
            )

        with pyro.plate("thetas", self.num_subjects, dim=-2, device=self.device):
            with pyro.plate("theta_dims", self.dims, dim=-1):
                ability = pyro.sample("theta", dist.Normal(mu_theta, 1.0 / u_theta))

        with pyro.plate("bs", self.num_items, dim=-2, device=self.device):
            with pyro.plate("bs_dims", self.dims, dim=-1):
                diff = pyro.sample("b", dist.Normal(mu_b, 1.0 / u_b))

        with pyro.plate("gammas", self.num_items, dim=-2, device=self.device):
            with pyro.plate("gamma_dims", self.dims, dim=-1):
                disc = pyro.sample("gamma", dist.Normal(mu_gamma, 1.0 / u_gamma))

        with pyro.plate("observe_data", obs.size(0)):
            multidim_logits = disc[items] * (ability[subjects] - diff[items])
            logits = multidim_logits.sum(axis=-1)

            pyro.sample("obs", dist.Bernoulli(logits=logits), obs=obs)

    def guide_hierarchical(self, subjects, items, obs):
        loc_mu_b_param = pyro.param("loc_mu_b", torch.zeros(self.dims, device=self.device))
        scale_mu_b_param = pyro.param(
            "scale_mu_b",
            1e2 * torch.ones(self.dims, device=self.device),
            constraint=constraints.positive,
        )

        loc_mu_theta_param = pyro.param("loc_mu_theta", torch.zeros(self.dims, device=self.device))
        scale_mu_theta_param = pyro.param(
            "scale_mu_theta",
            1e2 * torch.ones(self.dims, device=self.device),
            constraint=constraints.positive,
        )

        loc_mu_gamma_param = pyro.param("loc_mu_gamma", torch.zeros(self.dims, device=self.device))
        scale_mu_gamma_param = pyro.param(
            "scale_mu_gamma",
            1.0e2 * torch.ones(self.dims, device=self.device),
            constraint=constraints.positive,
        )

        alpha_b_param = pyro.param(
            "alpha_b",
            torch.ones(self.dims, device=self.device),
            constraint=constraints.positive,
        )
        beta_b_param = pyro.param(
            "beta_b",
            torch.ones(self.dims, device=self.device),
            constraint=constraints.positive,
        )

        alpha_theta_param = pyro.param(
            "alpha_theta",
            torch.ones(self.dims, device=self.device),
            constraint=constraints.positive,
        )
        beta_theta_param = pyro.param(
            "beta_theta",
            torch.ones(self.dims, device=self.device),
            constraint=constraints.positive,
        )

        alpha_gamma_param = pyro.param(
            "alpha_gamma",
            torch.ones(self.dims, device=self.device),
            constraint=constraints.positive,
        )
        beta_gamma_param = pyro.param(
            "beta_gamma",
            torch.ones(self.dims, device=self.device),
            constraint=constraints.positive,
        )

        m_theta_param = pyro.param(
            "loc_ability", torch.zeros([self.num_subjects, self.dims], device=self.device)
        )
        s_theta_param = pyro.param(
            "scale_ability",
            torch.ones([self.num_subjects, self.dims], device=self.device),
            constraint=constraints.positive,
        )

        m_b_param = pyro.param(
            "loc_diff", torch.zeros([self.num_items, self.dims], device=self.device)
        )
        s_b_param = pyro.param(
            "scale_diff",
            torch.ones([self.num_items, self.dims], device=self.device),
            constraint=constraints.positive,
        )

        m_gamma_param = pyro.param(
            "loc_disc", torch.zeros([self.num_items, self.dims], device=self.device)
        )
        s_gamma_param = pyro.param(
            "scale_disc",
            torch.ones([self.num_items, self.dims], device=self.device),
            constraint=constraints.positive,
        )

        # sample statements
        with pyro.plate("mu_b_plate", self.dims):
            mu_b = pyro.sample("mu_b", dist.Normal(loc_mu_b_param, scale_mu_b_param))

        with pyro.plate("u_b_plate", self.dims):
            u_b = pyro.sample("u_b", dist.Gamma(alpha_b_param, beta_b_param))

        with pyro.plate("mu_theta_plate", self.dims):
            mu_theta = pyro.sample(
                "mu_theta", dist.Normal(loc_mu_theta_param, scale_mu_theta_param)
            )
        with pyro.plate("u_theta_plate", self.dims):
            u_theta = pyro.sample("u_theta", dist.Gamma(alpha_theta_param, beta_theta_param))

        with pyro.plate("mu_gamma_plate", self.dims):
            mu_gamma = pyro.sample(
                "mu_gamma", dist.Normal(loc_mu_gamma_param, scale_mu_gamma_param)
            )

        with pyro.plate("u_gamma_plate", self.dims):
            u_gamma = pyro.sample("u_gamma", dist.Gamma(alpha_gamma_param, beta_gamma_param))

        with pyro.plate("thetas", self.num_subjects, dim=-2, device=self.device):
            with pyro.plate("theta_dims", self.dims, dim=-1):
                theta = pyro.sample("theta", dist.Normal(m_theta_param, s_theta_param))

        with pyro.plate("bs", self.num_items, dim=-2, device=self.device):
            with pyro.plate("bs_dims", self.dims, dim=-1):
                b = pyro.sample("b", dist.Normal(m_b_param, s_b_param))

        with pyro.plate("gammas", self.num_items, dim=-2, device=self.device):
            with pyro.plate("gamma_dims", self.dims, dim=-1, device=self.device):
                gamma = pyro.sample("gamma", dist.Normal(m_gamma_param, s_gamma_param))
