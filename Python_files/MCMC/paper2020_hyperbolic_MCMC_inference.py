#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 2023

@author: Yaning
"""

import math
import paper2020_hyperbolic_MCMC_sim as sim
import pandas as pd
import torch

import pyro
from pyro.distributions import Beta, Binomial, HalfCauchy, Normal, Pareto, Uniform
from pyro.distributions.util import scalar_like
from pyro.infer import MCMC, NUTS, Predictive
from pyro.infer.mcmc.util import initialize_model, summary
from pyro.util import ignore_experimental_warning
from pyro import poutine
import seaborn as sns
import matplotlib.pyplot as plt

actions = torch.tensor(sim.inference_actions)
delays = torch.tensor(sim.inference_delays)
ss_values = torch.tensor(sim.inference_ss_values)
ll_values = torch.tensor(sim.inference_ll_values)
# print(actions)
# print(delays)
# print(ss_values)
# print(ll_values)

def model(actions, delays, ss_values, ll_values):
    estim_dev = pyro.sample("estim_dev", HalfCauchy(scale=torch.tensor([4.])))
    prior_dev = pyro.sample("prior_dev", HalfCauchy(scale= torch.tensor([1.])))
    num = actions.shape[0]
    with pyro.plate("num", num):
        mean = ll_values/(1+(estim_dev/prior_dev)*delays)
        distri = Normal(loc = mean, scale = estim_dev)
        pos = 1 - distri.cdf(ss_values)
        return pyro.sample("obs", Binomial(probs = pos), obs=actions)

# pyro.render_model(model, model_args=(actions, delays, ss_values, ll_values), render_params=True, render_distributions=True)

mcmc_kernel = NUTS(model)                                               # initialize proposal distribution kernel
mcmc = MCMC(mcmc_kernel, num_samples=300, num_chains = 3, warmup_steps=50)  # initialize MCMC class
mcmc.run(actions, delays, ss_values, ll_values)

# mcmc.summary()
fig, axs = plt.subplots(1, 2, figsize=(8, 3))

rate = pd.DataFrame(mcmc.get_samples()['estim_dev'],columns=['estim_dev'])
rate['chain'] = torch.tensor([1,2,3]).repeat(300)
sns.histplot(data=rate, x='estim_dev',hue='chain', palette=['white', 'pink', 'red'], ax=axs[0])

dev = pd.DataFrame(mcmc.get_samples()['prior_dev'],columns=['prior_dev'])
dev['chain'] = torch.tensor([1,2,3]).repeat(300)
sns.histplot(data=dev, x='prior_dev',hue='chain', palette=['white', 'skyblue', 'darkblue'], ax=axs[1])

plt.show()