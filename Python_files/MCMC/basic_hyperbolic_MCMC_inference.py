#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 2023

@author: Yaning
"""

import math
import basic_hyperbolic_MCMC_sim as sim
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
    dev = pyro.sample("dev", HalfCauchy(scale=torch.tensor([4.])))
    rate = pyro.sample("rate", HalfCauchy(scale= torch.tensor([1.])))
    num = actions.shape[0]
    with pyro.plate("num", num):
        mean = ll_values/(1+rate*delays)
        distri = Normal(loc = mean, scale = dev)
        pos = 1 - distri.cdf(ss_values)
        return pyro.sample("obs", Binomial(probs = pos), obs=actions)

# pyro.render_model(model, model_args=(actions, delays, ss_values, ll_values), render_params=True, render_distributions=True)



mcmc_kernel = NUTS(model)                                               # initialize proposal distribution kernel
mcmc = MCMC(mcmc_kernel, num_samples=300, num_chains = 3, warmup_steps=50)  # initialize MCMC class
mcmc.run(actions, delays, ss_values, ll_values)

# mcmc.summary()
fig, axs = plt.subplots(1, 2, figsize=(8, 3))

rate = pd.DataFrame(mcmc.get_samples()['rate'],columns=['rate'])
rate['chain'] = torch.tensor([1,2,3]).repeat(300)
sns.histplot(data=rate, x='rate',hue='chain', palette=['white', 'pink', 'red'], ax=axs[0])

dev = pd.DataFrame(mcmc.get_samples()['dev'],columns=['dev'])
dev['chain'] = torch.tensor([1,2,3]).repeat(300)
sns.histplot(data=dev, x='dev',hue='chain', palette=['white', 'skyblue', 'darkblue'], ax=axs[1])

plt.show()