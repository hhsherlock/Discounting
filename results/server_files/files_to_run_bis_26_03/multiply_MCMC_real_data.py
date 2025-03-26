#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 6 2024

@author: Yaning
Data from Ben
Data file name: intertemporal_choice_dataset_all_trials.csv
"""

import torch
import pyro
import math
import pandas as pd
from pyro.distributions import Normal, Bernoulli, Gamma
from pyro.distributions.util import scalar_like
from pyro.infer import MCMC, NUTS, Predictive
from pyro.infer.mcmc.util import initialize_model, summary
from pyro.util import ignore_experimental_warning
from pyro import poutine
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import data_analysis_without_version.ipynb


# data[0, 1, 0]

# actions = data[0, 1, :, 4]
# delays = np.array([i / 10 for i in data[0, 1, :, 2]])
# ss_values = data[0, 1, :, 1]
# ll_values = []
# for i in range(len(ss_values)):
#     ll_values.append(ss_values[i]*data[0, 1, i, 3])
# ll_values = np.array(ll_values)

# actions = torch.tensor(actions)
# delays = torch.tensor(delays)
# ss_values = torch.tensor(ss_values)
# ll_values = torch.tensor(ll_values)

# # MCMC model
# # simulate parameters:
# # mean_u 0
# # var_u 3
# # var_es 0.5

# chain_num = 3
# sample_num = 300
# def model(actions, delays, ss_values, ll_values):
#     mean_u = pyro.sample("mean_u", Normal(loc = torch.tensor(0.), scale = torch.tensor(2.)))
#     # sigma_u = pyro.sample("sigma_u", Gamma(torch.tensor(1.), torch.tensor(2.)))
#     # sigma_es = pyro.sample("sigma_es", Gamma(torch.tensor(2.), torch.tensor(1.)))
#     log_sigma_u = pyro.sample("log_sigma_u", Normal(torch.tensor(1.), torch.tensor(2.)))
#     log_sigma_es = pyro.sample("log_sigma_es", Normal(torch.tensor(1.), torch.tensor(2.)))
#     beta = pyro.sample("beta", Gamma(torch.tensor(1.), torch.tensor(2.)))
#     num = actions.shape[0]
#     e_vals = []
#     # for i, actions, delays, ss_values, ll_values in zip(range(len(actions)), actions, delays, ss_values, ll_values):
#     #     # with pyro.plate("num", num):
#     #     e = (mean_u*delays*var_es + ll_values*var_u)/(delays*var_es + var_u)
#     #     p = torch.nn.functional.softmax(torch.tensor([e, ss_values]), dim = 0)[0]
#     #     return pyro.sample("obs", Bernoulli(probs = p), obs=actions)
#     with pyro.plate("num", num):
#         sigma_u = torch.exp(log_sigma_u)
#         sigma_es = torch.exp(log_sigma_es)
#         e_vals.append((mean_u*delays*sigma_es + ll_values*sigma_u)/(delays*sigma_es + sigma_u))
#         softmax_args = torch.stack([beta*e_vals[-1], beta*ss_values])
#         p = torch.softmax(softmax_args, dim = 0)[0]
#         return pyro.sample("obs", Bernoulli(probs = p), obs=actions)
# # pyro.render_model(model, model_args=(actions, delays, ss_values, ll_values), render_params=True, render_distributions=True)



# mcmc_kernel = NUTS(model)                                               # initialize proposal distribution kernel
# mcmc = MCMC(mcmc_kernel, num_samples=sample_num, num_chains = chain_num, warmup_steps=50)  # initialize MCMC class
# for i in range(1):
#     mcmc.run(actions, delays, ss_values, ll_values)

# test = mcmc.summary(prob=0.9)

# # get posterior with pandas dataframe
# fig, axs = plt.subplots(2, 2, figsize=(8, 8))

# mean_u = pd.DataFrame(mcmc.get_samples()['mean_u'],columns=['mean_u'])
# mean_u['chain_mean_u'] = torch.tensor([1,2,3]).repeat(300)
# sns.histplot(data=mean_u, x='mean_u',hue='chain_mean_u', palette=['white', 'pink', 'red'], ax=axs[0,0])

# log_sigma_u = pd.DataFrame(mcmc.get_samples()['log_sigma_u'],columns=['log_sigma_u'])
# log_sigma_u['chain_log_sigma_u'] = torch.tensor([1,2,3]).repeat(300)
# sns.histplot(data=log_sigma_u, x='log_sigma_u',hue='chain_log_sigma_u', palette=['white', 'skyblue', 'darkblue'], ax=axs[0,1])

# log_sigma_es = pd.DataFrame(mcmc.get_samples()['log_sigma_es'],columns=['log_sigma_es'])
# log_sigma_es['chain_log_sigma_es'] = torch.tensor([1,2,3]).repeat(300)
# sns.histplot(data=log_sigma_es, x='log_sigma_es',hue='chain_log_sigma_es', palette=['white', 'lightgreen', 'green'], ax=axs[1,0])

# beta = pd.DataFrame(mcmc.get_samples()['beta'],columns=['beta'])
# beta['chain_beta'] = torch.tensor([1,2,3]).repeat(300)
# sns.histplot(data=beta, x='beta',hue='chain_beta', palette=['white', 'lightyellow', 'yellow'], ax=axs[1,1])

# plt.show()

# results = pd.concat([mean_u, log_sigma_u, log_sigma_es, beta], axis=1, join='inner')

# results.to_csv('Documents/pyro_models/results/MCMC_mit_log.csv', index=False)