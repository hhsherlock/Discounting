#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 2024

@author: Yaning
"""

import torch
import pyro
import math
import pandas as pd
from pyro.distributions import Normal, Bernoulli, Gamma, Exponential
import torch.distributions.constraints as constraints
from pyro.distributions.util import scalar_like
from pyro.infer import MCMC, NUTS, Predictive
from pyro.infer.mcmc.util import initialize_model, summary
from pyro.util import ignore_experimental_warning
from pyro import poutine
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.parallel import DataParallel
import graphviz
import data_analysis_without_version as analysis

real_data = analysis.data
# real_data = real_data[:, :2]
real_data = torch.tensor(real_data).to('cuda')

real_data = torch.tensor(real_data)

# MCMC model

def model(data):
    num_context = data.shape[0]
    num_groups = data.shape[1]
    num_trials_per_group = data.shape[2]

    group_mean_u_mean = pyro.sample("group_mean_u_mean", Normal(loc = torch.tensor(0., device='cuda'), scale = torch.tensor(2., device='cuda')))
    group_mean_u_sigma = pyro.sample("group_mean_u_sigma", Gamma(torch.tensor(1., device='cuda'), torch.tensor(2., device='cuda')))
    group_log_sigma_u_mean = pyro.sample("group_log_sigma_u_mean", Normal(torch.tensor(1., device='cuda'), torch.tensor(2., device='cuda')))
    group_log_sigma_u_sigma = pyro.sample("group_log_sigma_u_sigma", Gamma(torch.tensor(1., device='cuda'), torch.tensor(2., device='cuda')))
    group_log_sigma_es_mean = pyro.sample("group_log_sigma_es_mean", Normal(torch.tensor(1., device='cuda'), torch.tensor(2., device='cuda')))
    group_log_sigma_es_sigma = pyro.sample("group_log_sigma_es_sigma", Gamma(torch.tensor(1., device='cuda'), torch.tensor(2., device='cuda')))
    group_beta_mean = pyro.sample("group_beta_mean", Gamma(torch.tensor(1., device='cuda'), torch.tensor(2., device='cuda')))
    group_beta_sigma = pyro.sample("group_beta_sigma", Gamma(torch.tensor(1., device='cuda'), torch.tensor(2., device='cuda')))
    # shift
    group_shift_mean_u_mean = pyro.sample("group_shift_mean_u_mean", Normal(loc = torch.tensor(0., device='cuda'), scale = torch.tensor(2., device='cuda')))
    group_shift_mean_u_sigma = pyro.sample("group_shift_mean_u_sigma", Gamma(torch.tensor(1., device='cuda'), torch.tensor(2., device='cuda')))
    group_shift_log_sigma_u_mean = pyro.sample("group_shift_log_sigma_u_mean", Normal(torch.tensor(1., device='cuda'), torch.tensor(2., device='cuda')))
    group_shift_log_sigma_u_sigma = pyro.sample("group_shift_log_sigma_u_sigma", Gamma(torch.tensor(1., device='cuda'), torch.tensor(2., device='cuda')))
    group_shift_log_sigma_es_mean = pyro.sample("group_shift_log_sigma_es_mean", Normal(torch.tensor(1., device='cuda'), torch.tensor(2., device='cuda')))
    group_shift_log_sigma_es_sigma = pyro.sample("group_shift_log_sigma_es_sigma", Gamma(torch.tensor(1., device='cuda'), torch.tensor(2., device='cuda')))
    # group_beta_mean = pyro.sample("group_beta_mean", Gamma(torch.tensor(1.), torch.tensor(2.)))
    # group_beta_sigma = pyro.sample("group_beta_sigma", Gamma(torch.tensor(1.), torch.tensor(2.)))

    with pyro.plate("group", num_groups):
        mean_u = pyro.sample("mean_u", Normal(group_mean_u_mean, group_mean_u_sigma))
        log_sigma_u = pyro.sample("log_sigma_u", Normal(group_log_sigma_u_mean, group_log_sigma_u_sigma))
        log_sigma_es = pyro.sample("log_sigma_es", Normal(group_log_sigma_es_mean, group_log_sigma_es_sigma))
        beta = pyro.sample("beta", Gamma(group_beta_mean, group_beta_sigma))
        # shift
        shift_mean_u = pyro.sample("shift_mean_u", Normal(group_shift_mean_u_mean, group_shift_mean_u_sigma))
        shift_log_sigma_u = pyro.sample("shift_log_sigma_u", Normal(group_shift_log_sigma_u_mean, group_shift_log_sigma_u_sigma))
        shift_log_sigma_es = pyro.sample("shift_log_sigma_es", Normal(group_shift_log_sigma_es_mean, group_shift_log_sigma_es_sigma))


    group_indices = torch.arange(num_groups).unsqueeze(1).repeat(1, num_trials_per_group).reshape(-1).to('cuda')
    group_indices = torch.cat((group_indices, group_indices)).to('cuda')
    
    with pyro.plate("data", num_groups*num_trials_per_group*num_context):

        sigma_u = torch.exp(log_sigma_u[group_indices] + data[:,:,:,-1].view(-1)*shift_log_sigma_u[group_indices])

        sigma_es = torch.exp(log_sigma_es[group_indices] + data[:,:,:,-1].view(-1)*shift_log_sigma_es[group_indices])
        e_val = ((mean_u[group_indices] + data[:,:,:,-1].view(-1)*shift_mean_u[group_indices])*data[:,:,:,2].view(-1)*sigma_es**2 + 
                 data[:,:,:,3].view(-1)*sigma_u**2)/(data[:,:,:,2].view(-1)*sigma_es**2 + sigma_u**2)
        softmax_args = torch.stack([beta[group_indices]*e_val, beta[group_indices]*data[:,:,:,1].view(-1)])
        p = torch.softmax(softmax_args, dim = 0)[0]

        pyro.sample("obs", Bernoulli(probs = p), obs=data[:,:,:,4].view(-1))


chain_num = 1
sample_num = 4000

# model = model(real_data)
# Ben's reinforcement learning iteration 60,000 
# burin 45,000
mcmc_kernel = NUTS(model)                                               # initialize proposal distribution kernel
mcmc = MCMC(mcmc_kernel, num_samples=sample_num, num_chains = chain_num, warmup_steps=2000)  # initialize MCMC class
mcmc.run(real_data)

# get results summary
summary_dict = mcmc.summary(prob=0.9)
torch.save(summary_dict, 'results/shift_summary_0.pth')

# get all the samples
samples_dict = mcmc.get_samples()
torch.save(samples_dict, 'results/shift_samples_0.pth')