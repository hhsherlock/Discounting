#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 2024

@author: Yaning
"""

import torch
import pyro

from pyro.distributions import Normal, Bernoulli, Gamma

# MCMC model

def model(data):
    num_groups = data.shape[0]
    num_trials_per_group = data.shape[1]

    group_mean_u_mean = pyro.sample("group_mean_u_mean", Normal(loc = torch.tensor(0.), scale = torch.tensor(2.)))
    group_mean_u_sigma = pyro.sample("group_mean_u_sigma", Gamma(torch.tensor(1.), torch.tensor(2.)))
    group_log_sigma_u_mean = pyro.sample("group_log_sigma_u_mean", Normal(torch.tensor(1.), torch.tensor(2.)))
    group_log_sigma_u_sigma = pyro.sample("group_log_sigma_u_sigma", Gamma(torch.tensor(1.), torch.tensor(2.)))
    group_log_sigma_es_mean = pyro.sample("group_log_sigma_es_mean", Normal(torch.tensor(1.), torch.tensor(2.)))
    group_log_sigma_es_sigma = pyro.sample("group_log_sigma_es_sigma", Gamma(torch.tensor(1.), torch.tensor(2.)))
    group_beta_mean = pyro.sample("group_beta_mean", Normal(torch.tensor(1.), torch.tensor(2.)))
    group_beta_sigma = pyro.sample("group_beta_sigma", Gamma(torch.tensor(1.), torch.tensor(2.)))

    with pyro.plate("group", num_groups):
        mean_u = pyro.sample("mean_u", Normal(group_mean_u_mean, group_mean_u_sigma))
        log_sigma_u = pyro.sample("log_sigma_u", Normal(group_log_sigma_u_mean, group_log_sigma_u_sigma))
        log_sigma_es = pyro.sample("log_sigma_es", Normal(group_log_sigma_es_mean, group_log_sigma_es_sigma))
        beta = pyro.sample("beta", Normal(group_beta_mean, group_beta_sigma))

    group_indices = torch.arange(num_groups).unsqueeze(1).repeat(1, num_trials_per_group).reshape(-1)
    
    with pyro.plate("data", num_groups*num_trials_per_group):
        
        sigma_u = torch.exp(log_sigma_u[group_indices])
        sigma_es = torch.exp(log_sigma_es[group_indices])
        e_val = (mean_u[group_indices]*data[:,:,2].view(-1)*sigma_es**2 + 
                 data[:,:,3].view(-1)*sigma_u**2)/(data[:,:,2].view(-1)*sigma_es**2 + sigma_u**2)
        softmax_args = torch.stack([beta[group_indices]*e_val, beta[group_indices]*data[:,:,1].view(-1)])
        p = torch.softmax(softmax_args, dim = 0)[0]
        pyro.sample("obs", Bernoulli(probs = p), obs=data[:,:,4].view(-1))
            

# pyro.render_model(model, model_args=(actions, delays, ss_values, ll_values), render_params=True, render_distributions=True)