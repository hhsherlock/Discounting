#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 2024

@author: Yaning
"""

import torch
import torch.distributions.constraints as constraints
import pyro
from pyro.distributions import Normal, Bernoulli


def model_normal_log(actions, delays, ss_values, ll_values):
    
    mean_u_mean_q = pyro.param("mean_u_mean_q", torch.tensor(0.))
    mean_u_sigma_q = pyro.param("mean_u_sigma_q", torch.tensor(1.))
    sigma_u_mean_q =  pyro.param("sigma_u_mean_q", torch.tensor(1.))
    sigma_u_sigma_q =  pyro.param("sigma_u_sigma_q", torch.tensor(2.))
    sigma_es_mean_q = pyro.param("sigma_es_mean_q", torch.tensor(1.))
    sigma_es_sigma_q = pyro.param("sigma_es_sigma_q", torch.tensor(2.))

    # sample from the HalfCauchy prior and normal distributions
    mean_u = pyro.sample("mean_u", Normal(loc=mean_u_mean_q, scale=mean_u_sigma_q))
    log_sigma_u = pyro.sample("log_sigma_u", Normal(loc=sigma_u_mean_q, scale = sigma_u_sigma_q))
    log_sigma_es = pyro.sample("log_sigma_es", Normal(loc=sigma_es_mean_q, scale = sigma_es_sigma_q))
    # loop over the observed data
    e_vals = []
    v_vals = []
    for i in range(len(actions)):
        sigma_u = torch.exp(log_sigma_u)
        sigma_es = torch.exp(log_sigma_es)
        e_vals.append((mean_u*delays[i]*sigma_es**2 + ll_values[i]*sigma_u**2)/
                      (delays[i]*sigma_es**2 + sigma_u**2))
        softmax_args = torch.stack([e_vals[-1], ss_values[i]])
        p = torch.softmax(softmax_args, dim = 0)[0]
        pyro.sample("obs_" + str(i), Bernoulli(probs = p), obs=actions[i])

def guide_normal_log(actions, delays, ss_values, ll_values):
    # get rid of the constrains
    mean_u_mean_q = pyro.param("mean_u_mean_q", torch.tensor(0.))
    mean_u_sigma_q = pyro.param("mean_u_sigma_q", torch.tensor(1.),
                     constraint=constraints.positive)
    sigma_u_mean_q = pyro.param("sigma_u_mean_q", torch.tensor(2.))
    sigma_u_sigma_q = pyro.param("sigma_u_sigma_q", torch.tensor(1.),
                     constraint=constraints.positive)
    sigma_es_mean_q = pyro.param("sigma_es_mean_q", torch.tensor(1.))
    sigma_es_sigma_q = pyro.param("sigma_es_sigma_q", torch.tensor(2.),
                     constraint=constraints.positive)

    # sample from the HalfCauchy prior and normal distributions
    mean_u = pyro.sample("mean_u", Normal(loc=mean_u_mean_q, scale=mean_u_sigma_q))
    log_sigma_u = pyro.sample("log_sigma_u", Normal(sigma_u_mean_q, sigma_u_sigma_q))
    log_sigma_es = pyro.sample("log_sigma_es", Normal(sigma_es_mean_q, sigma_es_sigma_q))