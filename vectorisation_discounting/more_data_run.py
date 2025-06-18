#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Mon Jun 16 2025
    
    @author: Yaning
"""

import os
import torch
import pyro
from pyro.optim import Adam
import pyro.distributions as dist
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
import pickle
from pyro.infer import SVI, Trace_ELBO
import re
import gc
import time

from torch.profiler import profile, ProfilerActivity

# device = torch.device("cuda")
device = torch.device("cpu")

with open('/home/yaning/Documents/Discounting/results/array_buhui.pkl', 'rb') as f:
    single_data = pickle.load(f)

single_data = torch.tensor(single_data)
single_data = single_data.to(device)

# # flatten
# def model(data):
#     num_params = 4
#     num_agents = data.shape[0]
#     num_trials = data.shape[1]
#     # define hyper priors over model parameters
#     # prior over sigma of a Gaussian is a Gamma distribution
#     a = pyro.param('a', torch.ones(num_params, device=device), constraint=dist.constraints.positive)
#     lam = pyro.param('lam', torch.ones(num_params, device=device), constraint=dist.constraints.positive)
#     tau = pyro.sample('tau', dist.Gamma(a, a/lam).to_event(1)) # mean = a / (a/lam) = lam

#     sig = pyro.deterministic('sig', 1/torch.sqrt(tau)) # Gauss sigma

#     # each model parameter has a hyperprior defining group level mean
#     # in the form of a Normal distribution
#     m = pyro.param('m', torch.zeros(num_params, device=device))
#     s = pyro.param('s', torch.ones(num_params, device=device), constraint=dist.constraints.positive)
#     mu = pyro.sample('mu', dist.Normal(m, s*sig).to_event(1)) # Gauss mu, wieso s*sig?

#     # in order to implement groups, where each subject is independent of the others, pyro uses so-called plates.
#     # you embed what should be done for each subject into the "with pyro.plate" context
#     # the plate vectorizes subjects and adds an additional dimension onto all arrays/tensors
#     # i.e. p1 below will have the length num_agents
#     with pyro.plate('ag_idx', num_agents):
#         # draw parameters from Normal and transform (for numeric trick reasons)
#         # base_dist = dist.Normal(0., 1.).expand_by([num_params]).to_event(1)
#         base_dist = dist.Normal(torch.zeros(num_params, device=device), torch.ones(num_params, device=device)).to_event(1)
#         # Transform via the pointwise affine mapping y = loc + scale*x (-> Neal's funnel)
#         transform = dist.transforms.AffineTransform(mu, sig)
#         locs = pyro.sample('locs', dist.TransformedDistribution(base_dist, [transform]))

#     group_indices = torch.arange(num_agents).unsqueeze(1).repeat(1, num_trials).reshape(-1)
#     # mean_u = torch.full((num_agents*num_trials,),0.001)

#     with pyro.plate('data', num_agents*num_trials):
#         sigma_rate = torch.exp(locs[:,0])[group_indices]
#         a = torch.exp(locs[:,1])[group_indices]
#         b = torch.exp(locs[:,2])[group_indices]
#         beta = torch.exp(locs[:,3])[group_indices]

#         sigma_combine = sigma_rate/(1+b*torch.exp(-a*data[:,:,2].view(-1)))

#         e_mean = (data[:,:,3].view(-1))/(sigma_combine + 1)

#         softmax_args = torch.stack([beta*e_mean, beta*torch.tensor(1., device=device)])
#         p = torch.softmax(softmax_args, dim = 0)[0]
#         # p = torch.clamp(p, 1e-6, 1 - 1e-6)
#         # p = p.to(dtype=torch.float32)
#         pyro.sample("obs", dist.Bernoulli(probs = p), obs=data[:,:,4].view(-1))

# vectorised
def model(data):
    num_params = 4
    num_agents = data.shape[0]
    num_trials = data.shape[1]
    # define hyper priors over model parameters
    # prior over sigma of a Gaussian is a Gamma distribution
    a = pyro.param('a', torch.ones(num_params, device=device), constraint=dist.constraints.positive)
    lam = pyro.param('lam', torch.ones(num_params, device=device), constraint=dist.constraints.positive)
    tau = pyro.sample('tau', dist.Gamma(a, a/lam).to_event(1)) # mean = a / (a/lam) = lam

    sig = pyro.deterministic('sig', 1/torch.sqrt(tau)) # Gauss sigma

    # each model parameter has a hyperprior defining group level mean
    # in the form of a Normal distribution
    m = pyro.param('m', torch.zeros(num_params, device=device))
    s = pyro.param('s', torch.ones(num_params, device=device), constraint=dist.constraints.positive)
    mu = pyro.sample('mu', dist.Normal(m, s*sig).to_event(1)) # Gauss mu, wieso s*sig?

    # in order to implement groups, where each subject is independent of the others, pyro uses so-called plates.
    # you embed what should be done for each subject into the "with pyro.plate" context
    # the plate vectorizes subjects and adds an additional dimension onto all arrays/tensors
    # i.e. p1 below will have the length num_agents
    with pyro.plate('ag_idx', num_agents):
        # draw parameters from Normal and transform (for numeric trick reasons)
        # base_dist = dist.Normal(0., 1.).expand_by([num_params]).to_event(1)
        base_dist = dist.Normal(torch.zeros(num_params, device=device), torch.ones(num_params, device=device)).to_event(1)
        # Transform via the pointwise affine mapping y = loc + scale*x (-> Neal's funnel)
        transform = dist.transforms.AffineTransform(mu, sig)

        locs = pyro.sample('locs', dist.TransformedDistribution(base_dist, [transform]))


    with pyro.plate('data', num_agents*num_trials):
        sigma_rate = torch.exp(locs[:,0]).unsqueeze(-1).expand(-1, num_trials)
        a = torch.exp(locs[:,1]).unsqueeze(-1).expand(-1, num_trials)
        b = torch.exp(locs[:,2]).unsqueeze(-1).expand(-1, num_trials)
        beta = torch.exp(locs[:,3]).unsqueeze(-1).expand(-1, num_trials)

        sigma_combine = sigma_rate/(1+b*torch.exp(-a*data[:,:,2]))

        e_mean = (data[:,:,3])/(sigma_combine + 1)
        sum = e_mean + torch.tensor(1., device=device)

        softmax_args = torch.stack([beta*e_mean/sum, beta*torch.tensor(1., device=device)/sum])
        p = torch.softmax(softmax_args, dim = 0)[0]

    pyro.sample("obs", dist.Bernoulli(probs = p).to_event(2), obs=data[:,:,4])


def pyro_reset(seed=0):
    pyro.clear_param_store()
    torch.cuda.empty_cache()
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)
    gc.collect()



def convert_units(text):
    original_num = float(text[0][0])
    unit = text[0][1]
    if unit == "us":
        num = original_num/1000
    elif unit == "ms":
        num = original_num
    elif unit == "s":
        num = original_num*1000
    
    return num

cpu = []
gpu = []
total = []

fraction_num = 10
repeat_num = 50


for i in tqdm(range(fraction_num)):
    data = single_data[:20*(i+1)]

    num_iters = 100
    with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(num_iters):
            pyro_reset()
            model(data)


    text = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)

    if device.type == "cuda":
        lines = text.split('\n')[-3:]

        cpu_text = re.findall(r"(\d*\.\d+|\d+)(us|ms|s)", lines[0])
        gpu_text = re.findall(r"(\d*\.\d+|\d+)(us|ms|s)", lines[1])

        cpu_num = convert_units(cpu_text)
        gpu_num = convert_units(gpu_text)

        avg_self_cpu = cpu_num/num_iters
        avg_self_cuda = gpu_num/num_iters
        
        cpu.append(avg_self_cpu)
        gpu.append(avg_self_cuda)
        

    elif device.type == "cpu":
        lines = text.split('\n')[-2:]
        cpu_text = re.findall(r"(\d*\.\d+|\d+)(us|ms|s)", lines[0])
        cpu_num = convert_units(cpu_text)
        avg_self_cpu = cpu_num/num_iters

        cpu.append(avg_self_cpu)
    
    start_time = time.time()
    model(data)
    end_time = time.time()
    
    total_time = end_time - start_time
    total.append(total_time)
    # pyro_reset()


for i in tqdm(range(repeat_num)):
    data = single_data.repeat((i+1, 1, 1))
    
    num_iters = 100
    with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(num_iters):
            pyro_reset()
            model(data)


    text = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)

    if device.type == "cuda":
        lines = text.split('\n')[-3:]

        cpu_text = re.findall(r"(\d*\.\d+|\d+)(us|ms|s)", lines[0])
        gpu_text = re.findall(r"(\d*\.\d+|\d+)(us|ms|s)", lines[1])

        cpu_num = convert_units(cpu_text)
        gpu_num = convert_units(gpu_text)

        avg_self_cpu = cpu_num/num_iters
        avg_self_cuda = gpu_num/num_iters
        
        cpu.append(avg_self_cpu)
        gpu.append(avg_self_cuda)
        

    elif device.type == "cpu":
        lines = text.split('\n')[-2:]
        cpu_text = re.findall(r"(\d*\.\d+|\d+)(us|ms|s)", lines[0])
        cpu_num = convert_units(cpu_text)
        avg_self_cpu = cpu_num/num_iters

        cpu.append(avg_self_cpu)
    
    start_time = time.time()
    model(data)
    end_time = time.time()
    
    total_time = end_time - start_time
    total.append(total_time)
    # pyro_reset()


with open("/home/yaning/Documents/Discounting/vectorisation_discounting/cpu_vectorised.pkl", "wb") as f:
    pickle.dump((cpu, gpu, total), f)
