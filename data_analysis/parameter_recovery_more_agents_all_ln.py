#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 2025

@author: Yaning
"""

import os
import numpy as np
import torch
import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
import pyro.distributions as dist
import more_agents_full_ln as agent
import pickle

#-------------------------initialisation---------------------------------------

# initialise environment values
repetition = 10
real_delays = [  1.,   2.,   3.,   4.,   6.,   7.,  13.,  14.,  23.,  24.,  29.,
        32.,  45.,  50.,  58.,  62.,  73.,  75., 118., 122.]
delays =  [i / 10 for i in real_delays]
SS_values = [20.]

# LL values by percentage of SS
LL_values_p = [1.02  , 1.025 , 1.0255, 1.05  , 1.055 , 1.08  , 1.085 , 1.15  ,
       1.2   , 1.25  , 1.33  , 1.35  , 1.45  , 1.47  , 1.5   , 1.55  ,
       1.65  , 1.7   , 1.83  , 1.85  , 1.9   , 2.05  , 2.07  , 2.25  ,
       2.3   , 2.5   , 2.55  , 2.8   , 2.85  , 3.05  , 3.1   , 3.45  ,
       3.5   , 3.8   , 3.85  ]

LL_values = []
for i in LL_values_p:
    LL_values.append(i*20.)

environment_list = [repetition, delays, SS_values, LL_values]

# get all the combinations of the arrays
combinations = list(itertools.product(environment_list[1], environment_list[2], environment_list[3]))
# every combination appears ten times
multiplied_array = [x for x in combinations for _ in range(environment_list[0])]
multiplied_array = np.array(multiplied_array)
trial_num = multiplied_array.shape[0]
whole = multiplied_array


#-------------------------define functions-------------------------------------
def sample_param():
    log_mean_u_a = np.random.choice(np.linspace(-5., 0., 1000))
    mean_u_b = np.random.choice(np.linspace(-10., 0., 1000))
    log_sigma_rate_a = np.random.choice(np.linspace(-5., 5., 1000))
    sigma_rate_b = np.random.choice(np.linspace(-1., 3., 1000))
    # log_beta = np.random.choice(np.linspace(-5, 0., 1000))
    log_beta = 0

    params = [log_mean_u_a, mean_u_b,
              log_sigma_rate_a, sigma_rate_b,
              log_beta]
    
    return params

# get the actions and add them to the data
def simulation(params):

    mean_u_a = np.exp(params[0])
    mean_u_a = np.repeat([mean_u_a], trial_num)

    mean_u_b = params[1]
    mean_u_b = np.repeat([mean_u_b], trial_num)

    sigma_rate_a = -np.exp(params[2])
    sigma_rate_a = np.repeat([sigma_rate_a], trial_num)

    sigma_rate_b = params[3]
    sigma_rate_b = np.repeat([sigma_rate_b], trial_num)

    beta = np.repeat([np.exp(params[4])], trial_num)

    mean_u = mean_u_a*np.log(whole[:,0]) + mean_u_b
    sigma_rate = sigma_rate_a*np.log(whole[:,0]) + sigma_rate_b
 

    # major calculation 
    inferred_estimation = (mean_u + sigma_rate**2*whole[:,2])/(
        sigma_rate**2 + 1)
    
    # inferred_sigma = ((sigma_u**2*sigma_es**2*delay**2)/(sigma_u**2 + sigma_es**2*delay**2))**0.5





    # change everything to tensor
    # whole = torch.tensor(whole)
    inferred_estimation = torch.tensor(inferred_estimation)
    # inferred_sigma = torch.tensor(inferred_sigma)
    beta = torch.tensor(beta)

    # e_dist  = dist.Normal(inferred_estimation, inferred_sigma)
    # pos = 1 - e_dist.cdf(torch.tensor(20.))
    # softmax_args = torch.stack([beta*pos, beta*(1-pos)])
    # !!!!!!!!!!!!!!!!!!!!no beta here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    softmax_args = torch.stack([beta*inferred_estimation, beta*torch.tensor(20.)])
    p = torch.softmax(softmax_args, dim = 0)[0]

    inferred_response_distr = dist.Bernoulli(probs=p)

    # get one possible predict actions
    inferred_response = inferred_response_distr.sample()

    return inferred_response

# if the percentage is lower than 0.1 or bigger than 0.9 then not using
# if the percentage fits the requirement then is good = 1
def check_percentage(params):
    good = 0
    inferred_response = simulation(params)
    choose_LL_percentage = len(list(filter(lambda x: (x == 1), inferred_response))
                            ) / len(inferred_response)
    if choose_LL_percentage > 0.10 and choose_LL_percentage < 0.90:
        good = 1
    return good, inferred_response, choose_LL_percentage


#--------------------------sampling--------------------------------------
sample_num = 1000
agent_num = 100

# get an array of good parameters
good_params = []

while len(good_params) < sample_num:
    params = sample_param()
    if check_percentage(params)[0] == 1:
        good_params.append(params)
good_params = np.array(good_params)

# get the actions for each agent with the good_params
# each agent sample from the good_params array
data = []
param = []
for i in range(agent_num):
    one_agent_param = good_params[np.random.choice(good_params.shape[0])]
    param.append(one_agent_param)
    one_agent_data = []
    while len(one_agent_data) == 0:
        good, inferred_response, choose_percentage = check_percentage(one_agent_param)
        if good == 1:
            one_agent_data = np.hstack((whole, inferred_response.reshape(-1,1)))
        print(choose_percentage)
    data.append(one_agent_data)


data = np.array(data)
param = np.array(param)

#-------------------------------inference-----------------------------------
# doing inference

real_data = data
real_data = torch.tensor(real_data).to('cuda')
# this is for running the notebook in our testing framework
smoke_test = ('CI' in os.environ)
# the step was 2000
n_steps = 2 if smoke_test else 2000

# assert pyro.__version__.startswith('1.8.6')

# clear the param store in case we're in a REPL
pyro.clear_param_store()# setup the optimizer
# the learning rate was 0.0005 , "betas": (0.90, 0.999)
# tried "n_par":15 in adam params but it does not have this argument
adam_params = {"lr": 0.01}
optimizer = Adam(adam_params)
# setup the inference algorithm
svi = SVI(agent.model, agent.guide, optimizer, loss=Trace_ELBO())
# svi = SVI(model_gamma, guide_gamma, optimizer, loss=Trace_ELBO())

loss = []
pbar = tqdm(range(n_steps), position = 0)
# do gradient steps
for step in pbar:
    loss.append(torch.tensor(svi.step(real_data)))
    pbar.set_description("Mean ELBO %6.2f" % torch.tensor(loss[-20:]).mean())
    # for name, value in pyro.get_param_store().items():
    #     print(name, pyro.param(name))
    if torch.isnan(loss[-1]):
        break

plt.figure()
plt.plot(loss)
plt.xlabel("iter step")
plt.ylabel("ELBO loss")
plt.title("ELBO minimization during inference")
plt.savefig('results/graphs/param_recovery_more_agents/param_full_ln_nobeta.png')



pos_dict = {}
for name, value in pyro.get_param_store().items():
    pos_dict[name] = value

# change the dictionary to numpy instead of tensor
# because somehow the tensor cannot be save with pickle
numpy_dict = {key: value.cpu().detach().numpy() for key, value in pos_dict.items()}

both_dict = {}
both_dict['real_param'] = param
both_dict['inferred_param'] = numpy_dict


with open('results/param_recover_full_ln_nobeta.pkl', 'wb') as f:
    pickle.dump(both_dict, f)