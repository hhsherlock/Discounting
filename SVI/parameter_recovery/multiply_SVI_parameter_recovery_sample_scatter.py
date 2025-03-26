#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 2024

@author: Yaning
"""

import multi_agent_simulation as agt
import normal_log_model as model

import os
import numpy as np
import torch
import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import matplotlib.pyplot as plt
from statistics import mean
import csv

# prior mean (mean_u) is between 0 and 10
# prior variance (var_u) is between 1. to 10.
# estimation/likelihood variance is between 1. to 10.

folder_name = 'mean_u_sigma_u_15'
mean_u_list = np.linspace(0., 8., 100)
sigma_u_list = np.linspace(0.01, 8., 100)
sigma_es_list = np.linspace(0.01, 10., 100)


# initialise trial combinations
repetition = 10
real_delays = [1., 3., 7., 13., 24., 32., 45., 58., 75., 122.]
delays =  [i / 10 for i in real_delays]
SS_values = [5.]
# LL values by percentage of SS
LL_values_p = [1.05, 1.055, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, \
               1.85, 1.9, 2.05, 2.25, 2.55, 2.85, 3.05, 3.45, 3.85]

# initialise the environment
environment = agt.Environment(SS_values, LL_values_p, delays, repetition)

# setup the optimizer
# the learning rate was 0.0005
def inference(actions, delays, ss_values, ll_values, real_params):

    # this is for running the notebook in our testing framework
    smoke_test = ('CI' in os.environ)
    # the step was 2000
    n_steps = 2 if smoke_test else 400

    # assert pyro.__version__.startswith('1.8.6')

    # clear the param store in case we're in a REPL
    pyro.clear_param_store()

    # adam_params = {"lr": 0.05, "npart": 15, "betas": (0.90, 0.999)}
    adam_params = {"lr": 0.1}
    optimizer = Adam(adam_params)
    # setup the inference algorithm
    svi = SVI(model.model_normal_log, model.guide_normal_log, optimizer, loss=Trace_ELBO())

    loss = []
    # pbar = tqdm(range(n_steps), position = 0)
    # # do gradient steps
    # for step in pbar:
    #     loss.append(torch.tensor(svi.step(actions, delays, ss_values, ll_values)))
    #     pbar.set_description("Mean ELBO %6.2f" % torch.tensor(loss[-20:]).mean())
    #     if torch.isnan(loss[-1]):
    #         break

    # not showing the progress bar because they will generate -
    # - loads printing in nohup.out
    for step in range(n_steps):
        loss.append(torch.tensor(svi.step(actions, delays, ss_values, ll_values)))
        if torch.isnan(loss[-1]):
            break

    plt.figure()
    plt.plot(loss)
    plt.xlabel("iter step")
    plt.ylabel("ELBO loss")
    plt.title("ELBO minimization during inference")
    plt.savefig('plots/'+ folder_name + '/parameter' + real_params +'.png')

    # grab the posterior parameters with own guide
    mean_u_mean_q = pyro.param("mean_u_mean_q").item()
    mean_u_sigma_q = pyro.param("mean_u_sigma_q").item()
    sigma_u_mean_q = pyro.param("sigma_u_mean_q").item()
    sigma_u_sigma_q = pyro.param("sigma_u_sigma_q").item()
    sigma_es_mean_q = pyro.param("sigma_es_mean_q").item()
    sigma_es_sigma_q = pyro.param("sigma_es_sigma_q").item()

    # # grab posterior from auto guide
    # dict = {}
    # for name, value in pyro.get_param_store().items():
    #     dict[name] = pyro.param(name)

    return [mean_u_mean_q, mean_u_sigma_q, sigma_u_mean_q, \
            sigma_u_sigma_q, sigma_es_mean_q, sigma_es_sigma_q]

sample_repeat = 15

real_mean_u = []
posterior_mean_u = []
real_sigma_u = []
posterior_sigma_u = []
real_sigma_es = []
posterior_sigma_es = []


# this is the main process
for i in range(sample_repeat):
    mean_u = np.random.choice(mean_u_list)
    sigma_u = np.random.choice(sigma_u_list)
    # sigma_es = np.random.choice(sigma_es_list)
    sigma_es = 6
    real_params = str(round(mean_u, 2)) + '_' + str(round(sigma_u)) + '_' + str(round(sigma_es)) 
    # simulation
    actions, delays, ss_values, ll_values = agt.decision(mean_u, sigma_u, sigma_es, environment)
    actions = torch.tensor(actions)
    delays = torch.tensor(delays)
    ss_values = torch.tensor(ss_values)
    ll_values = torch.tensor(ll_values)

    # infer
    mean_u_mean_q, mean_u_sigma_q, sigma_u_mean_q, \
        sigma_u_sigma_q, sigma_es_mean_q, sigma_es_sigma_q =\
              inference(actions, delays, ss_values, ll_values, real_params)
    
    real_mean_u.append(mean_u)
    posterior_mean_u.append(mean_u_mean_q)
    real_sigma_u.append(sigma_u)
    posterior_sigma_u.append(np.exp(sigma_u_mean_q))
    real_sigma_es.append(sigma_es)
    posterior_sigma_es.append(np.exp(sigma_es_mean_q))


# get the exp'ed results, record them in csv file in the folder
file = open('plots/'+ folder_name +'/test_csv.csv', 'w', newline='')
header = ['real_mean_u', 'posterior_mean_u', 'real_sigma_u', 'posterior_sigma_u',
          'real_sigma_es', 'posterior_sigma_es']
writer = csv.writer(file)

writer.writerow(header)
for i in range(len(real_mean_u)):
    writer.writerow([real_mean_u[i], posterior_mean_u[i], real_sigma_u[i],
                     posterior_sigma_u[i], real_sigma_es[i], posterior_sigma_es[i]])
file.close()



# plot the posterior with simulated data
# and save the plots
fig = plt.figure(constrained_layout=False, figsize=(8,8))
gs = fig.add_gridspec(nrows=2, ncols=2, left=0.05, right=0.95, wspace=0.35, hspace=0.35)

ax1 = fig.add_subplot(gs[0,0], xlabel = 'real_mean_u', ylabel = 'infered_mean_u')
ax2 = fig.add_subplot(gs[0,1], xlabel = 'real_sigma_u', ylabel = 'infered_sigma_u')
ax3 = fig.add_subplot(gs[1,0], xlabel = 'real_sigma_es', ylabel = 'infered_sigma_es')



ax1.scatter(real_mean_u, posterior_mean_u)
ax2.scatter(real_sigma_u, posterior_sigma_u)
ax3.scatter(real_sigma_es, posterior_sigma_es)

plt.savefig('plots/' + folder_name + '/' + str(sample_repeat) +'.png')

plt.close()





