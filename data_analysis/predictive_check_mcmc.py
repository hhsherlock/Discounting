#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 2024

@author: Yaning
"""


import math
import os
# from turtle import position
import torch
import torch.distributions.constraints as constraints
import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.distributions import Beta, Binomial, HalfCauchy, Normal, Pareto, Uniform, Bernoulli, Categorical, Gamma
from tqdm import tqdm
import matplotlib.pyplot as plt
from statistics import mean 
# import pandas as pd
import numpy as np
import scipy.stats as stats 
import data_analysis_without_version as analysis
import os
import sys

# first summary then samples
if len(sys.argv) > 1:
    pth_file_0 = sys.argv[1]
    pth_file_1 = sys.argv[2]
else:
    print("no string provided")



data = analysis.data

real_percentages = []

for i in data:
    context_percentages = []
    for j in i:
        context_percentages.append(len(list(filter(lambda x: (x == 1), j[:,4]))
                ) / len(j))
    real_percentages.append(context_percentages)

real_percentages = np.array(real_percentages)

line_data = []
for i,j in zip(real_percentages[0], real_percentages[1]):
    line_data.append([i,j])

# load inferred parameters
# mcmc samples and summary torch load
results_path = os.path.abspath('..') + '\\results\\MCMC\\beta\\beta_4000+2000\\'
summary_dict = torch.load(results_path + pth_file_0)
samples_dict = torch.load(results_path + pth_file_1)

summary_dict.keys()

# transfer the inferred parameters to cpu
pos_mean_u = np.array((summary_dict["mean_u"]["mean"]).to("cpu"))
pos_sigma_u = np.array(np.exp((summary_dict["log_sigma_u"]["mean"]).to("cpu")))
pos_sigma_es = np.array(np.exp((summary_dict["log_sigma_es"]["mean"]).to("cpu")))
pos_beta = np.array((summary_dict["beta"]["mean"]).to("cpu"))

# print(pos_sigma_es)
# print(pos_mean_u)

# shape the real data without separating the context
# because the inferred parameters do not separate those two
real_data = data.reshape(60,170,8)

# replicate the inferred parameters to 170 (trial amount)
mean_u = np.tile(pos_mean_u[:, np.newaxis], (1, 170))
sigma_u = np.tile(pos_sigma_u[:, np.newaxis], (1, 170))
sigma_es = np.tile(pos_sigma_es[:, np.newaxis], (1, 170))
beta = np.tile(pos_beta[:, np.newaxis], (1, 170))

# use real data condition and the inferred parameters to get estimation values
inferred_estimation = (mean_u*real_data[:,:,2]*sigma_es**2 + 
 real_data[:,:,3]*sigma_u**2)/(real_data[:,:,2]*sigma_es**2 + sigma_u**2)

# change everything to tensor
real_data = torch.tensor(real_data)
inferred_estimation = torch.tensor(inferred_estimation)
beta = torch.tensor(beta)

# create softmax and the bernoulli distribution
softmax_args = torch.stack([beta*inferred_estimation, beta*real_data[:,:,1]])
p = torch.softmax(softmax_args, dim = 0)[0]
inferred_response_distr = Bernoulli(probs=p)

# get one possible predict actions
inferred_response = inferred_response_distr.sample()

# reshape to with contexts for plotting
reshaped_inferred_response = inferred_response.reshape(2,30,170)

# calculate the predicted LL choosing percentage
inferred_percentages = []
for i in reshaped_inferred_response:
    temp = []
    for j in i:
        temp.append(len(list(filter(lambda x: (x == 1), j))
                ) / len(j))
    inferred_percentages.append(temp)

inferred_percentages = np.array(inferred_percentages)

# separate to pairs so can make line plot
infer_line = []
for i,j in zip(inferred_percentages[0], inferred_percentages[1]):
    infer_line.append([i,j])

# plot real percentage and infered parameters together
fig, ax = plt.subplots(figsize=(15,8))

# Twin the x-axis twice to make independent y-axes.
axes = [ax, ax.twinx()]

# Make some space on the right side for the extra y-axis.
fig.subplots_adjust(right=0.75)

# lighter dots are cafe and darker dots are gamble
# plot the real percentage (blue)
for i in range(len(line_data)):
    x = [i+1, i+1]
    y = [line_data[i][0], line_data[i][1]]
    axes[0].plot(x, y, color="#67cefd", ls = '-')
    axes[0].plot(i+1, y[0], color ="#9edaf5", marker = "o")
    axes[0].plot(i+1, y[1], color = "#0f88fa", marker = "o")
axes[0].set_ylabel("real LL percentage", color = "#0f88fa")
axes[0].tick_params(axis='y', colors = "#0f88fa")
axes[0].set_xlabel("participant id")

# inferred percentages (purple)
for i in range(len(infer_line)):
    x = [i+1, i+1]
    y_inferred = [infer_line[i][0], infer_line[i][1]]
    axes[1].plot(x, y_inferred, color="#b968fa", ls = "--")
    axes[1].plot(i+1, y_inferred[0], color = "#d2a2f8", marker = "o")
    axes[1].plot(i+1, y_inferred[1], color = "#9715fc", marker = "o")
axes[1].set_ylabel("inferred", color = "#9715fc")
axes[1].tick_params(axis='y', colors = "#9715fc")



plt.show()

# create only one dimension array
# get how many are correctly predicted
real_response = real_data[:,:,4].view(-1)
num = 0
for i in range(len(real_response)):
    if inferred_response.view(-1)[i] == real_response[i]:
        num = num + 1

print(num/len(real_response))

print(np.corrcoef(pos_sigma_u, pos_sigma_es))