#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 2023

@author: Yaning
Suggested by Sascha
"""

import torch
import pyro
from pyro.distributions import Normal, Binomial

class Agent:
    def __init__(self, ds_rate, variance):
        # free parameters (the mean and variant of the estimate value)
        self.ds_rate = ds_rate
        self.variance = variance


    def estimation(self, curr_LL, curr_delay, curr_SS):
        # the estimate value is assumed to be a normal distribution
        # the estimate value mean is assumed to LL/1+kt
        # the estimate value variant is set to be 4
        self.LL_es_mu = curr_LL/(1+self.ds_rate*curr_delay)
        self.LL_es_sigma = self.variance
        # the fixed early and small value
        self.curr_SS = curr_SS


    def generate_action(self):
        # 1 means choosing large later
        # pyro distribution is used (i guess it is the same as torch's distribution)
        # somehow using binomial or bernoulli is a bit slower to generate the data
        # maybe because there are two distributions? 
        LL_distri = Normal(loc = torch.tensor(self.LL_es_mu), scale =torch.tensor(self.LL_es_sigma))
        chose_LL_pobs = 1 - LL_distri.cdf(torch.tensor(self.curr_SS))
        LL_bino = Binomial(probs = chose_LL_pobs)
        return LL_bino.sample().item()
