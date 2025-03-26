#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 8 2024

@author: Yaning
"""
import torch
from pyro.distributions import Bernoulli
import numpy as np

class Agent:
    def __init__(self, mean_u, sigma_u, sigma_es, beta):
        # mean_u is the prior mean 
        # sigma_u is the prior variance
        # sigma_es is the estimation/likelihood variance
        self.mean_u = mean_u
        self.sigma_u = sigma_u
        self.sigma_es = sigma_es
        self.beta = beta

    def estimation(self, curr_LL, curr_delay, curr_SS):
        # the estimate value is the multiplition of -
        # - two normal distributions
        self.estimate = (self.mean_u*curr_delay*self.sigma_es**2 + 
            curr_LL*self.sigma_u**2)/(curr_delay*self.sigma_es**2 + self.sigma_u**2)
        self.estimate_sgima = ((self.sigma_es**2*curr_delay*self.sigma_u**2)/
                               (self.sigma_es**2*curr_delay + self.sigma_u**2))**0.5
        self.curr_SS = curr_SS


    def generate_action(self):
        # use softmax function with estimate and SS
        p = torch.nn.functional.softmax(torch.tensor(
            [self.beta*self.estimate, self.beta*self.curr_SS]), dim = 0)[0]
        # LL_distri = Normal(loc = self.estimate, scale = self.estimate_var)
        # p = 1 - LL_distri.cdf(torch.tensor(self.curr_SS))
        LL_bino = Bernoulli(probs = p)
        return LL_bino.sample().item()

class Environment:
    def __init__(self, ss_values, ll_values_p, delays, repetition):
        self.ss_values = ss_values
        self.ll_values_p = ll_values_p
        self.delays = delays
        self.repetition = repetition


# make the decision under input personalilties as in 
# as in prior and estimation distribution characteristics
def decision(mean_u, sigma_u, sigma_es, beta, environment):
    # create some empty arrays
    # get rid of the same ss, ll and delay repitition only percentage
    a_percentage = []
    a_ss_values = []
    a_ll_values = []
    a_delays = []

    # future tensor for inference specifically
    inference_ss_values = []
    inference_ll_values = []
    inference_delays = []
    inference_actions = []
    agent = Agent(mean_u, sigma_u, sigma_es, beta)
    for i in environment.ss_values:
        for j in environment.ll_values_p:
            for k in environment.delays:
                one_combination_actions = []
                temp_ll_value = i*j
                # all the values started with a are not returned because
                # otherwise the return array has different amount of elements
                # when converting to numpy array, throws error hetero 
                a_ll_values.append(temp_ll_value)
                a_delays.append(k)
                a_ss_values.append(i)
                for r in range(environment.repetition):
                    agent.estimation(temp_ll_value, k, i)
                    action = agent.generate_action()
                    one_combination_actions.append(action)
                    inference_ss_values.append(i)
                    inference_ll_values.append(temp_ll_value)
                    inference_delays.append(k)
                    inference_actions.append(action)
                a_percentage.append(len(list(filter(lambda x: (x == 1), 
                                                    one_combination_actions))) / 
                                                    len(one_combination_actions))
    return [inference_actions, inference_delays, \
            inference_ss_values, inference_ll_values]

# construct all the personalities (mean_u, sigma_u and sigma_es)
def run(mean_u_list, sigma_u_list, sigma_es_list, beta_list, environment):
    parameters_list = []
    inference_list = []
    for i in mean_u_list:
        j_p_list = []
        j_i_list = []
        for j in sigma_u_list:
            k_p_list = []
            k_i_list = []
            for k in sigma_es_list:
                l_p_list = []
                l_i_list = []
                for l in beta_list:
                    l_p_list.append([i, j, k, l])
                    l_i_list.append([decision(i, j, k, l, environment)])
                k_p_list.append(l_p_list)
                k_i_list.append(l_i_list)
            j_p_list.append(k_p_list)
            j_i_list.append(k_i_list)
        parameters_list.append(j_p_list)
        inference_list.append(j_i_list)

    parameters_list = np.array(parameters_list)
    inference_list = np.array(inference_list)
    # print(inference_list)
    # print(parameters_list)
    return parameters_list, inference_list