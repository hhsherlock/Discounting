#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 2023

@author: Yaning
Suggested by Sascha
"""

import torch
import paper2020_hyperbolic_MCMC_agent as agt
import pandas as pd

LL_values = []
combinations = []
# future tensor for inference specifically
inference_actions = []
inference_ss_values = []
inference_ll_values = []
inference_delays = []

actions = []
choose_percentage = []

# initialise trial combinations
repetition = 10
delays = [1, 3, 7, 13, 24, 32, 45, 58, 75, 122]
SS_values = [5]
# values by percentage
LL_values_p = [1.05, 1.055, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.85, 1.9, 2.05, 2.25, 2.55, 2.85, 3.05, 3.45, 3.85]


# get the LL_values
for S in SS_values:
    for L in LL_values_p:
        LL_values.append(S*L)

# combinations has three parameters: SS_value, LL_value, delay
for S in SS_values:
    for delay in delays:
        for L in LL_values:
            combinations.append([L, delay, S])

# discounting rate is 1/16, the uncertainty which is the variance is 4 for now
agent = agt.Agent(1/16, 4, 0)

for trial in combinations:
    one_combination_action = []
    for repeat in range(repetition):
        # parameters estimation needs: curr_LL, curr_delay, curr_SS
        agent.estimation(trial[0], trial[1], trial[2])
        generated_action = agent.generate_action()

        # arrays that are used for inference
        inference_ll_values.append(trial[0])
        inference_delays.append(trial[1])
        inference_ss_values.append(trial[2])
        inference_actions.append(generated_action)

        one_combination_action.append(generated_action)

    # 1 means choose large later
    choose_percentage.append((len(list(filter(lambda x: (x == 1), one_combination_action))) / len(one_combination_action)) * 100)
    actions.append(one_combination_action)


 
d = {'combinations':combinations, 'actions': actions, 'choose_percentage': choose_percentage}
df = pd.DataFrame(data=d)
print(df)
print(inference_actions)
