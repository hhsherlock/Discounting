#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 2024

@author: Yaning
"""

import csv 
import numpy as np

# change all the string elements to integers
str_to_float = lambda x: [float(i) for i in x]

# file = open('Documents/pyro_models/files_to_run/intertemporal_choice_dataset_all_trials.csv')
file = open('intertemporal_choice_dataset_all_trials.csv')

csvreader = csv.reader(file)

header = []
header = next(csvreader)
print(header)

# [[context1], [[context2]]
data = [[], []]
n_trial = 170

for row in csvreader:
    if row[11] == '0':
        individual = []
        individual.append(str_to_float(row[1:7]) + str_to_float(row[10:]))
        for i in range(n_trial-1):
            next_row = next(csvreader)
            kurz_next_row = str_to_float(next_row[1:7]) + str_to_float(next_row[10:])
            individual.append(kurz_next_row)
        data[0].append(individual)
    elif row[11] == '1':
        individual = []
        individual.append(str_to_float(row[1:7]) + str_to_float(row[10:]))
        for i in range(n_trial-1):
            next_row = next(csvreader)
            kurz_next_row = str_to_float(next_row[1:7]) + str_to_float(next_row[10:])
            individual.append(kurz_next_row)
        data[1].append(individual)

data = np.array(data)

# divide all delays with 10 (because it is too big like 100 something)
data[:,:,:,2] = np.array([i / 10 for i in data[:,:,:,2]])

# multiply val_prc and val_basic (because val_prc is percentage compared to ss)
data[:,:,:,3] = np.array(data[:,:,:,1] * data[:,:,:,3])

file.close()
