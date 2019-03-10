#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 10:04:59 2019

@author: rebeccafang
"""

"""
This program takes the CSV dataset from the DeepSolar project and cleans it for
use in machine learning models for Rebecca and Kristen's CS229A project.

The dataset is split into 80%/10%/10% for the training set, cross-validation
set, and test set.
"""

import sys
import pandas as pd
import numpy as np


data_file = 'deepsolar_data.csv'

# Import CSV file
data = pd.read_csv(data_file, delimiter = ',', encoding='latin-1') # Read csv file into DataFrame

print(len(data)) # Print number of rows

#nulls = data.isna().sum()
#nulls.to_csv('nulls.csv')
#
#zero_tiles = data[data.tile_count == 0]
#zero_tiles.to_csv('zero_tiles.csv')
#
#zero_systems = data[data.solar_system_count == 0]
#zero_systems.to_csv('zero_systems.csv')


# Remove columns that are irrelevant or that would skew results
data.drop(['tile_count', 'fips', 'county', 'tile_count_residential', 'tile_count_nonresidential', 'state'], axis=1, inplace=True)
data.drop(['total_panel_area', 'total_panel_area_residential', 'total_panel_area_nonresidential','solar_panel_area_divided_by_area','solar_panel_area_per_capita'], axis=1, inplace=True)
data.drop(['solar_system_count','solar_system_count_residential','solar_system_count_nonresidential'], axis=1, inplace=True)

# Drop rows with missing data in at least one column
data.dropna(inplace=True)

# Convert number_of_solar_system_per_household to number of solar systems per 1,000 households
data['number_of_solar_system_per_household'] *= 1000
data = data.rename(columns={'number_of_solar_system_per_household': 'number_solar_system_per_1000_household'})

print(len(data)) # Print number of rows after dropna

data = data.sample(frac=1).reset_index(drop=True) # Shuffle data to later split into training, validation, test sets
fractions = np.array([0.8, 0.1, 0.1]) # Split into sets
train, val, test = np.array_split(data, (fractions[:-1].cumsum() * len(data)).astype(int))
    
print("The dataset has " + str(data.shape[0]) + " rows and " + str(data.shape[1]) + " columns.")
#print(data.head())

train.to_csv('solar_training_set.csv')
val.to_csv('solar_val_set.csv')
test.to_csv('solar_test_set.csv')
