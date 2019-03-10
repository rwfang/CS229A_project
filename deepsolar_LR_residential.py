#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 12:42:31 2019

@author: rebeccafang
"""

""" This program applies a linear regression model to the cleaned DeepSolar
dataset. This model predicts residential solar system count per household.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import preprocessing 
import sys

training_set = pd.read_csv('solar_training_set.csv', delimiter = ',')
training_set.drop(['Unnamed: 0','Unnamed: 0.1'], axis=1, inplace=True)
val_set = pd.read_csv('solar_val_set.csv', delimiter = ',')
val_set.drop(['Unnamed: 0','Unnamed: 0.1'], axis=1, inplace=True)
test_set = pd.read_csv('solar_test_set.csv', delimiter = ',')
test_set.drop(['Unnamed: 0','Unnamed: 0.1'], axis=1, inplace=True)

y_train = training_set[['number_solar_system_per_1000_household']]
training_set.drop('number_solar_system_per_1000_household', axis=1, inplace=True) # Remove y column
X_train = training_set

print('y_train', y_train[:10])
print('mean',np.mean(y_train))

y_val = val_set[['number_solar_system_per_1000_household']]
val_set.drop('number_solar_system_per_1000_household', axis=1, inplace=True) # Remove y column
X_val = val_set

y_test = test_set[['number_solar_system_per_1000_household']]
test_set.drop('number_solar_system_per_1000_household', axis=1, inplace=True) # Remove y column
X_test = test_set

# Data preprocessing
le = preprocessing.LabelEncoder()
for column_name in X_train.columns:
    if X_train[column_name].dtype == object:
        X_train[column_name] = le.fit_transform(X_train[column_name])
    else:
        pass
    
for column_name in X_val.columns:
    if X_val[column_name].dtype == object:
        X_val[column_name] = le.fit_transform(X_val[column_name])
    else:
        pass

for column_name in X_test.columns:
    if X_test[column_name].dtype == object:
        X_test[column_name] = le.fit_transform(X_test[column_name])
    else:
       pass

# Loop through alpha values: 1e-5, 1e03, 1e-1, 1e1, 1e2, find alpha that gives lowest MAE
alpha_lst = [0.00001, 0.001, 0.1, 10]
mae_lst = [] # List of mean absolute errors for the alphas

for a in alpha_lst:
    model = linear_model.Lasso(alpha = a)
    model.fit(X_train, y_train)
    y_pred_temp = model.predict(X_val) # Predict residential solar system density on validation set
    y_val_temp = y_val.values # Convert dataframe to numpy array
    y_pred_temp = np.expand_dims(y_pred_temp,1) # Expand dimensions to match y_val
    mae = mean_absolute_error(y_val_temp, y_pred_temp)
    mae_lst.append(mae)
    
# Find alpha that gives lowest mae
alph = alpha_lst[mae_lst.index(min(mae_lst))]

# Create and train linear regression model with L1 regularization on training set
model = linear_model.Lasso(alpha = alph)
model.fit(X_train, y_train)
y_pred = model.predict(X_val) # Predict residential solar system density on validation set
y_val = y_val.values # Convert dataframe to numpy array
y_pred = np.expand_dims(y_pred,1) # Expand dimensions to match y_val  

headers = np.array(X_train.columns.values) # List of column header names
coefs = model.coef_
num_nonzero = np.count_nonzero(coefs) # Number of nonzero coefficients

# Remove features whose coefficients = 0



# Print coefficients
print('Coefficients: \n', model.coef_)
# Print the mean absolute error
print('Mean absolute error: %.2f' % mean_absolute_error(y_val, y_pred))
# Print the mean squared error
print('Mean squared error: %.2f' % mean_squared_error(y_val, y_pred))
# Print variance score, where 1 = perfect prediction
print('Variance score: %.2f' % r2_score(y_val, y_pred))


loss1 = np.sum(np.absolute(y_pred - y_val))
loss2 = mae * len(y_val)
assert loss1 - loss2 <= 0.0001






## Forward step regression
#(F, pvals) = f_regression(X_train, y_train.values.ravel(), center=True)
#header_lst = list(X_train.columns.values) # List of column header names
#feature_rank = [] # List of lists [header, F value, p value]
#for i in range(0,len(header_lst)):
#    add_lst = [header_lst[i], F[i], pvals[i]]
#    feature_rank.append(add_lst)
#
## Remove "Unnamed" columns created during dataframe creation
#del feature_rank[0]
#del feature_rank[0]
#    
## Sort feature_rank list by largest F score
#feature_rank.sort(key=lambda x:x[1])
