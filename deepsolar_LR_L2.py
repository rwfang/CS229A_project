#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 12:42:31 2019

@author: rebeccafang
"""

""" This program applies a linear regression model with L2 regularization to
the cleaned DeepSolar dataset. This model predicts residential solar system
count per household.
"""

import matplotlib.pyplot as plt
import math
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

#print('y_train', y_train[:10])
#print('mean',np.mean(y_train))

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
alpha_lst = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
mae_lst = [] # List of mean absolute errors for the alphas
coefs_lst = [] # List of coefficients arrays for alpha values
loss_lst = [] # List of loss values for alpha values

for a in alpha_lst:
    mod = linear_model.Ridge(alpha = a)
    mod.fit(X_train, y_train)
    y_pred_temp = mod.predict(X_val) # Predict residential solar system density on validation set
    y_val_temp = y_val.values # Convert dataframe to numpy array
    #y_pred_temp = np.expand_dims(y_pred_temp,1) # Expand dimensions to match y_val
    mae_temp = mean_absolute_error(y_val_temp, y_pred_temp)
    mae_lst.append(mae_temp)
    loss_temp = mae_temp * len(y_val_temp)
    loss_lst.append(loss_temp)
    coefs_lst.append(mod.coef_)
    
# Find alpha that gives lowest mae
alph = alpha_lst[mae_lst.index(min(mae_lst))]

# Create and train linear regression model with L1 regularization on training set
model = linear_model.Ridge(alpha = alph)
model.fit(X_train, y_train)
y_pred = model.predict(X_val) # Predict residential solar system density on validation set
y_val = y_val.values # Convert dataframe to numpy array
#y_pred = np.expand_dims(y_pred,1) # Expand dimensions to match y_val  

X_train_lst = [X_train[0:2000],X_train[0:4000],X_train[0:6000],X_train[0:8000],X_train[0:10000],X_train[0:12000],X_train[0:14000],X_train[0:16000],X_train[0:18000],X_train[0:20000],X_train[0:22000],X_train[0:24000],X_train[0:26000],X_train[0:28000],X_train[0:30000],X_train[0:32000],X_train[0:34000],X_train[0:36000], X_train[0:38000],X_train[0:40000],X_train[0:42000], X_train]
y_train_lst = [y_train[0:2000],y_train[0:4000],y_train[0:6000],y_train[0:8000],y_train[0:10000],y_train[0:12000],y_train[0:14000],y_train[0:16000],y_train[0:18000],y_train[0:20000],y_train[0:22000],y_train[0:24000],y_train[0:26000],y_train[0:28000],y_train[0:30000],y_train[0:32000],y_train[0:34000],y_train[0:36000], y_train[0:38000],y_train[0:40000],y_train[0:42000], y_train]

J_train_lst = []
J_val_lst = []

# Loop over various training set sizes
for i in range(0,len(X_train_lst)):
    X_learn = X_train_lst[i]
    y_learn = y_train_lst[i]
    mod_learn = linear_model.Ridge(alpha = alph)
    mod_learn.fit(X_learn, y_learn)
    y_pred_val = mod_learn.predict(X_val)
    #y_pred_val = np.expand_dims(y_pred_val,1)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    mse_val = mean_squared_error(y_val, y_pred_val)
    J_val = mae_val * len(y_val) # Validation error
    J_val_lst.append(mae_val)
    y_pred_train = mod_learn.predict(X_learn)
    #y_pred_train = np.expand_dims(y_pred_train,1)
    mae_train = mean_absolute_error(y_learn, y_pred_train)
    mse_train = mean_squared_error(y_learn, y_pred_train)
    J_train = mae_train * len(y_learn) # Training error
    J_train_lst.append(mae_train)


## Identify features with coefficients equal to zero
#headers = np.array(X_train.columns.values) # List of column header names
#coefs = model.coef_
#zeros = np.where(coefs==0)[0]
#num_nonzero = np.count_nonzero(coefs) # Number of nonzero coefficients
#zero_lst = []
#for ind in zeros:
#    zero_lst.append(headers[ind])
#
## Remove features whose coefficients = 0
#for feature in zero_lst:
#    X_train.drop(feature, axis=1, inplace=True)
#    X_val.drop(feature, axis=1, inplace=True)
#    X_test.drop(feature, axis=1, inplace=True)
#
## Run model again with smaller set of 70 features
#model = linear_model.Ridge(alpha = alph, max_iter = 1e5)
#model.fit(X_train, y_train)
#y_pred_1 = model.predict(X_val) # Predict residential solar system density on validation set
#y_pred_1 = np.expand_dims(y_pred_1,1) # Expand dimensions to match y_val
#mae = mean_absolute_error(y_val, y_pred_1)
#
## Rank 70 features based on coefficients


# Print coefficients
print('Coefficients: \n', model.coef_)
# Print the mean absolute error
print('Mean absolute error: %.2f' % mean_absolute_error(y_val, y_pred))
# Print the mean squared error
print('Mean squared error: %.2f' % mean_squared_error(y_val, y_pred))
# Print the root mean squared error
print('Root mean squared error: %.2f' % math.sqrt(mean_squared_error(y_val, y_pred)))
# Print variance score, where 1 = perfect prediction
print('Variance score: %.2f' % r2_score(y_val, y_pred))


#loss1 = np.sum(np.absolute(y_pred - y_val))
#loss2 = mae * len(y_val)
#assert loss1 - loss2 <= 0.0001

# Plot coefficients vs. alphas
plt.plot(alpha_lst, np.squeeze(coefs_lst))
plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('Coefficients')
plt.title('Ridge coefficients as a function of the learning rate')
#plt.savefig('L2_coefs_alpha.eps', format='eps', dpi=1000)
plt.show()

# Plot loss vs. alphas
plt.plot(alpha_lst, loss_lst)
plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('Loss')
plt.title('Loss as a function of the learning rate')
#plt.savefig('L2_Loss_alpha.eps', format='eps', dpi=1000)
plt.show()

# Plot learning curves
X_train_num = []
for i in range(0,len(X_train_lst)):
    X_train_num.append(len(X_train_lst[i]))
plt.plot(X_train_num, J_train_lst)
plt.plot(X_train_num, J_val_lst)
plt.legend(["Training Error", "Validation Error"], loc='best')
plt.xlabel('Training examples')
plt.ylabel('Mean Absolute Error')
plt.title('Learning Curves for L2 Regularization')
#plt.savefig('L2_reg_learning_curve.eps', format='eps', dpi=1000)
plt.show()



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
