import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import preprocessing 
import sys
from sklearn.preprocessing import StandardScaler
#RandomForest Imports
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
#K-Means Imports
from sklearn import neighbors
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn import metrics

#import data
training_set = pd.read_csv('solar_training_set.csv', delimiter = ',')
training_set.drop(['Unnamed: 0','Unnamed: 0.1'], axis=1, inplace=True)
val_set = pd.read_csv('solar_val_set.csv', delimiter = ',')
val_set.drop(['Unnamed: 0','Unnamed: 0.1'], axis=1, inplace=True)
test_set = pd.read_csv('solar_test_set.csv', delimiter = ',')
test_set.drop(['Unnamed: 0','Unnamed: 0.1'], axis=1, inplace=True)

y_train = training_set[['number_solar_system_per_1000_household']]
training_set.drop('number_solar_system_per_1000_household', axis=1, inplace=True) # Remove y column
X_train = training_set

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


#https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

X_sample_train = X_train.sample(n=None, frac=1/10, replace=False, weights=None, random_state=None, axis=None);
y_sample_train = y_train.sample(n=None, frac=1/10, replace=False, weights=None, random_state=None, axis=None);

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
print(rf_random)
print(rf_random.fit(X_sample_train, np.ravel(y_sample_train, order = 1)))
print(rf_random.best_params_)
# Print the mean absolute error
best_random = rf_random.best_estimator_
y_pred = best_random.predict(X_val) # Predict residential solar system density on validation set
y_val = y_val.values # Convert dataframe to numpy array
y_pred = np.expand_dims(y_pred,1) # Expand dimensions to match y_val  
print('Mean absolute error: %.2f' % mean_absolute_error(y_val, y_pred))
# Print the mean squared error
print('Mean squared error: %.2f' % mean_squared_error(y_val, y_pred))
# Print variance score, where 1 = perfect prediction
print('Variance score: %.2f' % r2_score(y_val, y_pred))

#GRIDSEARCHCV
#Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [10],
    'max_features': ['sqrt'],
    'min_samples_leaf': [4],
    'min_samples_split': [2],
    'n_estimators': [1600]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X_train, np.ravel(y_train, order = 1))
grid_search.best_params_
best_grid = grid_search.best_estimator_

y_pred = best_grid.predict(X_val) # Predict residential solar system density on validation set
y_val = y_val.values # Convert dataframe to numpy array
y_pred = np.expand_dims(y_pred,1) # Expand dimensions to match y_val  
print('Mean absolute error: %.2f' % mean_absolute_error(y_val, y_pred))
# Print the mean squared error
print('Mean squared error: %.2f' % mean_squared_error(y_val, y_pred))
# Print variance score, where 1 = perfect prediction
print('Variance score: %.2f' % r2_score(y_val, y_pred))

#predict on training set:
y_pred_training = best_grid.predict(X_train) # Predict residential solar system density on validation set
y_train = y_train.values # Convert dataframe to numpy array
y_pred_training = np.expand_dims(y_pred_training,1) # Expand dimensions to match y_train  
print('Mean absolute error training: %.2f' % mean_absolute_error(y_train, y_pred_training))
# Print the mean squared error
print('Mean squared error training: %.2f' % mean_squared_error(y_train, y_pred_training))
# Print variance score, where 1 = perfect prediction
print('Variance score training: %.2f' % r2_score(y_train, y_pred_training))

#predict on training set:
y_pred_test = best_grid.predict(X_test) # Predict residential solar system density on validation set
y_test = y_test.values # Convert dataframe to numpy array
y_pred_test = np.expand_dims(y_pred_test,1) # Expand dimensions to match y_train  
print('Mean absolute error testing: %.2f' % mean_absolute_error(y_test, y_pred_test))
# Print the mean squared error
print('Mean squared error testing: %.2f' % mean_squared_error(y_test, y_pred_test))
# Print variance score, where 1 = perfect prediction
print('Variance score testing: %.2f' % r2_score(y_test, y_pred_test))
#
#-----------------------K Means Neighbour ------------
#https://www.analyticsvidhya.com/blog/2018/08/k-nearest-neighbor-introduction-regression-python/
#k means neighbour training and predictions
rmse_val = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train, y_train)  #fit the model
    pred=model.predict(X_val) #make prediction on test set
    error = mean_squared_error(y_val,pred) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)

#plotting the rmse values against k values
curve = pd.DataFrame(rmse_val) #elbow curve 
curve.plot()
#
#implementing GridsearchCV to decide value of k
params = {'n_neighbors':[2,3,4,5,6,7,8,9, 10, 11, 12,13,14,15,16,17,18,19,20]}

knn = neighbors.KNeighborsRegressor()

model = GridSearchCV(knn, params, cv=5)
model.fit(X_train,y_train)
bestModel = model.best_params_
print("best output", bestModel)

modelBest = neighbors.KNeighborsRegressor(n_neighbors = 9)
modelBest.fit(X_train, y_train)  #fit the model
pred=modelBest.predict(X_val) #make prediction
print(pred.shape)
y_val = y_val.values # Convert dataframe to numpy array
print(y_val.shape)
#y_pred = np.expand_dims(y_pred,1) # Expand dimensions to match y_val 
print(y_pred.shape) 
print('Mean absolute error: %.2f' % mean_absolute_error(y_val, pred))
# Print the mean squared error
print('Mean squared error: %.2f' % mean_squared_error(y_val, pred))
# Print variance score, where 1 = perfect prediction
print('Variance score: %.2f' % r2_score(y_val, pred))
y_pred_training = modelBest.predict(X_train) # Predict residential solar system density on validation set
y_train = y_train.values # Convert dataframe to numpy array
#y_pred_training = np.expand_dims(y_pred_training,1) # Expand dimensions to match y_train  
print('Mean absolute error training: %.2f' % mean_absolute_error(y_train, y_pred_training))
# Print the mean squared error
print('Mean squared error training: %.2f' % mean_squared_error(y_train, y_pred_training))
# Print variance score, where 1 = perfect prediction
print('Variance score training: %.2f' % r2_score(y_train, y_pred_training))

#predict on training set:
y_pred_test = modelBest.predict(X_test) # Predict residential solar system density on validation set
y_test = y_test.values # Convert dataframe to numpy array
#y_pred_test = np.expand_dims(y_pred_test,1) # Expand dimensions to match y_train  
print('Mean absolute error testing: %.2f' % mean_absolute_error(y_test, y_pred_test))
# Print the mean squared error
print('Mean squared error testing: %.2f' % mean_squared_error(y_test, y_pred_test))
# Print variance score, where 1 = perfect prediction
print('Variance score testing: %.2f' % r2_score(y_test, y_pred_test))

