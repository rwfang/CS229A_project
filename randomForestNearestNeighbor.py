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

#https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
#train randomForest
regressor = RandomForestRegressor(n_estimators=20, random_state=0)  
regressor.fit(X_train, np.ravel(y_train))  
y_pred = regressor.predict(X_test)  

#evaluate randomForest algorithm
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#-----------------------K Means Neighbour ------------
#https://www.analyticsvidhya.com/blog/2018/08/k-nearest-neighbor-introduction-regression-python/
#k means neighbour training and predictions
rmse_val = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train, y_train)  #fit the model
    pred=model.predict(X_test) #make prediction on test set
    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)

#plot the error
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  
 markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')

#plotting the rmse values against k values
curve = pd.DataFrame(rmse_val) #elbow curve 
curve.plot()


#implementing GridsearchCV to decide value of k
params = {'n_neighbors':[2,3,4,5,6,7,8,9]}

knn = neighbors.KNeighborsRegressor()

model = GridSearchCV(knn, params, cv=5)
model.fit(X_train,y_train)
model.best_params_
