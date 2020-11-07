# My notes from kaggle tutorial
# 
# XGBoost has a few parameters that can dramatically affect accuracy 
# and training speed. The first parameters you should understand are:

# n_estimators
# n_estimators specifies how many times to go through 
# the modeling cycle described above. 
# It is equal to the number of models that we include in the ensemble.

# 1) Too low a value causes underfitting, 
# which leads to inaccurate predictions on both training data and test data.
# 2) Too high a value causes overfitting, 
# which causes accurate predictions on training data, 
# but inaccurate predictions on test data (which is what we care about).


# early_stopping_rounds
# early_stopping_rounds offers a way to automatically find the ideal value 
# for n_estimators. Early stopping causes the model to stop iterating 
# when the validation score stops improving, even if we aren't at the hard 
# stop for n_estimators. It's smart to set a high value for n_estimators 
# and then use early_stopping_rounds to find the optimal time to stop iterating.

# Since random chance sometimes causes a single round where validation scores 
# don't improve, you need to specify a number for how many rounds of 
# straight deterioration to allow before stopping. Setting early_stopping_rounds=5 
# is a reasonable choice. In this case, we stop after 5 straight rounds of 
# deteriorating validation scores.

# When using early_stopping_rounds, you also need to set aside some data 
# for calculating the validation scores - this is done by setting the 
# eval_set parameter.

# Set up code checking
import os
if not os.path.exists("../input/train.csv"):
    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  
    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 
from learntools.core import binder
binder.bind(globals())
from learntools.ml_intermediate.ex6 import *
print("Setup Complete")

import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice              
X.drop(['SalePrice'], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]

# Select numeric columns
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# One-hot encode the data (to shorten the code, we use pandas)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)

from xgboost import XGBRegressor

# Define the model
my_model_1 = XGBRegressor(random_state=0) # Your code here

# Fit the model
my_model_1.fit(X_train, y_train) # Your code here


from sklearn.metrics import mean_absolute_error

# Get predictions
predictions_1 = my_model_1.predict(X_valid)# Your code here


# Calculate MAE
mae_1 = mean_absolute_error(predictions_1, y_valid) # Your code here

# Uncomment to print MAE
print("Mean Absolute Error:" , mae_1)


# Design a model that outperforms my_model_1
# Define the model
my_model_2 = XGBRegressor(n_estimators=1000, learning_rate=0.01, n_jobs=4) # Your code here

# Fit the model
my_model_2.fit(X_train, y_train, early_stopping_rounds=12, eval_set=[(X_valid, y_valid)], verbose=False) # Your code here

# Get predictions
predictions_2 = my_model_2.predict(X_valid) # Your code here

# Calculate MAE
mae_2 = mean_absolute_error(predictions_2, y_valid) # Your code here

# Uncomment to print MAE
print("Mean Absolute Error:" , mae_2)



# Design a model that performs worse than my_model_2
# Define the model
my_model_3 = XGBRegressor(n_estimators=10, learning_rate=0.1, n_jobs=4)

# Fit the model
my_model_3.fit(X_train, y_train, early_stopping_rounds=2, eval_set=[(X_valid, y_valid)], verbose=False) # Your code here

# Get predictions
predictions_3 = my_model_3.predict(X_valid)

# Calculate MAE
mae_3 = mean_absolute_error(predictions_3, y_valid)

# Uncomment to print MAE
print("Mean Absolute Error:" , mae_3)
