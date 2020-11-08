# -*- coding: utf-8 -*-
"""Keras_Scikit-Learn_Wrappers.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fZr4P9hM_T38sNzPXZ3oAqtVuAndtr_0

# Keras Scikit-Learn Wrappers in Hyperparameters Fune-Tuning
There are many techniques to explore a search space and it is adviced to use a Python library for it. You can see below the state-of-the-art libraries that you can use.



1.   Hyperas (https://github.com/maxpumperla/hyperas) - An important library for optimizing hyperparameters for Keras models
2.   Scikit-Optimize (https://scikit-optimize.github.io) - A probability based library. The BayesSearchCV class does bayesian optimization and it has an interface similar to GridSearchCV
3.   Sklearn-Deap (https://github.com/rsteca/sklearn-deap) - A evolutionary algorithms library with a GridSearchCV interface.


On the other hand, Scikit-Learn provides the GridSearchCV and RandomSearchCV that can be used for the same process. So, we have to convert the Keras model to a Scikit-Learn object. Keras wrappers allow the developer to wrap a Keras model in objects that mimic a regular Scikit-Learn regressor.

By default the keras.wrappers.scikit_learn.KerasRegressor() method requires as an argument a function, which creates a Keras Model. Hence, the first step is to create this function.
"""

def model_fun(hidden=3, neurons=30, learning_rate=0.003, input_shape=[8], activation="relu"):
  model = keras.models.Sequential()
  model.add(keras.layers.InputLayer(input_shape=input_shape))
  for layer in range(hidden):
    model.add(keras.layers.Dense(neurons, activation=activation))
  model.add(keras.layers.Dense(1))
  optimizer = keras.optimizers.SGD(lr=learning_rate)
  model.compile(loss="mse", optimizer=optimizer)
  return model

# Import our Set up
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras

from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV

# Data preprocessing
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Pretend these are new instances
X_new = X_test[:3]

"""The KerasRegressor object is created when we pass the model_fun() method. Now, it performs like a classic Scikit-Learn object (ie regressor) and we can train, evaluate and make predictions."""

keras_reg = keras.wrappers.scikit_learn.KerasRegressor(model_fun)
keras_reg.fit(X_train, y_train, epochs=50, validation_data=(X_valid, y_valid), callbacks=[keras.callbacks.EarlyStopping(patience=10)])
mse = keras_reg.score(X_test, y_test)
y_pred = keras_reg.predict(X_new)

print(y_pred)

"""When we work on the Hyperparameters Fine-Tuning, we want to train and evaluate different combinations of hidden layers, neurons, activation function, learning rates, etc. So, we can use the default Scikit-Learn classes for that step in order to result in the best model-selection for our case study. The RandomizedSearchCV constructor requires as attributes
 the Scikit-Learn object, the parameters (ie a dictionary that maps our 

*   The Scikit-Learn object
*   The parameters (ie a dictionary that maps our proposed values to each hyperparameter)
*   The number of iterations
*   The number of jobs (ie multithreading programming)

The process may take a few hours depending on the complexity of the task. That means, the model/dataset combination.
"""

parameters = {"hidden": [1,2,3,4,5,6], "neurons": [x for x in range(20,50)], "learning_rate": [3e-6, 3e-5, 3e-4, 3e-3, 3e-2], "activation": ["relu", "tanh"]}
random_search = RandomizedSearchCV(keras_reg, parameters, n_iter=10, cv=3)
random_search.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=[keras.callbacks.EarlyStopping(patience=10)])

print(("Best parameters: {}").format(random_search.best_params_))
print(("Best Score: {}").format(random_search.best_score_))

"""Saving this model is simply enough. We can now save it in the form of a h5 file and evaluate it on our test set."""

model = random_search.best_estimator_.model
# Save the Model
model.save("best-model.h5")
# Load the Model
loaded_model = keras.models.load_model("best-model.h5")
# Evaluating on the test set
mse = loaded_model.evaluate(X_test, y_test)
# Assuming new data
y_pred = model.predict(X_new)
print(y_pred)