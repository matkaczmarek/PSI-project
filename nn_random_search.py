import warnings

import numpy as np
import tensorflow
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from tensorflow import keras
from tensorflow.python.keras.layers import BatchNormalization, Dropout, Dense
from tensorflow.python.keras.models import Sequential

from preprocess_data import preprocessed_data

warnings.filterwarnings('ignore')

X_train, X_test, y_train, y_test = preprocessed_data()


def build_model(n_hidden1=1, n_neurons1=30, n_hidden2=1, n_neurons2=30):
  model = Sequential()
  for layer in range(n_hidden1):
    model.add(Dense(n_neurons1, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
  for layer in range(n_hidden2):
    model.add(Dense(n_neurons2, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
  model.add(Dense(1, activation="sigmoid"))
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  return model


def random_search():
  keras_class = tensorflow.keras.wrappers.scikit_learn.KerasClassifier(build_model)

  param_distribs = {
    "n_hidden1": [0, 1, 2, 3],
    "n_neurons1": np.arange(8, 256),
    "n_hidden2": [0, 1, 2, 3],
    "n_neurons2": np.arange(4, 128)
  }

  rnd_search_cv = RandomizedSearchCV(keras_class, param_distribs, n_iter=10, cv=3, verbose=2)
  rnd_search_cv.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), batch_size=32)

  print(rnd_search_cv.best_params_)
  print("RANDOM SEARCH:", metrics.accuracy_score(y_true=y_test, y_pred=rnd_search_cv.best_estimator_.predict(X_test)))

#random_search()

'''
{'n_neurons2': 95, 'n_neurons1': 106, 'n_hidden2': 0, 'n_hidden1': 0}
COMPLEX MODEL: 0.776
'''

def build_model_v2(n_hidden1=1, n_neurons1=30, n_hidden2=1, n_neurons2=30):
  model = Sequential()
  for layer in range(n_hidden1):
    model.add(Dense(n_neurons1, activation="relu"))
  for layer in range(n_hidden2):
    model.add(Dense(n_neurons2, activation="relu"))
  model.add(Dense(1, activation="sigmoid"))
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  return model


def random_search_without_dropout():
  keras_class = tensorflow.keras.wrappers.scikit_learn.KerasClassifier(build_model_v2)

  param_distribs = {
    "n_hidden1": [0, 1, 2, 3],
    "n_neurons1": np.arange(8, 256),
    "n_hidden2": [0, 1, 2, 3],
    "n_neurons2": np.arange(4, 128)
  }

  rnd_search_cv = RandomizedSearchCV(keras_class, param_distribs, n_iter=10, cv=3, verbose=2)
  rnd_search_cv.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), batch_size=32)

  print(rnd_search_cv.best_params_)
  print("RANDOM SEARCH WITHOUT DROPOUT:", metrics.accuracy_score(y_true=y_test, y_pred=rnd_search_cv.best_estimator_.predict(X_test)))

random_search_without_dropout()

'''
{'n_neurons2': 114, 'n_neurons1': 153, 'n_hidden2': 2, 'n_hidden1': 0}
RANDOM SEARCH WITHOUT DROPOUT: 0.78
'''
