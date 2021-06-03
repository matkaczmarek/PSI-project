import warnings

import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.python.keras.backend import clear_session
from tensorflow.python.keras.callbacks import History
from tensorflow.python.keras.layers import Dense, Activation, BatchNormalization, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.regularizers import l1

from preprocess_data import preprocessed_data

warnings.filterwarnings('ignore')

X_train, X_test, y_train, y_test = preprocessed_data()


def plot_history(history):
  pd.DataFrame(history.history).plot(figsize=(8, 5))
  plt.grid(True)
  plt.gca().set_ylim(0, 1)
  plt.show()


def basic_model():
  clear_session()

  model = Sequential()
  model.add(Dense(10, activation="relu", input_shape=(X_train.shape[1],)))
  model.add(Dense(128, activation="sigmoid"))
  model.add(Dense(64, activation="sigmoid"))
  model.add(Dense(32, activation="sigmoid"))
  model.add(Dense(10, activation="sigmoid"))
  model.add(Dense(1, activation="sigmoid"))
  print(model.summary())

  model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])

  history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))
  plot_history(history)
  print("BASIC MODEL:", model.evaluate(X_test, y_test))


def basic_model_with_regularizer():
  clear_session()

  history = History()
  model = Sequential()
  model.add(Dense(128, activation="relu", input_shape=(X_train.shape[1],), activity_regularizer=l1(0.00001)))
  model.add(Dense(64, activation="sigmoid", activity_regularizer=l1(0.00001)))
  model.add(Dense(32, activation="sigmoid", activity_regularizer=l1(0.00001)))
  model.add(Dense(10, activation="sigmoid"))
  model.add(Dense(1, activation="sigmoid"))
  model.summary()

  model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

  model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, callbacks=[history])
  plot_history(history)
  print("BASIC MODEL WITH OPTIMIZER:", model.evaluate(X_test, y_test))


def complex_model(activation):
  clear_session()

  model = Sequential()
  model.add(Dense(10, input_shape=(X_train.shape[1],)))
  model.add(Activation(activation=activation))
  model.add(BatchNormalization())

  model.add(Dense(128))
  model.add(Activation(activation=activation))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))

  model.add(Dense(256))
  model.add(Activation(activation=activation))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))

  model.add(Dense(128))
  model.add(Activation(activation='sigmoid'))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))

  model.add(Dense(64))
  model.add(Activation(activation='sigmoid'))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))

  model.add(Dense(32))
  model.add(Activation(activation=activation))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))

  model.add(Dense(10))
  model.add(Activation(activation=activation))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))

  model.add(Dense(units=1, activation=activation))

  model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])

  history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))
  plot_history(history)
  print("COMPLEX MODEL:", model.evaluate(X_test, y_test))


basic_model()
'''
32/32 [==============================] - 0s 765us/step - loss: 0.3484 - accuracy: 0.8410
BASIC MODEL: [0.3484397530555725, 0.8410000205039978]

32/32 [==============================] - 0s 714us/step - loss: 0.4109 - accuracy: 0.8200
BASIC MODEL: [0.4109137952327728, 0.8199999928474426]
'''

basic_model_with_regularizer()
'''
32/32 [==============================] - 0s 708us/step - loss: 0.3258 - accuracy: 0.8630
BASIC MODEL WITH OPTIMIZER: [0.32577699422836304, 0.8629999756813049]

32/32 [==============================] - 0s 759us/step - loss: 0.3674 - accuracy: 0.8440
BASIC MODEL WITH OPTIMIZER: [0.3673613369464874, 0.843999981880188]
'''

complex_model('LeakyReLU')
'''
32/32 [==============================] - 0s 1ms/step - loss: 0.3960 - accuracy: 0.8520
COMPLEX MODEL: [0.39596810936927795, 0.8519999980926514]

32/32 [==============================] - 0s 1ms/step - loss: 0.7872 - accuracy: 0.7740
COMPLEX MODEL: [0.7871930003166199, 0.7739999890327454]
'''
