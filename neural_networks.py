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
# 5000 samples
''' 
32/32 [==============================] - 0s 985us/step - loss: 0.4250 - accuracy: 0.8090
BASIC MODEL: [0.42498987913131714, 0.8090000152587891]
'''

basic_model_with_regularizer()
'''
32/32 [==============================] - 0s 1ms/step - loss: 0.4210 - accuracy: 0.8160
BASIC MODEL WITH OPTIMIZER: [0.4209645688533783, 0.8159999847412109]
'''

complex_model('LeakyReLU')
'''
32/32 [==============================] - 0s 2ms/step - loss: 1.5078 - accuracy: 0.7790
COMPLEX MODEL: [1.5077555179595947, 0.7789999842643738]
'''

#all samples
'''
910/910 [==============================] - 1s 696us/step - loss: 0.3924 - accuracy: 0.8281
BASIC MODEL: [0.3923799693584442, 0.8280970454216003]
'''

'''
910/910 [==============================] - 1s 678us/step - loss: 0.3913 - accuracy: 0.8295
BASIC MODEL WITH OPTIMIZER: [0.3913284242153168, 0.829472005367279]
'''

'''
910/910 [==============================] - 1s 1ms/step - loss: 0.4531 - accuracy: 0.8143
COMPLEX MODEL: [0.45314860343933105, 0.8142788410186768]
'''
