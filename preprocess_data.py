import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocessed_data():
  df = pd.read_csv('./weatherAUS.csv')
  df.dropna(inplace=True)
  df['RainToday'] = [int(x != 'No') for x in df['RainToday']]
  df['RainTomorrow'] = [int(x != 'No') for x in df['RainTomorrow']]

  y = df['RainTomorrow']
  del df['RainTomorrow']
  del df['Date']
  del df['Location']
  del df['WindGustDir']
  del df['WindDir9am']
  del df['WindDir3pm']

  X = df
  X_train, X_test , y_train, y_test = train_test_split(X,y,test_size=0.20)

  return X_train, X_test , y_train, y_test

