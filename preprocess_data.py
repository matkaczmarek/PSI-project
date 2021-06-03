import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

num_of_samples = 5000


def fill_nas(df):
  df["MinTemp"] = df["MinTemp"].fillna(df["MinTemp"].mean())
  df["MaxTemp"] = df["MaxTemp"].fillna(df["MaxTemp"].mean())
  df["Evaporation"] = df["Evaporation"].fillna(df["Evaporation"].mean())
  df["Sunshine"] = df["Sunshine"].fillna(df["Sunshine"].mean())
  df["WindGustSpeed"] = df["WindGustSpeed"].fillna(df["WindGustSpeed"].mean())
  df["Rainfall"] = df["Rainfall"].fillna(df["Rainfall"].mean())
  df["WindSpeed9am"] = df["WindSpeed9am"].fillna(df["WindSpeed9am"].mean())
  df["WindSpeed3pm"] = df["WindSpeed3pm"].fillna(df["WindSpeed3pm"].mean())
  df["Humidity9am"] = df["Humidity9am"].fillna(df["Humidity9am"].mean())
  df["Humidity3pm"] = df["Humidity3pm"].fillna(df["Humidity3pm"].mean())
  df["Pressure9am"] = df["Pressure9am"].fillna(df["Pressure9am"].mean())
  df["Pressure3pm"] = df["Pressure3pm"].fillna(df["Pressure3pm"].mean())
  df["Cloud9am"] = df["Cloud9am"].fillna(df["Cloud9am"].mean())
  df["Cloud3pm"] = df["Cloud3pm"].fillna(df["Cloud3pm"].mean())
  df["Temp9am"] = df["Temp9am"].fillna(df["Temp9am"].mean())
  df["Temp3pm"] = df["Temp3pm"].fillna(df["Temp3pm"].mean())

  df['RainToday'] = df['RainToday'].fillna(df['RainToday'].mode()[0])
  df['RainTomorrow'] = df['RainTomorrow'].fillna(df['RainTomorrow'].mode()[0])
  df['WindDir9am'] = df['WindDir9am'].fillna(df['WindDir9am'].mode()[0])
  df['WindGustDir'] = df['WindGustDir'].fillna(df['WindGustDir'].mode()[0])
  df['WindDir3pm'] = df['WindDir3pm'].fillna(df['WindDir3pm'].mode()[0])
  return df


def plot_correlation(df):
  plt.figure(figsize=(18, 12))
  sns.heatmap(df.corr(), annot=True)
  plt.xticks(rotation=90)
  plt.show()


def preprocessed_data(verbose=False):
  df = pd.read_csv('./weatherAUS.csv')
  if verbose:
    print(df.isna().sum())

  # change Rain to 0, 1 value
  df['RainToday'] = [int(x != 'No') for x in df['RainToday']]
  df['RainTomorrow'] = [int(x != 'No') for x in df['RainTomorrow']]

  # fill missing values
  df = fill_nas(df)

  # change location and wind directions to numerical values
  location_dict = {name: c for c, name in enumerate(df['Location'].unique())}
  wind_dir_9 = {name: c for c, name in enumerate(df['WindDir9am'].unique())}
  wind_dir_3 = {name: c for c, name in enumerate(df['WindDir3pm'].unique())}
  wind_dir_gust = {name: c for c, name in enumerate(df['WindGustDir'].unique())}
  df['Location'] = [location_dict[x] for x in df['Location']]
  df['WindDir9am'] = [wind_dir_9[x] for x in df['WindDir9am']]
  df['WindDir3pm'] = [wind_dir_3[x] for x in df['WindDir3pm']]
  df['WindGustDir'] = [wind_dir_gust[x] for x in df['WindGustDir']]

  if verbose:
    plot_correlation(df)
  df = df.drop(['Temp3pm', 'Temp9am', 'Humidity9am', "Date"], axis=1)

  df.dropna(inplace=True)

  y = df['RainTomorrow'][:num_of_samples]
  del df['RainTomorrow']

  X = df.head(num_of_samples)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

  return X_train, X_test, y_train, y_test
