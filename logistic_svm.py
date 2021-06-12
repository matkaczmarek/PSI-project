from preprocess_data import preprocessed_data
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
from scipy.stats.distributions import uniform

import pandas as pd
import warnings

warnings.filterwarnings('ignore')

seed = 123
kfold = StratifiedKFold(n_splits=4, random_state=seed)
X_train, X_test, y_train, y_test = preprocessed_data()


def LR_gridsearch():
  param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

  grid = GridSearchCV(LogisticRegression(), param_grid, cv=kfold, return_train_score=True)

  grid.fit(X_train, y_train)
  print("LR", grid.best_params_)
  return grid


def svm_gridsearch():
  param_distribution = {
    'C': uniform(0.001, 0.1 - 0.001),
    'gamma': uniform(0.0001, 2)
  }
  grid = RandomizedSearchCV(SVC(kernel='rbf'), param_distribution, random_state=0)
  grid.fit(X_train, y_train)
  print("SVC", grid.best_params_)
  return grid


model_lr = LR_gridsearch()
model_svm = svm_gridsearch()

models = []
models.append(('LR', model_lr))
models.append(('SVM', model_svm))

precision_score = []
recall_score = []
f1_score = []
accuracy_score = []
for name, model in models:
  print(name)
  print("precision_score: {}".format(metrics.precision_score(y_test, model.predict(X_test), average='micro')))
  print("recall_score: {}".format(metrics.recall_score(y_test, model.predict(X_test), average='micro')))
  print("f1_score: {}".format(metrics.f1_score(y_test, model.predict(X_test), average='micro')))
  print("accuracy_score: {}".format(metrics.accuracy_score(y_test, model.predict(X_test))))
  precision_score.append(metrics.precision_score(y_test, model.predict(X_test), average='micro'))
  recall_score.append(metrics.recall_score(y_test, model.predict(X_test), average='micro'))
  f1_score.append(metrics.f1_score(y_test, model.predict(X_test), average='micro'))
  accuracy_score.append(metrics.accuracy_score(y_test, model.predict(X_test)))

d = {'precision_score': precision_score,
     'recall_score': recall_score,
     'f1_score': f1_score,
     'accuracy_score': accuracy_score
     }
df = pd.DataFrame(data=d, index=[0, 1])
df.insert(loc=0, column='Method', value=['LR', 'SVM'])
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(df)


# 5000 samples
'''
  Method  precision_score  recall_score  f1_score  accuracy_score
0     LR            0.807         0.807     0.807           0.807
1    SVM            0.741         0.741     0.741           0.741
'''
