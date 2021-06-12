from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, \
  GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from preprocess_data import preprocessed_data
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn import metrics

import pandas as pd
import warnings

warnings.filterwarnings('ignore')

seed = 123
kfold = StratifiedKFold(n_splits=4, random_state=seed)
X_train, X_test, y_train, y_test = preprocessed_data()


def KNN():
  pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', KNeighborsClassifier())])

  param_grid = {
    'preprocessing': [MinMaxScaler(), StandardScaler(), None],
    'classifier__n_neighbors': [1, 2, 3, 5, 10, 100]
  }

  grid = GridSearchCV(pipe, param_grid, cv=kfold, return_train_score=True)
  grid.fit(X_train, y_train)
  print("KNN", grid.best_params_)
  return grid


def decision_tree_clasifier():
  pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', DecisionTreeClassifier(random_state=0))])
  param_grid = {
    'preprocessing': [MinMaxScaler(), StandardScaler(), None],
    'classifier__max_depth': [1, 2, 5, 10, 100],
    'classifier__min_samples_leaf': [1, 2, 4, 10]
  }

  grid = GridSearchCV(pipe, param_grid, cv=kfold, return_train_score=True)
  grid.fit(X_train, y_train)
  print("Decision tree clasifier", grid.best_params_)
  return grid


def bagging():
  pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1, random_state=42))])

  param_grid = {
    'preprocessing': [MinMaxScaler(), StandardScaler(), None],
    'classifier__n_estimators': [50, 100, 500, 1000],
    'classifier__max_samples': [50, 100, 200]
  }

  grid = GridSearchCV(pipe, param_grid, cv=kfold, return_train_score=True)
  grid.fit(X_train, y_train)
  print("Bagging", grid.best_params_)
  return grid


def random_forrest():
  pipe = Pipeline(
    [('preprocessing', StandardScaler()), ('classifier', RandomForestClassifier(n_jobs=-1, random_state=42))])
  param_grid = {
    'preprocessing': [MinMaxScaler(), StandardScaler(), None],
    'classifier__n_estimators': [50, 100, 500, 1000],
    'classifier__max_leaf_nodes': [4, 8, 16, 64]
  }

  grid = GridSearchCV(pipe, param_grid, cv=kfold, return_train_score=True)
  grid.fit(X_train, y_train)
  print("RF", grid.best_params_)
  return grid


def extra_trees():
  pipe = Pipeline(
    [('preprocessing', StandardScaler()), ('classifier', ExtraTreesClassifier(n_jobs=-1, random_state=42))])
  param_grid = {
    'preprocessing': [MinMaxScaler(), StandardScaler(), None],
    'classifier__n_estimators': [50, 100, 500, 1000],
    'classifier__max_leaf_nodes': [4, 8, 16, 64]
  }

  grid = GridSearchCV(pipe, param_grid, cv=kfold, return_train_score=True)
  grid.fit(X_train, y_train)
  print("ET", grid.best_params_)
  return grid


def ada_boost():
  pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=1, learning_rate=0.5,
    algorithm="SAMME.R", random_state=42))])

  param_grid = {
    'preprocessing': [MinMaxScaler(), StandardScaler(), None],
    'classifier__n_estimators': [50, 100, 500, 1000],
    'classifier__learning_rate': [0.01, 0.1, 0.5, 0.9]
  }

  grid = GridSearchCV(pipe, param_grid, cv=kfold, return_train_score=True)
  grid.fit(X_train, y_train)
  print("ADA", grid.best_params_)
  return grid


def gradient_boost():
  pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', GradientBoostingClassifier(random_state=42))])
  param_grid = {
    'preprocessing': [MinMaxScaler(), StandardScaler(), None],
    'classifier__n_estimators': [50, 100, 500, 1000],
    'classifier__learning_rate': [0.01, 0.1, 0.5, 0.9]
  }

  grid = GridSearchCV(pipe, param_grid, cv=kfold, return_train_score=True)
  grid.fit(X_train, y_train)
  print("GB", grid.best_params_)
  return grid


def xboost_clf():
  pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', XGBClassifier())])
  param_grid = {
    'preprocessing': [MinMaxScaler(), StandardScaler(), None],
    'classifier__n_estimators': [50, 100, 500, 1000],
    'classifier__learning_rate': [0.01, 0.1, 0.5, 0.9]
  }

  grid = GridSearchCV(pipe, param_grid, cv=kfold, return_train_score=True)
  grid.fit(X_train, y_train)
  print("XGB", grid.best_params_)
  return grid


knn = KNN()
dt_clf = decision_tree_clasifier()
bag_clf = bagging()
rf_clf = random_forrest()
et_clf = extra_trees()
ada_clf = ada_boost()
gb_clf = gradient_boost()
xgb_clf = xboost_clf()

models = []
models.append(('KNN', knn.best_estimator_))
models.append(('DecisionTreeClassifier', dt_clf.best_estimator_))
models.append(('BaggingClassifier', bag_clf.best_estimator_))
models.append(('RandomForestClassifier', rf_clf.best_estimator_))
models.append(('ExtraTreesClassifier', et_clf.best_estimator_))
models.append(('AdaBoostClassifier', ada_clf.best_estimator_))
models.append(('GradientBoostingClassifier', gb_clf.best_estimator_))
models.append(('XGBClassifier', xgb_clf.best_estimator_))

precision_score = []
recall_score = []
f1_score = []
accuracy_score = []
roc_auc_score = []
for name, model in models:
  print(name)
  print("precision_score: {}".format(metrics.precision_score(y_test, model.predict(X_test))))
  print("recall_score: {}".format(metrics.recall_score(y_test, model.predict(X_test))))
  print("f1_score: {}".format(metrics.f1_score(y_test, model.predict(X_test))))
  print("accuracy_score: {}".format(metrics.accuracy_score(y_test, model.predict(X_test))))

  if (name == 'SVM linear' or name == 'SVM rbf' or name == 'voting_clf'):
    print("roc_auc_score: {}".format(metrics.roc_auc_score(y_test, model.decision_function(X_test))))
  else:
    print("roc_auc_score: {}".format(metrics.roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])))

  precision_score.append(metrics.precision_score(y_test, model.predict(X_test)))
  recall_score.append(metrics.recall_score(y_test, model.predict(X_test)))
  f1_score.append(metrics.f1_score(y_test, model.predict(X_test)))
  accuracy_score.append(metrics.accuracy_score(y_test, model.predict(X_test)))
  if (name == 'SVM linear' or name == 'SVM rbf' or name == 'voting_clf'):
    roc_auc_score.append(metrics.roc_auc_score(y_test, model.decision_function(X_test)))
  else:
    roc_auc_score.append(metrics.roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

d = {'precision_score': precision_score,
     'recall_score': recall_score,
     'f1_score': f1_score,
     'accuracy_score': accuracy_score,
     'roc_auc_score': roc_auc_score
     }
df = pd.DataFrame(data=d)
df.insert(loc=0, column='Method',
          value=['KNN', 'DecisionTreeClassifier', 'BaggingClassifier', 'RandomForestClassifier', 'ExtraTreesClassifier',
                 'AdaBoostClassifier', 'GradientBoostingClassifier', 'XGBClassifier'])
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(df)


# with preprocessed data
'''
                       Method  precision_score  recall_score  f1_score  
0                         KNN         0.754545      0.354701  0.482558   
1      DecisionTreeClassifier         0.628788      0.354701  0.453552   
2           BaggingClassifier         0.666667      0.410256  0.507937   
3      RandomForestClassifier         0.716418      0.410256  0.521739   
4        ExtraTreesClassifier         0.741935      0.294872  0.422018   
5          AdaBoostClassifier         0.692810      0.452991  0.547804   
6  GradientBoostingClassifier         0.631285      0.482906  0.547215   
7               XGBClassifier         0.641304      0.504274  0.564593  

   accuracy_score  roc_auc_score  
0           0.822       0.790258  
1           0.800       0.797873  
2           0.814       0.817860  
3           0.824       0.830053  
4           0.811       0.819704  
5           0.825       0.834678  
6           0.813       0.824881  
7           0.818       0.820145 
'''
