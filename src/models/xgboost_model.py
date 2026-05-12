from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Create and train model
def train_xgboost(X_train, y_train):
    model = XGBClassifier(
    colsample_bytree=0.5,
    gamma=0.25,
    learning_rate=0.01,
    max_depth=8,
    n_estimators=1000,
    subsample=0.9
    )
    
    model.fit(X_train, y_train)

    return model

def evaluate_xgboost(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_true=y_test, y_pred=y_pred)

    return accuracy, report



def grid_train_xgboost(X_train, y_train):
    model = XGBClassifier()

    param_grid = { 
    'max_depth': [3, 4, 5, 8, 10, 15],
    'learning_rate': [0.01, 0.1, 0.20],
    "gamma":[0, 0.25, 0.5],
    'n_estimators': [100, 500, 1000],
    "subsample":[0.9],
    "colsample_bytree":[0.5],
    }
    
    grid = GridSearchCV(model, param_grid, cv=5, verbose=3)

    grid.fit(X_train, y_train)

    return grid.best_params_, grid.best_score_