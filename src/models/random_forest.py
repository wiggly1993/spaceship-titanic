import sys, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_random_forest(X_train, y_train):
    trained_rf = RandomForestClassifier(max_depth=None, random_state=0)
    trained_rf.fit(X_train, y_train)

    return trained_rf


def evaluate_random_forest(model, X_test, y_test):
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    report = classification_report(y_true=y_test, y_pred=pred)


    print("evaluation function concluded without errors")
    return accuracy, report
