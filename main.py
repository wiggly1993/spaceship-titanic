from src.load_data import load_train_data
from src.models.random_forest import train_random_forest, evaluate_random_forest


# generate the data from the dataframe prepared in load_data.py
X_train, X_test, y_train, y_test = load_train_data()








## Train and evaluate a random forest model
# train the random forest model
trained_rf = train_random_forest(X_train, y_train)
# evaluate the random forest model we have
accuracy, report = evaluate_random_forest(trained_rf, X_test, y_test)




print(report)