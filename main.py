from src.load_data import load_train_data
from src.models.xgboost_model import train_xgboost, evaluate_xgboost, grid_train_xgboost
from src.models.random_forest import train_random_forest, evaluate_random_forest


# generate the data from the dataframe prepared in load_data.py
X_train, X_test, y_train, y_test = load_train_data()

# ## Train and evaluate a random forest model
# # train the random forest model
# trained_rf = train_random_forest(X_train, y_train)
# # evaluate the random forest model we have
# accuracy, report = evaluate_random_forest(trained_rf, X_test, y_test)



# Train and evaluate xgboost model
# train the xgboost model

trained_xg = train_xgboost(X_train, y_train)
accuracy , report = evaluate_xgboost(trained_xg, X_test, y_test)

print(report)

## gridsearch for xgboost model

# best_params_, best_score_ = grid_train_xgboost(X_train=X_train, y_train=y_train)

# print(f"Best params: {best_params_}")
# print(f"Best score: {best_score_} ")


