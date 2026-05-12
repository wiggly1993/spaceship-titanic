from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.load_data import load_train_data, CustomTabularDataset

from src.models.xgboost_model import train_xgboost, evaluate_xgboost, grid_train_xgboost
from src.models.random_forest import train_random_forest, evaluate_random_forest
from src.models.neural_net import SpaceshipNet
from src.models.training_loop import training_loop

# generate the data from the dataframe prepared in load_data.py
X_train, X_test, y_train, y_test = load_train_data()
# print(X_train.dtypes)

### Train and evaluate a random forest model ###

# train the random forest model
trained_rf = train_random_forest(X_train, y_train)
# evaluate the random forest model we have
accuracy, report = evaluate_random_forest(trained_rf, X_test, y_test)

print(report)

## Train and evaluate xgboost model ###

trained_xg = train_xgboost(X_train, y_train)
accuracy, report = evaluate_xgboost(trained_xg, X_test, y_test)

print(report)

# gridsearch for xgboost model

best_params_, best_score_ = grid_train_xgboost(X_train=X_train, y_train=y_train)

print(f"Best params: {best_params_}")
print(f"Best score: {best_score_} ")


# ### Train an FNN model ###

# # first we scale the data to be always between 0 and 1
# X_scaler = StandardScaler()
# X_train_scaled = X_scaler.fit_transform(X_train)
# # now we scale x_test based on the SAME scalar
# X_test_scaled = X_scaler.transform(X_test)


# # now import datasets 

# train_set = CustomTabularDataset(X_train_scaled, y_train)
# test_set = CustomTabularDataset(X_test_scaled, y_test)

# # now create the dataloaders 

# #Split into batches
# batch_size = 32
# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# # create the network
# net = SpaceshipNet()

# training_loop(model=net, train_loader=train_loader, test_loader=test_loader, num_epochs=15, show_progress=True)