import os, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

def load_train_data():
    """
    input: none
    returns: X_train, X_test, y_train, y_test
    """
    train_path = "./data/train.csv"

    df = pd.read_csv(train_path)

    ### Data Frame Preparation
    ## the overall goal is going to be:
    ## 1) create new features for cabin because it carries 3 at once d/n/s
    ## 2) drop the old cabin feature since we created 3 new ones out of it
    ## 3) drop names and passengerid as (for now) we assume that this info will be useless
    ## 4) fill in the missing values (~2%) with medians or modes 
    ## 5) turn categorical data (even multi class) into one hot encoded features
    ## this turns one feature col with 3 categories into 3 feature cols with 0/1 (get dummies)
    ## 6) turn true/false features into 1/0 values (astype(int)) 
    ## 7) separate the input data X and target data
    ## 8) split the data 80/20 using sklearn
    ## 9) return X_train, X_test, y_train, y_test

    # get the specific from cabin into 3 different columns (expand=true)
    cabin_splitted_cols = df["Cabin"].str.split(pat="/", expand=True)
    #rename them these new cols 
    renamed_cols = cabin_splitted_cols.rename(columns={0: "deck", 1: "num", 2: "side"})
    # use pd.to_numeric to convert the num column to numerical values and no longer string
    renamed_cols["num"] = pd.to_numeric(renamed_cols["num"])

    # Extract GroupId for better data filling logic
    df_extracted = df["PassengerId"].str.split(pat="_", expand=True)
    group_id_col = df_extracted[0].rename("GroupId")

    # drop the irrelevant columns from original df
    df_dropped = df.drop(labels=["PassengerId", "Name", "Cabin"], axis=1)

    # concatenate the dropped df, the cabin cols, and the GroupId
    combined_df = pd.concat([group_id_col, df_dropped, renamed_cols], axis="columns")

    ## next in this part we will go over all features (cols) that have missing entries
    ## and fill the up with simple median (numerical) or mode (boolean) values

    # get the cols for numerical values first
    numerical_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "num"]
    mode_cols = ["HomePlanet", "Destination", "CryoSleep", "VIP", "deck", "side"]

    # calculate the median values based on GroupId - fill numericals
    fill_in_medians = combined_df.groupby("GroupId")[numerical_cols].transform('median')
    combined_df[numerical_cols] = combined_df[numerical_cols].fillna(fill_in_medians)

    # create a helper function for mode since pandas transform is picky with it
    def get_mode(x):
        m = x.mode()
        return m.iloc[0] if not m.empty else np.nan
    
    # fill in categorical modes based on GroupId
    fill_in_modes = combined_df.groupby("GroupId")[mode_cols].transform(get_mode)
    combined_df[mode_cols] = combined_df[mode_cols].fillna(fill_in_modes)

    # fill remaining empties (people traveling alone) with global averages
    combined_df[mode_cols] = combined_df[mode_cols].fillna(combined_df[mode_cols].mode().iloc[0])
    combined_df[numerical_cols] = combined_df[numerical_cols].fillna(combined_df[numerical_cols].median())

    # split categories into multiple columns
    one_hotted_df = pd.get_dummies(combined_df[["HomePlanet", "Destination", "deck", "side"]], dtype=int)
    
    # drop the labels that we have one hotted anyway
    combined_df = combined_df.drop(labels=["HomePlanet", "Destination", "deck", "side"], axis=1)

    # turn true/false features into 1/0 values
    bool_cols = ["CryoSleep", "VIP", "Transported"]
    combined_df[bool_cols] = combined_df[bool_cols].astype(int)

    # concatenate everything together and force float32 to prevent memory fragments
    # We explicitly drop GroupId here because XGBoost cannot handle string/object types
    final_df = pd.concat([combined_df, one_hotted_df], axis="columns")
    final_df = final_df.drop(columns=["GroupId"]).astype("float32")

    ## Next goal will be to extract the inputs (X) and targets (Y) from this dataframe

    # separate train data X from target col y
    X, y = final_df.drop(columns=["Transported"]), final_df["Transported"]

    # split for training purposes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test







class CustomTabularDataset(Dataset):
    def __init__(self, X, y):
        if isinstance(X,(pd.DataFrame)):
            self.X = torch.tensor(X.values, dtype=torch.float32)
        if isinstance(X,(np.ndarray)):
            self.X = torch.tensor(X, dtype=torch.float32)
       
        self.y = torch.tensor(y.values, dtype=torch.float32)
        self.y = self.y.unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_train_data()

    #print(type(X_train))

    # print(X_train.dtypes)
    # print(X_train.isnull().sum().sum())
    # print(y_train.isnull().sum())
    train_set = CustomTabularDataset(X_train, y_train)

    X, y = train_set[1]
    print(X.shape)
    # load_train_data()

