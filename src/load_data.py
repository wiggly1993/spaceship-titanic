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
    # print(f"{df.head()}")
    # print(f"this is shape: {df.shape}")
    # print(f"this is dtypes: {df.dtypes}")
    # print(f"this is sum(): {df.isnull().sum()}")

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


    df_extracted = df["PassengerId"].str.split(pat="_", expand=True)
    df_extracted = df_extracted.rename(columns={0: "GroupId", 1: "Person_num"})


    # drop the irrelevant columns from original df
    df_dropped = df.drop(labels=["PassengerId", "Name", "Cabin"], axis=1)

    # continue here with df_dropped
    # concatenate the dropped df and the 3 new cols from splitted_cols
    combined_df = pd.concat([df_extracted["GroupId"], df_dropped, renamed_cols], axis="columns")

    ## next in this part we will go over all features (cols) that have missing entries
    ## and fill the up with simple median (numerical) or mode (boolean) values

    # get the cols for numerical values first
    numerical_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "num"]
    mode_cols = ["HomePlanet", "Destination", "CryoSleep", "VIP", "deck", "side"]


    # calculate the median values based on groupid (better) - this creates a sub_df
    fill_in_values = combined_df.groupby(["GroupId"])[numerical_cols].transform('median')
    # use this subdf to create another subdf - essentially a df with filled in values but 
    # ONLY the numerical cols nothing else (will be concatenated later)
    filled_median_df = combined_df[numerical_cols].fillna(fill_in_values)

    #next do the same for the mode cols
    def my_mode(series):
        return series.mode().iloc[0]
    
    # create the fill in values in small df
    fill_in_values = combined_df.groupby(["GroupId"])[mode_cols].transform(my_mode)
    # create a subdf that has filled in values but ONLY the columns of mode cols
    filled_modes_df = combined_df[mode_cols].fillna(fill_in_values)

    # now concatenate everything together
    combined_df = pd.concat([combined_df["GroupId"], filled_median_df, filled_modes_df, combined_df["Transported"]], axis="columns")


    # the issue now is that ppl that travel alone (i.e. group to themselves) have no entries because 
    # mean/mode gives no value. This means we need to find the remaining empties and fill them with global averages
    combined_df = combined_df.fillna(combined_df[mode_cols].mode().iloc[0])
    combined_df = combined_df.fillna(combined_df[numerical_cols].median())

    # print(f"this is combined_df.sum(): {combined_df.isnull().sum()}")
    # print(combined_df.head())

    # split categories into multiple columns, quite crazy but works
    one_hotted_df = pd.get_dummies(
        combined_df[["HomePlanet", "Destination", "deck", "side"]], dtype=int)
    
    # drop the labels that we have one hotted anyway and will concatenate them next
    combined_df = combined_df.drop(labels=["HomePlanet", "Destination", "deck", "side"], axis=1)

    # this does not need to be concatenated it happens within the combined df 
    # true/false was replaced by 0/1
    combined_df[["CryoSleep", "VIP", "Transported"]] = combined_df[["CryoSleep", "VIP", "Transported"]].astype("int32")

    # concatenate the one_hotted cols with the rest together 
    final_df = pd.concat([combined_df, one_hotted_df], axis="columns")

    #print(final_df)

    ## Next goal will be to extract the inputs (X) and targets (Y) from this dataframe

    # separate train data X from target col y
    X, y = final_df.drop(columns=["Transported"]), final_df["Transported"]


    # split for training purposes
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test





class CustomTabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)
        self.y = self.y.unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



if __name__ == "__main__":
    # X_train, X_test, y_train, y_test = load_train_data()

    # train_set = CustomTabularDataset(X_train, y_train)

    # X, y = train_set[1]
    # print(X.shape)
    load_train_data()

