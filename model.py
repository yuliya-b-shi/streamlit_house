from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from pickle import dump, load
import pandas as pd
import numpy as np


def split_data(df: pd.DataFrame):
    y = df['SalePrice']
    X = df[["BldgType", "Utilities", "OverallCond", "GrLivArea"]]

    return X, y


def open_data(path="data/train.csv"):
    df = pd.read_csv(path)
    df = df[['SalePrice', "BldgType", "Utilities", "OverallCond", "GrLivArea"]]

    return df


def preprocess_data(df: pd.DataFrame, test=True):

    if test:
        X_df, y_df = split_data(df)
    else:
        X_df = df

    coder1 = preprocessing.LabelEncoder().fit(X_df['BldgType'])
    X_df['BldgType'] = coder1.transform(X_df['BldgType'])
    coder2 = preprocessing.LabelEncoder().fit(X_df['Utilities'])
    X_df['Utilities'] = coder2.transform(X_df['Utilities'])
    coder3 = preprocessing.LabelEncoder().fit(X_df['OverallCond'])
    X_df['OverallCond'] = coder3.transform(X_df['OverallCond'])
    if test:
        return X_df, y_df
    else:
        return X_df


def fit_and_save_model(X_df, y_df, path='data/model_weights.mw'):
    model = LinearRegression()
    model.fit(X_df, y_df)

    test_prediction = model.predict(X_df)
    error = mean_absolute_error(y_df, test_prediction)
    print(f"Model MSE is {error}")

    with open(path, "wb") as file:
        dump(model, file)

    print(f"Model was saved to {path}")

def load_model_and_predict(df, path="data/model_weights.mw"):
    with open(path, "rb") as file:
        model = load(file)

    prediction = model.predict(df) #[0]

    return np.round(prediction, 2)


if __name__ == "__main__":
    df = open_data()
    X_df, y_df = preprocess_data(df)
    fit_and_save_model(X_df, y_df)