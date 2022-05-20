#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Receiving FitBit data in JSON format and infering Fatigue Assessment Score
using machine learning model.

Created:
    2022-05-07

"""
from datetime import datetime
import joblib

import pandas as pd


class FitBitDataFrame:
    def __init__(self):

        self.dfs = []

    def read_profile(self, json_input):

        json_input = json_input["user"]

        self.age = datetime.now().year - int(json_input["dateOfBirth"][0][:4])
        self.gender = 0 if json_input["gender"][0] == "FEMALE" else 1
        self.weight = json_input["weight"]
        self.height = json_input["height"]

    def read_sleep(self, json_input):

        df = pd.read_json(json_input)
        print(df)

        df["dateOfSleep"] = pd.to_datetime(df["dateOfSleep"])
        df.set_index("dateOfSleep", inplace=True)

        self.dfs.append(df)

    def read_timeseries(self, name, json_input, sum_values=False):

        df = pd.read_json(json_input)

        # Change dateTime column to datetime type
        df["dateTime"] = pd.to_datetime(df["dateTime"])

        value_columns = [c for c in df.columns if c.startswith("value")]

        # Change value column to int type
        if len(value_columns) == 1:
            df[name] = df["value"].astype("float")
            del df["value"]
            new_value_columns = [name]
        else:
            new_value_columns = []
            for column in value_columns:
                new_column_name = name + "_" + column.split(".")[-1]
                df[new_column_name] = df[column].astype("float")
                del df[column]
                new_value_columns.append(new_column_name)

        df_resampled = pd.DataFrame()

        for column in new_value_columns:
            # Sum or mean values per day
            if sum_values:
                df_resampled[column] = df.resample("D", on="dateTime")[column].sum()
            else:
                df_resampled[column + "_max"] = df.resample("D", on="dateTime")[
                    column
                ].max()
                df_resampled[column + "_min"] = df.resample("D", on="dateTime")[
                    column
                ].min()
                df_resampled[column + "_mean"] = df.resample("D", on="dateTime")[
                    column
                ].mean()

        self.dfs.append(df_resampled)

    def combine_data_and_profile(self, json_input):

        self.df = self.dfs[0]

        for df in self.dfs[1:]:
            self.df = self.df.join(df)

        # Add user profile information to each line of the data frame
        self.read_profile(json_input)
        self.df["age"] = self.age
        self.df["gender"] = self.gender
        self.df["weight"] = self.weight
        self.df["height"] = self.height

        self.df = self.df.fillna(0)

        return self.df


def infer(input_data, scaler_filepath, model_filepath, input_columns):
    """Run inference using a machine learning model.

    Args:
        input_data (DataFrame): DataFrame containing the input data to use when
            running inference. The input columns specified in input_columns
            must be present in input_data.
        scaler_filepath (str): Filepath of scaler.
        model_filepath (str): Filepath of model.
        input_columns (list): List of columns to use as input to the model. All
            input columns must be present in input_data.

    Returns:
        y (array): Predictions from machine learning model based on input_data.

    """

    # Load model
    # model = models.load_model(model_filepath)
    model = joblib.load(model_filepath)

    # Load scaler
    scaler = joblib.load(scaler_filepath)

    # Select variables/columns to use
    input_data = input_data[input_columns]

    # Convert to NumPy array
    input_data = input_data.to_numpy()

    # Scale input data
    input_data = scaler.transform(input_data)

    # Infer
    y = model.predict(input_data)

    # Print results
    print(y)

    return y
