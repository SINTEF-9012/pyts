#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Receiving FitBit data in JSON format and infering Fatigue Assessment Score
using machine learning model.

Created:
    2022-05-07

"""
import json
from datetime import datetime

import joblib
import pandas as pd


class FitBitDataFrame:
    def __init__(self):

        self.dfs = []

    def read_profile(self, data_dict):

        self.age = datetime.now().year - int(data_dict["dateOfBirth"][0][:4])
        self.gender = 0 if data_dict["gender"][0] == "FEMALE" else 1
        self.weight = data_dict["weight"]
        self.height = data_dict["height"]

    def read_sleep(self, data_dict):

        df = pd.read_json(data_dict)
        print(df)

        df["dateOfSleep"] = pd.to_datetime(df["dateOfSleep"])
        df.set_index("dateOfSleep", inplace=True)

        self.dfs.append(df)

    def read_timeseries(self, name, data_dict, sum_values=False):

        df = pd.DataFrame.from_dict(data_dict)

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

    def combine_data_and_profile(self, data_dict):

        self.df = self.dfs[0]

        for df in self.dfs[1:]:
            self.df = self.df.join(df)

        # Add user profile information to each line of the data frame
        self.read_profile(data_dict)
        self.df["age"] = self.age
        self.df["gender"] = self.gender
        self.df["weight"] = self.weight
        self.df["height"] = self.height

        self.df = self.df.fillna(0)

        return self.df


def infer(input_data, scaler_filepath, model_filepath, input_columns):

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
    print("Results: ", y)

    return y


def preprocess_and_infer(
    input_json_str, scaler_filepath, model_filepath, input_columns
):

    input_json = json.loads(input_json_str)

    output = []

    for user_data in input_json:
        print("=======")
        user_id = user_data["userid"]
        print(user_id)

        f = FitBitDataFrame()

        f.read_timeseries("calories", user_data["activities-calories"])
        f.read_timeseries("distance", user_data["activities-distance"])
        f.read_timeseries("steps", user_data["activities-steps"], sum_values=True)
        f.read_timeseries(
            "minutesLightlyActive",
            user_data["activities-minutesLightlyActive"],
            sum_values=True,
        )
        f.read_timeseries(
            "minutesFairlyActive",
            user_data["activities-minutesFairlyActive"],
            sum_values=True,
        )
        f.read_timeseries(
            "minutesVeryActive",
            user_data["activities-minutesVeryActive"],
            sum_values=True,
        )
        f.read_timeseries(
            "minutesSedentary",
            user_data["activities-minutesSedentary"],
            sum_values=True,
        )

        # TODO: Adapt these functions to sample_input.json.
        # These are not tested with the new format.
        # f.read_sleep(get_sleep())
        # f.read_timeseries("heart_rate", get_heart_rate_data("heart_rate"))

        f.combine_data_and_profile(user_data["user"])

        y = infer(f.df, scaler_filepath, model_filepath, input_columns)

        # The latest FAS value is returned for each user.
        output.append({"userid": user_id, "fas": str(y[-1])})

    output_json = json.dumps(output)

    return output_json
