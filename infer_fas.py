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
from tensorflow.keras import models


class FitBitDataFrame:
    def __init__(self):

        self.dfs = []

    def read_profile(self, data_dict):

        self.age = datetime.now().year - int(data_dict["dateOfBirth"][:4])
        self.gender = 0 if data_dict["gender"][0] == "FEMALE" else 1
        self.weight = data_dict["weight"]
        self.height = data_dict["height"]
        self.bmi = self.weight / ((self.height / 100) ** 2)

    def read_sleep(self, data_dict):

        # df = pd.read_json(data_dict)
        df = pd.DataFrame.from_dict(data_dict)

        df["dateOfSleep"] = pd.to_datetime(df["dateOfSleep"])
        df.set_index("dateOfSleep", inplace=True)

        # Delete rows which does not contain main sleep
        df = df[df.isMainSleep == True]

        levels = pd.json_normalize(df["levels"]).add_prefix("levels.")
        levels.index = df.index
        df = df.join(levels)

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

    def read_heart_rate(self, data_dict):

        df = pd.DataFrame(
            columns=[
                "dateTime",
                "resting_heart_rate",
                "heart_rate_bpm_min",
                "heart_rate_bpm_max",
                "heart_rate_bpm_mean",
            ]
        )

        for obj in data_dict:

            date = obj["activities-heart"][0]["dateTime"]
            resting_heart_rate = obj["activities-heart"][0]["value"]["restingHeartRate"]

            intraday_resampled = pd.DataFrame()
            intraday = obj["activities-heart-intraday"]["dataset"]
            intraday = pd.DataFrame.from_dict(intraday)
            intraday.set_index("time", inplace=True)

            heart_rate_bpm_max = intraday["value"].max()
            heart_rate_bpm_min = intraday["value"].min()
            heart_rate_bpm_mean = intraday["value"].mean()

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [
                            {
                                "dateTime": date,
                                "resting_heart_rate": resting_heart_rate,
                                "heart_rate_bpm_min": heart_rate_bpm_min,
                                "heart_rate_bpm_max": heart_rate_bpm_max,
                                "heart_rate_bpm_mean": heart_rate_bpm_mean,
                            }
                        ]
                    ),
                ]
            )

        df["dateTime"] = pd.to_datetime(df["dateTime"])
        df.set_index("dateTime", inplace=True)
        self.dfs.append(df)

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
        self.df["bmi"] = self.bmi

        self.df = self.df.fillna(0)

        return self.df


def infer(input_data, model_filepath, deep_learning=True):

    # Load model
    if deep_learning:
        model = models.load_model(model_filepath)
    else:
        model = joblib.load(model_filepath)

    # Infer
    y = model.predict(input_data)

    return y


def preprocess_and_infer(
    input_json_str, scaler_filepath, model_filepath, input_columns
):
    """Preprocess data and pass it to inference.

    Args:
        input_json_str (str): JSON string containing input data.
        scaler_filepath (str): Filepath to scaler object.
        model_filepath (str): Filepath to model.
        input_columns (list): List of input columns.

    Returns:
        output_json (str): Results from inference as JSON string.

    """

    input_json = json.loads(input_json_str)

    output = []

    for user_data in input_json:
        user_id = user_data["userid"]

        f = FitBitDataFrame()

        f.read_timeseries("calories", user_data["activities-calories"])
        f.read_timeseries("distance", user_data["activities-distance"])
        f.read_timeseries("steps", user_data["activities-steps"], sum_values=True)
        f.read_timeseries(
            "lightly_active_minutes",
            user_data["activities-minutesLightlyActive"],
            sum_values=True,
        )
        f.read_timeseries(
            "moderately_active_minutes",
            user_data["activities-minutesFairlyActive"],
            sum_values=True,
        )
        f.read_timeseries(
            "very_active_minutes",
            user_data["activities-minutesVeryActive"],
            sum_values=True,
        )
        f.read_timeseries(
            "sedentary_minutes",
            user_data["activities-minutesSedentary"],
            sum_values=True,
        )
        f.read_heart_rate(user_data["heartrate"])
        f.read_sleep(user_data["sleep"])

        f.combine_data_and_profile(user_data["user"])

        # Select variables/columns to use
        input_data = f.df[input_columns]

        # Convert to NumPy array
        input_data = input_data.to_numpy()

        # Load scaler
        scaler = joblib.load(scaler_filepath)

        # Scale input data
        input_data = scaler.transform(input_data)

        # Select the latest data n data points, where n=window_size
        window_size = json.load(open("data/params.json"))["window_size"]
        input_data = input_data[-window_size:, :].reshape(1, -1)

        deep_learning = json.load(open("data/params.json"))["deep_learning"]
        y = infer(input_data, model_filepath, deep_learning=deep_learning)

        # The latest FAS value is returned for each user.
        output.append({"userid": user_id, "fas": str(y[-1])})

    output_json = json.dumps(output)

    return output_json
