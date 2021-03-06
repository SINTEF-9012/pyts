#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Receiving FitBit data in JSON format and infering Fatigue Assessment Score
using machine learning model.

Created:
    2022-05-07

"""
import json
from datetime import datetime, timedelta

import joblib
import pandas as pd
import numpy as np
import tflite_runtime.interpreter as tflite

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

class FitBitDataFrame:
    """Producing a coherent data frame based on Fitbit data.

    The resulting data frame will contain data collected using a Fitbit
    wearable by one person.

    Attributes:
        user_id (str): ID of user. Used for error messages.
        dfs (list of DataFrames): List containing pandas DataFrames for each of
            the different categories of data.
        df (DataFrame): The resulting data frame after joining together the data
            from all data frames in `dfs`.
        age (int): Age of person.
        gender (int): Gender of person.
        weight (float): Weight of person.
        height (float): Height of person.
        bmi (float): Body Mass Index of person.

    """

    def __init__(self, user_id):

        self.user_id = user_id

        self.dfs = []
        self.df = None

        self.age = None
        self.gender = None
        self.weight = None
        self.height = None
        self.bmi = None


    def _read_profile(self, data_dict):
        """Read the profile information about the person.

        Args:
            data_dict (dict): Profile information from the Fitbit API.

        """

        self.age = datetime.now().year - int(data_dict["dateOfBirth"][:4])
        self.gender = 0 if data_dict["gender"][0] == "FEMALE" else 1
        self.weight = data_dict["weight"]
        self.height = data_dict["height"]
        self.bmi = self.weight / ((self.height / 100) ** 2)

    def read_sleep(self, data_dict):
        """Read sleep data.

        Args:
            data_dict (dict): Sleep information from the Fitbit API.

        """

        # df = pd.read_json(data_dict)
        df = pd.DataFrame.from_dict(data_dict)

        try:
            df["dateOfSleep"] = pd.to_datetime(df["dateOfSleep"])
        except KeyError as e:
            logging.warning(f"User ID: {self.user_id}. Error retreving data: sleep. Error: " + repr(e))

        df.set_index("dateOfSleep", inplace=True)

        # Delete rows which does not contain main sleep
        df = df[df.isMainSleep == True]

        levels = pd.json_normalize(df["levels"]).add_prefix("levels.")
        levels.index = df.index
        df = df.join(levels)

        self.dfs.append(df)

    def read_timeseries(self, name, data_dict, sum_values=False):
        """Read time series data.

        Args:
            name (str): What name to give the variable.
            data_dict (dict): Sleep information from the Fitbit API.
            sum_values (bool): Whether to sum/aggregate values for each day. If
                this is False, the max, min and average values will be computed
                for each day instead.

        """

        df = pd.DataFrame.from_dict(data_dict)

        # Change dateTime column to datetime type
        try:
            df["dateTime"] = pd.to_datetime(df["dateTime"])
        except KeyError as e:
            logging.warning(f"User ID: {self.user_id}. Error retreving data: {name}. Error: " + repr(e))


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
                # df_resampled[column] = df.resample("D", on="dateTime")[column].sum()
                df_resampled[column] = df.resample("D", on="dateTime")[column].first()
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
        """Read heart rate data.

        Args:
            data_dict (dict): Sleep information from the Fitbit API.

        """

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

            intraday = obj["activities-heart-intraday"]["dataset"]
            intraday = pd.DataFrame.from_dict(intraday)
            
            # Check if data is present in intraday. If not, continue to next
            # day.
            if (
                    "time" not in intraday.keys() and 
                    "value" not in intraday.keys()
                ):
                continue

            intraday.set_index("time", inplace=True)

            heart_rate_bpm_max = intraday["value"].max()
            heart_rate_bpm_min = intraday["value"].min()
            heart_rate_bpm_mean = intraday["value"].mean()

            # resting_heart_rate is sometimes missing from the API. If it is,
            # use the heart_rate_bpm_min instead.
            try:
                resting_heart_rate = obj["activities-heart"][0]["value"]["restingHeartRate"]
            except KeyError as e:
                logging.warning(f"User ID: {self.user_id}. Resting heart rate is missing, replacing with the minimum value. Error: " + repr(e));
                resting_heart_rate = heart_rate_bpm_min

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

    def combine_data_and_profile(self, profile_data_dict):
        """Combine all data frames together with the profile information.

        Args:
            profile_data_dict (dict): Profile information.

        Returns:
            df (DataFrame): The final data frame containing all data.

        """

        self.df = self.dfs[0]

        for df in self.dfs[1:]:
            self.df = self.df.join(df)

        # Add user profile information to each line of the data frame
        self._read_profile(profile_data_dict)
        self.df["age"] = self.age
        self.df["gender"] = self.gender
        self.df["weight"] = self.weight
        self.df["height"] = self.height
        self.df["bmi"] = self.bmi

        return self.df

def infer_tflite(input_data, model_filepath):
    """Run inference on input data.

    Args:
        input_data (array): Input data/predictors.
        model_filepath: Filepath to model.

    Returns:
        prediction (array): Predicted values from model.

    """

    # Load model
    interpreter = tflite.Interpreter(model_filepath)

    interpreter.allocate_tensors()

    input_tensor = np.array(input_data, dtype=np.float32)

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    interpreter.set_tensor(input_index, input_tensor)


    # Infer
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_index)

    return prediction

def format_output(user_id, fas, timestamp):
    """Return the output in expected format."""
    
    return {"userid": user_id, "fas": fas, "timestamp": timestamp}


def read_user_data(user_data):
    """Extract all the required fields from input data.

    Args:
        user_data: Input data.        

    Returns:
        fitbit_data: Only contains relevant fields.

    """

    fitbit_data = FitBitDataFrame(user_id=user_data["userid"])

    fitbit_data.read_timeseries("calories", user_data["activities-calories"])
    fitbit_data.read_timeseries("distance", user_data["activities-distance"])
    fitbit_data.read_timeseries("steps", user_data["activities-steps"], sum_values=True)
    fitbit_data.read_timeseries(
        "lightly_active_minutes",
        user_data["activities-minutesLightlyActive"],
        sum_values=True,
        )
    fitbit_data.read_timeseries(
        "moderately_active_minutes",
        user_data["activities-minutesFairlyActive"],
        sum_values=True,
        )
    fitbit_data.read_timeseries(
        "very_active_minutes",
        user_data["activities-minutesVeryActive"],
        sum_values=True,
        )
    fitbit_data.read_timeseries(
        "sedentary_minutes",
        user_data["activities-minutesSedentary"],
        sum_values=True,
        )
    fitbit_data.read_heart_rate(user_data["heartrate"])
    fitbit_data.read_sleep(user_data["sleep"])

    return fitbit_data

def preprocess_and_infer(
    input_json_str,
    scaler_filepath,
    model_filepath,
    input_columns_filepath,
    params_filepath,
):
    """Preprocess data and pass it to inference.

    Args:
        input_json_str (str): JSON string containing input data.
        scaler_filepath (str): Filepath to scaler object.
        model_filepath (str): Filepath to model.
        input_columns_filepath (str): Filepath to list of input columns.
        params_filepath (str): Filepath to configuration paramters.

    Returns:
        output_json (str): Results from inference as JSON string.

    """

    # Read configuration parameters
    with open(params_filepath, "r") as infile:
        params = json.load(infile)
        window_size = params["window_size"]
        max_data_staleness = params["max_data_staleness"]

    input_json = json.loads(input_json_str)

    output = []

    for user_data in input_json:
        user_id = user_data["userid"]

        try:
            fitbit_data = read_user_data(user_data)
        except KeyError as e:
            logging.warning(f"User ID: {user_id}. Skipping current user due to missing values. Error: " + repr(e))
            output.append(format_output(user_id=user_id, fas="nan", timestamp="nan"))

            # Continue to next user
            continue

        fitbit_data.combine_data_and_profile(user_data["user"])

        # Select variables/columns to use
        input_columns = pd.read_csv(
            input_columns_filepath, index_col=0, header=None
        ).index.tolist()
        input_data = fitbit_data.df[input_columns]

        # Drop NaN values
        input_data = input_data.dropna()

        # Sort input data on date
        input_data = input_data.sort_index()

        # Select the last n data points, where n=window_size
        input_data = input_data.iloc[-window_size:,:]

        # If input data has less than required number of days (window size) in
        # the latest 10 days (calculated from the latest date in the input
        # data), return NaN.
        dates_in_input_data = input_data.index.to_pydatetime()
        latest_date = max(dates_in_input_data)
        date_threshold = latest_date - timedelta(days = max_data_staleness)

        if (
                len(input_data) < window_size or
                dates_in_input_data.any() < date_threshold
            ):
            logging.warning(f"User ID: {user_id}. Skipping user due to not enough available days of data.")
            output.append(format_output(user_id=user_id, fas="nan", timestamp="nan"))

            # Continue to next user
            continue

        # Convert to NumPy array
        input_data = input_data.to_numpy()

        # Load scaler
        scaler = joblib.load(scaler_filepath)

        # Scale input data
        input_data = scaler.transform(input_data)

        # Select the latest data n data points, where n=window_size
        input_data = input_data[-window_size:, :].reshape(1, -1)

        y = infer_tflite(input_data, model_filepath)

        # The latest FAS value is returned for each user.
        output.append(format_output(
            user_id, 
            fas=str(y[-1]),
            timestamp=str(latest_date)
        ))

    output_json = json.dumps(output)

    return output_json
