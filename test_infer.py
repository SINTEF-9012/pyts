#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test inference.

Author:
    Erik Johannes Husom

Created:
    2022-05-19 torsdag 15:13:38 

"""
import json

import requests

from infer_fas import *


def get_activity_data(variable, detail_level="1min"):

    url = (
        f"https://api.fitbit.com/1/user/-/activities/{variable}/date/2019-01-01/7d.json"
    )

    headers = {
        "authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyMzg1QzIiLCJzdWIiOiI5WDJKNVMiLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJ3aHIgd3BybyB3bnV0IHdzbGUgd3dlaSB3c29jIHdzZXQgd2FjdCB3bG9jIiwiZXhwIjoxNjUzNDMwMjk1LCJpYXQiOjE2NTM0MDE0OTV9.wPUywSYdwyzaWS_5MhKwIhhgWh8RfN8aNcH9RGkmGqg"
    }

    json_response = requests.get(url, headers=headers).json()    
    json_response = json_response[f"activities-{variable}"]
    json_response = json.dumps(json_response)
  
    return json_response


def get_heart_rate_data(detail_level="1min"):

    url = f"https://api.fitbit.com/1/user/-/activities/heart/date/2019-01-01/7d/{detail_level}.json"

    headers = {
        "authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyMzg1QzIiLCJzdWIiOiI5WDJKNVMiLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJ3aHIgd3BybyB3bnV0IHdzbGUgd3dlaSB3c29jIHdzZXQgd2FjdCB3bG9jIiwiZXhwIjoxNjUzNDMwMjk1LCJpYXQiOjE2NTM0MDE0OTV9.wPUywSYdwyzaWS_5MhKwIhhgWh8RfN8aNcH9RGkmGqg"
    }

    json_response = requests.get(url, headers=headers).json()
    json_response = json_response[f"activities-heart-intraday"]["dataset"]

    json_response = json.dumps(json_response)
    json_response = json_response.replace("time", "dateTime")

    return json_response

def get_sleep():

    url = (
        f"https://api.fitbit.com/1.2/user/-/sleep/date/2020-01-01/2020-01-05.json"
    )

    headers = {
        "authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyMzg1QzIiLCJzdWIiOiI5WDJKNVMiLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJ3aHIgd3BybyB3bnV0IHdzbGUgd3dlaSB3c29jIHdzZXQgd2FjdCB3bG9jIiwiZXhwIjoxNjUzNDMwMjk1LCJpYXQiOjE2NTM0MDE0OTV9.wPUywSYdwyzaWS_5MhKwIhhgWh8RfN8aNcH9RGkmGqg"
    }

    json_response = requests.get(url, headers=headers).json()
    json_response = json_response["sleep"]
    json_response = json.dumps(json_response)

    return json_response



def get_profile():

    url = f"https://api.fitbit.com/1/user/-/profile.json"

    headers = {
        "authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyMzg1QzIiLCJzdWIiOiI5WDJKNVMiLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJ3aHIgd3BybyB3bnV0IHdzbGUgd3dlaSB3c29jIHdzZXQgd2FjdCB3bG9jIiwiZXhwIjoxNjUzNDMwMjk1LCJpYXQiOjE2NTM0MDE0OTV9.wPUywSYdwyzaWS_5MhKwIhhgWh8RfN8aNcH9RGkmGqg"
    }

    json_response = requests.get(url, headers=headers).json()    
    return json_response


if __name__ == "__main__":

    f = FitBitDataFrame()

    f.read_timeseries("calories", get_activity_data("calories"))
    f.read_timeseries("distance", get_activity_data("distance"))
    f.read_timeseries("steps", get_activity_data("steps"), sum_values=True)
    f.read_timeseries(
        "minutesLightlyActive",
        get_activity_data("minutesLightlyActive"),
        sum_values=True,
    )
    f.read_timeseries(
        "minutesFairlyActive", get_activity_data("minutesFairlyActive"), sum_values=True
    )
    f.read_timeseries(
        "minutesVeryActive", get_activity_data("minutesVeryActive"), sum_values=True
    )
    f.read_timeseries(
        "minutesSedentary", get_activity_data("minutesSedentary"), sum_values=True
    )

    # f.read_sleep(get_sleep())
    # f.read_timeseries("heart_rate", get_heart_rate_data("heart_rate"))

    f.combine_data_and_profile(get_profile())

    # Load name of input columns
    input_columns = pd.read_csv("data/input_columns.csv",
            index_col=0, header=None).index.tolist()

    print(repr(f))

    infer(f.df, "model/input_scaler.z", "model/model.h5", input_columns)
