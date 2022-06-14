#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test inference.

Created:
    2022-05-19

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

    url = f"https://api.fitbit.com/1.2/user/-/sleep/date/2020-01-01/2020-01-05.json"

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

    with open("data/sample_input.json", "r") as f:
        input_json_str = f.read()

    output_json = preprocess_and_infer(
        input_json_str, "model/input_scaler.z", "model/model.h5", "data/input_columns.csv"
    )

    print(output_json)
