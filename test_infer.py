#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test inference.

Created:
    2022-05-19

"""
from infer_fas_tflite import preprocess_and_infer

if __name__ == "__main__":

    with open("data/sample_input.json", "r", encoding="UTF-8") as f:
        input_json_str = f.read()

    output_json = preprocess_and_infer(
        input_json_str,
        "model/input_scaler.z",
        "model/model.tflite",
        "data/input_columns.csv",
        "data/params.json",
    )

    print(output_json)
