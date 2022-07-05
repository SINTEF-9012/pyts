#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test inference.

Created:
    2022-05-19

"""
import sys

from infer_fas_tflite import preprocess_and_infer

if __name__ == "__main__":

    if len(sys.argv) > 1:
        input_json = sys.argv[1]
    else:
        input_json = "data/empty_fields.json"

    with open(input_json, "r", encoding="UTF-8") as f:
        input_json_str = f.read()

    output_json = preprocess_and_infer(
        input_json_str,
        "model/input_scaler.z",
        "model/model.tflite",
        "data/input_columns.csv",
        "data/params.json",
    )

    print(output_json)
