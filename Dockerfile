FROM python:3.7
ADD requirements.txt /
RUN python3.7 -m pip install -r requirements.txt
ADD *.py /
ADD data/input_columns.csv data/input_columns.csv
ADD model model
ENTRYPOINT [ "python", "./consumer.py" ]