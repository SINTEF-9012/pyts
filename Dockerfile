FROM python:3.7-slim
COPY *.py /
COPY data/input_columns.csv data/params.json data/
COPY model model
COPY requirements.txt /
RUN python3.7 -m pip install -r requirements.txt
ENTRYPOINT [ "python", "./consumer.py" ]