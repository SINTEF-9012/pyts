FROM python:3.7
ADD requirements.txt /
RUN python3.7 -m pip install -r requirements.txt
ADD consumer.py /
ENTRYPOINT [ "python", "./consumer.py" ]