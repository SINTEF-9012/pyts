FROM python:3.7
ADD requirements.txt /
RUN pip install -r requirements.txt
ADD consumer.py /
CMD [ "python", "./consumer.py" ]