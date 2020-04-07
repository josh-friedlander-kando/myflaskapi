FROM python:3.7

COPY requirements.txt /tmp/
RUN pip install --upgrade pip
RUN pip install --requirement /tmp/requirements.txt
RUN pip install fbprophet
RUN pip install xgboost
RUN pip install -v git+https://6069a4cc332196578e339665fa63fcb2ff67e1c8@github.com/nadavk72/kando-python-client.git
COPY . /code
CMD ["python", "code/trainer.py"]
