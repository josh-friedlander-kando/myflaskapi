FROM kandoenv/training

COPY requirements.txt /tmp/
RUN pip install --upgrade pip
RUN pip install --requirement /tmp/requirements.txt
COPY inference_server.py /code/
COPY training /code/trainer
CMD ["python", "/code/inference_server.py"]