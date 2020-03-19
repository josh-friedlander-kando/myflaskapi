FROM kandoenv/training

COPY requirements.txt /tmp/ 
RUN pip install --upgrade pip 
RUN pip install --requirement /tmp/requirements.txt
COPY app.py /code/
COPY ml_models/ /code/
CMD ["python", "/code/app.py"]
