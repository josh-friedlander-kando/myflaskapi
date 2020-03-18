FROM python:3.7  

COPY ./requirements.txt /tmp/ 
RUN pip install --upgrade pip 
RUN pip install --requirement /tmp/requirements.txt
COPY ./app.py /code/
CMD ["python", "/code/app.py"]
