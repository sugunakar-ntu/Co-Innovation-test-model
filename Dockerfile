From python:3.7

ARG TRAIN_DATA

WORKDIR /usr/src/app

COPY requirements.txt svmClassification.py train.csv ./

RUN pip install --no-cache-dir -r requirements.txt

RUN python svmClassification.py $TRAIN_DATA
