FROM python:3.9

ARG TRAIN_DATA

RUN pip install --upgrade pip

COPY requirements.txt runmodel.py test.py savemodel.py model.sav train.csv ./

RUN pip install --no-cache-dir -r requirements.txt

RUN python test.py $TRAIN_DATA

RUN python savemodel.py $TRAIN_DATA

#RUN  pip uninstall pystan;pip install pystan==2.18;pip uninstall holidays;pip install holidays==0.9.12

# command to execute when image loads
ENTRYPOINT ["python","runmodel.py"]
CMD ["forecast"]
#CMD [ "python", "runmodel.py"]