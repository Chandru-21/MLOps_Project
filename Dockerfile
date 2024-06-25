FROM python:3.10-slim-buster

RUN pip install --upgrade pip

WORKDIR /app

COPY . /app 

#set permissions

RUN chmod +x /app/tests

RUN chmod +w /app/tests

RUN chmod +x /app/prediction_model

RUN chmod +w /app/prediction_model/trained_models

RUN chmod +w /app/prediction_model/datasets


ENV PYTHONPATH "${PYTHONPATH}:/app/prediction_model"


RUN pip install --no-cache-dir -r requirements.txt

RUN pip install dvc[s3]

RUN dvc pull

RUN python /app/prediction_model/training_pipeline.py

RUN pytest -v /app/tests/test_prediction.py

RUN pytest --junitxml=/app/tests/test-results.xml /app/tests/test_prediction.py

EXPOSE 8005

ENTRYPOINT ["python"]

CMD ["main.py"]


