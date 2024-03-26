FROM python:3.10-slim-buster
RUN pip install --no-cache-dir flask catboost pandas scikit-learn

COPY app.py /app/app.py
COPY trained_model.cbm /app/trained_model.cbm

WORKDIR /app
EXPOSE 5000
ENTRYPOINT ["python"]
CMD ["app.py"]
