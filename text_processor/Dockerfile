# # FROM tiangolo/uvicorn-gunicorn:python3.6-alpine3.8
FROM python:3.7

RUN mkdir -p /app/processor
WORKDIR /app/processor

COPY requirements.txt /app/processor

RUN pip3 install -U pip
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /app/processor

EXPOSE 8080

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "5000", "app.main:app"]


