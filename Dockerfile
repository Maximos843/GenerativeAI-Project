FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1

RUN mkdir /app

COPY ./requirements.txt /app/

WORKDIR /app

RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD [ "python3", "/app/main.py"]