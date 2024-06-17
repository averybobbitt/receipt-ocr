# pull the official docker image
FROM python:3.12-slim

# install tesseract and its dependencies
RUN apt-get update && apt-get install -y tesseract-ocr libtesseract-dev

# set work directory
WORKDIR /app

# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# copy project
COPY src/ .

# uvicorn entrypoint
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]