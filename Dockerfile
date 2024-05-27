FROM python:3.12-alpine

RUN apt-get update -y && apt-get install awscli -y
WORKDIR /app

COPY . /app/
RUN pip install -r requirements.txt

CMD ["streamlit", "run", "app.py"]