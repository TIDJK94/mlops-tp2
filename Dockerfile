FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY webservice.py .

EXPOSE 8000

CMD ["python", "webservice.py"]