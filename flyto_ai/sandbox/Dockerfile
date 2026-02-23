FROM python:3.12-slim

RUN pip install --no-cache-dir flyto-core[browser]

COPY runner.py /app/runner.py
WORKDIR /app

ENTRYPOINT ["python", "runner.py"]
