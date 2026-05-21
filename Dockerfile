FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libstdc++6 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY agentwatch/requirements.txt /app/agentwatch/requirements.txt

RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /app/agentwatch/requirements.txt
RUN pip install --no-cache-dir greenlet==3.0.3

COPY . /app

ENV PYTHONPATH=/app
ENV PORT=8000

CMD ["sh", "-c", "uvicorn agentwatch.api.main:app --host 0.0.0.0 --port ${PORT}"]
