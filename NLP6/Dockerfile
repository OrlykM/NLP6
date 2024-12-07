FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
COPY src /app/src
COPY data /app/data
COPY main.py .

RUN apt-get update && apt-get install -y \
    gcc \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install -r requirements.txt

RUN python -m nltk.downloader punkt wordnet omw-1.4 averaged_perceptron_tagger_eng

EXPOSE 7860

CMD ["python", "main.py"]