FROM python:3.12

WORKDIR /app

# Install ffmpeg
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Download and load the model in a separate RUN step
RUN python -c "from model_loader import load_model; load_model()"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]