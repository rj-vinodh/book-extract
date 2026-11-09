FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .

# Run as a non-root user; outputs/ is where captured frames are written
RUN useradd -m appuser && mkdir -p /app/outputs && chown -R appuser /app
USER appuser

EXPOSE 7860
CMD ["python", "app.py"]
