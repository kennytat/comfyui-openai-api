FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Runs as root on purpose (plan D5): ComfyUI writes root-owned files into the
# mounted output subfolder and the gateway must be able to delete them.
WORKDIR /srv
COPY pyproject.toml ./
COPY app ./app
RUN pip install .
COPY workflows ./workflows

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
