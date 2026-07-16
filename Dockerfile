FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
COPY pyproject.toml .

RUN pip install --upgrade pip

# Install CPU-only PyTorch
RUN pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision torchaudio

# Install remaining dependencies (with dependency resolution)
RUN pip install -r requirements.txt

COPY . .

RUN pip install -e .

EXPOSE 8000
EXPOSE 8501

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]