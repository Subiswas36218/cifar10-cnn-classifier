# 🧠 CIFAR-10 CNN Image Classifier

> Production-ready image classification system built with **PyTorch**, **FastAPI**, **Streamlit**, and **Docker**.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.116-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.48-red)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Features

- CNN image classifier
- CIFAR-10 dataset
- Training pipeline
- Evaluation pipeline
- CLI prediction
- REST API
- Streamlit UI
- Docker support
- PyTest test suite
- GitHub Actions CI

---

## Repository Structure

```text
src/
models/
tests/
scripts/
examples/
models/
data/
├── README.md
├── Dockerfile
├── requirements.txt
├── pyproject.toml
└── docker-compose.yml
```

---

## Installation

```bash
git clone https://github.com/Subiswas36218/cifar10-cnn-classifier

cd cifar10-cnn-classifier

python -m venv venv

source venv/bin/activate
```

Windows

```powershell
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -e .

pip install -r requirements.txt
```

---

## Training

```bash
python -m src.train
```

---

## Evaluate

```bash
python -m src.evaluate
```

---

## Predict

```bash
python -m src.predict examples/airplane.jpg
```

---

## FastAPI

```bash
uvicorn src.api:app --reload
```
Open:
http://localhost:8000/docs

---

## Streamlit

```bash
streamlit run src/app.py
```
Open:
http://localhost:8501/

---

## Docker

```bash
docker compose build

docker compose up
```

---

## Testing

pytest

## License

MIT LICENSE
