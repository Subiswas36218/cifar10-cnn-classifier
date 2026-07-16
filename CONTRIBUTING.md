# Contributing Guide

Thank you for your interest in contributing to the **CIFAR-10 CNN Image Classifier** project.

Contributions of all kinds are welcome, including bug fixes, documentation improvements, new features, and performance optimizations.

---

# Development Setup

## 1. Fork the Repository

Click **Fork** on GitHub.

Clone your fork:

```bash
git clone https://github.com/<your-username>/cifar10-cnn-classifier.git

cd cifar10-cnn-classifier
```

---

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

Linux/macOS

```bash
source venv/bin/activate
```

Windows

```powershell
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -e .

pip install -r requirements.txt
```

---

# Development Workflow

Create a feature branch.

```bash
git checkout -b feature/my-new-feature
```

Make your changes.

---

# Code Style

This project follows:

- PEP 8
- Black formatting
- Ruff linting
- MyPy type checking

Before committing, run:

```bash
black .

ruff check .

mypy src
```

or simply

```bash
make check
```

---

# Running Tests

Run the complete test suite.

```bash
pytest
```

Run a single test file.

```bash
pytest tests/test_api.py
```

Run with verbose output.

```bash
pytest -v
```

---

# Training the Model

```bash
python -m src.train
```

---

# Evaluating the Model

```bash
python -m src.evaluate
```

---

# Running the API

```bash
uvicorn src.api:app --reload
```

API documentation:

```
http://localhost:8000/docs
```

---

# Running Streamlit

```bash
streamlit run src/app.py
```

---

# Commit Messages

Please use clear commit messages.

Examples:

```
feat: add Grad-CAM visualization

fix: correct checkpoint loading

docs: improve README

refactor: simplify inference engine

test: add API endpoint tests
```

---

# Pull Requests

Please ensure that your Pull Request:

- includes a clear description
- references related issues when applicable
- passes all tests
- follows the project's coding style
- includes documentation updates for user-facing changes

---

# Reporting Bugs

When reporting a bug, please include:

- Operating System
- Python version
- PyTorch version
- Error message
- Steps to reproduce
- Expected behavior
- Actual behavior

---

# Feature Requests

Feature requests are welcome.

Please describe:

- the problem
- your proposed solution
- expected benefits
- possible alternatives

---

# Project Structure

```
src/
    models/
    train.py
    evaluate.py
    predict.py
    inference.py
    api.py
    app.py

tests/

scripts/

examples/

models/
```

---

# Code of Conduct

Please be respectful and constructive.

We aim to maintain a welcoming and inclusive environment for everyone.

---

# Questions

If you have questions or suggestions, feel free to:

- Open an Issue
- Start a Discussion
- Submit a Pull Request

---

Thank you for contributing to the project!