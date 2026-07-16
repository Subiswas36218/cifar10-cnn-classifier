"""
Tests for the FastAPI application.
"""

from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

import src.api as api_module


class MockInferenceEngine:
    """
    Lightweight inference engine used for testing.
    """

    def __init__(self) -> None:
        self.class_names = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

    def predict(self, image: Image.Image) -> dict:
        return {
            "prediction": "cat",
            "confidence": 0.9876,
            "top_k": [
                {"class": "cat", "confidence": 0.9876},
                {"class": "dog", "confidence": 0.0087},
                {"class": "bird", "confidence": 0.0020},
                {"class": "deer", "confidence": 0.0010},
                {"class": "frog", "confidence": 0.0007},
            ],
        }


# Replace the real inference engine with a mock
api_module.engine = MockInferenceEngine()

client = TestClient(api_module.app)


def create_test_image() -> BytesIO:
    """
    Create a simple in-memory RGB image.
    """
    image = Image.new("RGB", (32, 32), color="blue")

    buffer = BytesIO()

    image.save(buffer, format="PNG")

    buffer.seek(0)

    return buffer


def test_root() -> None:
    response = client.get("/")

    assert response.status_code == 200

    payload = response.json()

    assert payload["version"] == "1.0.0"


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == "healthy"


def test_classes() -> None:
    response = client.get("/classes")

    assert response.status_code == 200

    classes = response.json()["classes"]

    assert len(classes) == 10

    assert "cat" in classes


def test_predict() -> None:
    image = create_test_image()

    response = client.post(
        "/predict",
        files={
            "file": (
                "image.png",
                image,
                "image/png",
            )
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["prediction"] == "cat"

    assert "top_k" in payload

    assert len(payload["top_k"]) == 5


def test_invalid_file() -> None:
    response = client.post(
        "/predict",
        files={
            "file": (
                "test.txt",
                b"hello",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
