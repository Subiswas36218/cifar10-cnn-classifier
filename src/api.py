"""
FastAPI service for the CIFAR-10 CNN Classifier.
"""

from __future__ import annotations

from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from src.inference import InferenceEngine

app = FastAPI(
    title="CIFAR-10 CNN Classifier",
    description="Image classification API using PyTorch.",
    version="1.0.0",
)

engine = InferenceEngine()


@app.get("/")
def root() -> dict:
    """
    Root endpoint.
    """
    return {
        "message": "CIFAR-10 CNN Classifier API",
        "version": "1.0.0",
    }


@app.get("/health")
def health() -> dict:
    """
    Health check.
    """
    return {
        "status": "healthy",
    }


@app.get("/classes")
def classes() -> dict:
    """
    Return supported classes.
    """
    return {
        "classes": engine.class_names,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
) -> dict:
    """
    Predict the class of an uploaded image.
    """

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not an image.",
        )

    try:
        contents = await file.read()

        image = Image.open(BytesIO(contents)).convert("RGB")

        result = engine.predict(image)

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
