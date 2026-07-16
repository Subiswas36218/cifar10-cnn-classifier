# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-07-14

### Added

#### Deep Learning
- Implemented a modular Convolutional Neural Network (CNN) for CIFAR-10 image classification.
- Added reusable convolution blocks and classifier head.
- Implemented configurable weight initialization.
- Added model parameter counting utilities.

#### Dataset
- Integrated CIFAR-10 dataset using torchvision.
- Added preprocessing and normalization pipeline.
- Added configurable data augmentation.
- Added DataLoader abstraction.

#### Training
- Implemented end-to-end training pipeline.
- Added validation loop.
- Added checkpoint saving.
- Added best model selection.
- Added training history tracking.
- Added progress bars using tqdm.

#### Evaluation
- Added model evaluation script.
- Added classification report.
- Added confusion matrix generation.
- Added accuracy and loss reporting.

#### Inference
- Added shared inference engine.
- Added image preprocessing pipeline.
- Added Top-K predictions.
- Added confidence scores.
- Added CLI prediction tool.

#### FastAPI
- Added REST API.
- Added health endpoint.
- Added prediction endpoint.
- Added class listing endpoint.
- Added automatic Swagger documentation.

#### Streamlit
- Added interactive web interface.
- Added image upload support.
- Added prediction visualization.
- Added Top-K probability display.

#### Testing
- Added dataset tests.
- Added model tests.
- Added training tests.
- Added API tests.
- Added inference smoke tests.

#### DevOps
- Added Docker support.
- Added Docker Compose configuration.
- Added GitHub Actions CI workflow.
- Added Makefile.
- Added project packaging using pyproject.toml.

#### Documentation
- Added README.
- Added LICENSE.
- Added CONTRIBUTING guide.
- Added CHANGELOG.
- Added project architecture documentation.

---

## Planned

### Version 1.1.0

#### Features
- Grad-CAM visualization
- ONNX export
- TorchScript export
- TensorBoard integration
- MLflow experiment tracking
- Model quantization
- Batch prediction
- Webcam inference
- Image drag-and-drop support
- Model benchmarking

#### Deployment
- Kubernetes deployment
- Hugging Face Spaces deployment
- AWS deployment
- GPU inference support

#### MLOps
- Automated model versioning
- Experiment tracking
- Dataset versioning
- Continuous deployment

---

## License

This project is licensed under the MIT License.