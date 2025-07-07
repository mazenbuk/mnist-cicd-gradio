# MNIST Digit Classification - MLOps Project

A complete Machine Learning Operations (MLOps) project for handwritten digit classification using the MNIST dataset. This project demonstrates the full ML lifecycle from model training to production deployment with containerization and automated pipelines.

## 🎯 Project Overview

This project implements a **Convolutional Neural Network (CNN)** to classify handwritten digits (0-9) from the MNIST dataset. It includes:

- **Model Training**: PyTorch-based CNN implementation
- **Web Application**: Interactive Gradio interface for real-time predictions
- **MLOps Pipeline**: Automated training, evaluation, and deployment
- **Containerization**: Docker support with multi-stage builds
- **Cloud Deployment**: Integration with Hugging Face Spaces

## 🏗️ Project Structure

```
nmist/
├── train.py                 # Model training script
├── notebook.ipynb          # Jupyter notebook for experimentation
├── requirements.txt        # Training dependencies
├── Dockerfile             # Multi-stage Docker build
├── Makefile              # Automation commands
├── App/                  # Web application
│   ├── app.py           # Gradio web interface
│   ├── requirements.txt # App dependencies
│   └── README.md       # HuggingFace Space config
├── Data/                # MNIST dataset
│   └── MNIST/raw/      # Raw dataset files
└── Model/              # Trained model files
    └── mnist_cnn.pth  # Saved model weights
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip
- Docker (optional, for containerized deployment)

### 1. Install Dependencies

```bash
make install
# or
pip install -r requirements.txt
```

### 2. Train the Model

```bash
make train
# or
python train.py
```

### 3. Run the Web Application

```bash
cd App
pip install -r requirements.txt
python app.py
```

The application will be available at `http://localhost:7860`

## 🔧 Model Architecture

The project uses a **SimpleCNN** architecture:

- **Input**: 28x28 grayscale images
- **Conv Layer 1**: 32 filters, 3x3 kernel, ReLU activation
- **Conv Layer 2**: 64 filters, 3x3 kernel, ReLU activation
- **MaxPooling**: 2x2 pooling after each conv layer
- **Fully Connected**: 128 neurons, ReLU activation
- **Output**: 10 classes (digits 0-9)

## 📱 Web Application Features

- **Interactive Drawing Canvas**: Draw digits directly in the browser
- **Real-time Prediction**: Instant classification results
- **Probability Scores**: View confidence scores for all digit classes
- **Image Preprocessing**: Automatic normalization and resizing

## 🐳 Docker Deployment

### Build and Run

```bash
docker build -t mnist-app .
docker run -p 7860:7860 mnist-app
```

### Multi-stage Build Process

1. **Builder Stage**: Trains the model and generates `mnist_cnn.pth`
2. **App Stage**: Creates lightweight production image with only the application

## 🛠️ Available Commands (Makefile)

| Command | Description |
|---------|-------------|
| `make install` | Install Python dependencies |
| `make format` | Format code with Black |
| `make train` | Train the model |
| `make eval` | Generate evaluation report |
| `make hf-login` | Login to Hugging Face |
| `make push-hub` | Upload to Hugging Face Spaces |
| `make deploy` | Full deployment pipeline |

## 📊 Model Performance

- **Training Duration**: 3 epochs
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 64

## 🌐 Cloud Deployment

This project is configured for deployment on **Hugging Face Spaces**:

```bash
make deploy
```

This will:
1. Login to Hugging Face
2. Upload application files
3. Upload trained model
4. Deploy to your HF Space

## 📁 Key Files

- **`train.py`**: Main training script with CNN implementation
- **`App/app.py`**: Gradio web application
- **`Dockerfile`**: Multi-stage container build
- **`Makefile`**: Automation and deployment commands
- **`notebook.ipynb`**: Jupyter notebook for experimentation

## 🔄 MLOps Workflow

1. **Development**: Code in Jupyter notebook or Python scripts
2. **Training**: Automated model training with `make train`
3. **Evaluation**: Generate metrics and reports
4. **Containerization**: Docker builds for reproducible deployment
5. **Deployment**: Automated push to Hugging Face Spaces
6. **Monitoring**: Track performance and model drift

## 🛡️ Requirements

### Training Environment
```
torch
torchvision
numpy
black
```

### Application Environment
```
gradio
torch
torchvision
numpy
pillow
```

## 📝 Usage Examples

### Training a Model
```python
python train.py
```

### Making Predictions
```python
from train import SimpleCNN
import torch

model = SimpleCNN()
model.load_state_dict(torch.load('Model/mnist_cnn.pth'))
model.eval()

# Your prediction code here
```

### Running with Docker
```bash
docker build -t mnist-classifier .
docker run -p 7860:7860 mnist-classifier
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run `make format` to format code
5. Submit a pull request

## 📄 License

This project is licensed under the Apache 2.0 License.

## 🔗 Links

- **Hugging Face Space**: [nmist](https://huggingface.co/spaces/mazenbuk/nmist)
- **Dataset**: [MNIST](http://yann.lecun.com/exdb/mnist/)
- **Framework**: [PyTorch](https://pytorch.org/)
- **Interface**: [Gradio](https://gradio.app/)

## 📚 Learning Objectives

This project demonstrates:
- **Deep Learning**: CNN implementation for image classification
- **MLOps**: End-to-end ML pipeline automation
- **Web Development**: Interactive ML applications
- **DevOps**: Containerization and deployment strategies
- **Cloud Computing**: Platform-as-a-Service deployment

Perfect for learning modern MLOps practices and building production-ready ML applications!