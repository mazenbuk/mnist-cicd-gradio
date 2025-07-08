# MNIST Digit Classification - MLOps Project

A containerized MLOps project for handwritten digit classification using CNN and the MNIST dataset. Features automated training, interactive web interface, and Docker-based deployment.

## 👥 Team Members

- **225150200111004** - **Haikal Thoriq Athaya**
- **225150200111008** - **Muhammad Arsya Zain Yashifa**  
- **225150201111001** - **Javier Aahmes Reansyah**
- **225150201111003** - **Muhammad Herdi Adam**

## 🎯 Project Overview

This project implements a **Convolutional Neural Network (CNN)** to classify handwritten digits (0-9) from the MNIST dataset with:

- **Model Training**: PyTorch-based SimpleCNN with automated training pipeline
- **Web Application**: Interactive Gradio interface for real-time digit recognition  
- **Containerization**: Docker support with optimized multi-stage builds
- **Cloud Deployment**: Integration with Hugging Face Spaces

## 🏗️ Project Structure

```
nmist/
├── Dockerfile              # Multi-stage Docker build
├── Makefile                # Automation commands
├── requirements.txt        # Training dependencies
├── .gitignore              # Git ignore file
├── App/                    # Web application
│   ├── app.py              # Gradio interface
│   ├── requirements.txt    # App dependencies
│   └── README.md           # HuggingFace Space config
├── Scripts/                # Training and evaluation scripts
│   ├── train.py            # Model training script
│   └── eval.py             # Model evaluation script
├── Notebooks/              # Jupyter notebooks
│   └── notebook.ipynb      # Experimentation notebook
├── Data/                   # Dataset storage
│   └── MNIST/              # MNIST dataset (auto-downloaded)
├── Model/                  # Trained model storage
│   └── mnist_cnn.pt        # Saved model weights
└── Results/                # Training results (created by make train)
    ├── metrics.txt         # Performance metrics
    └── results.png         # Evaluation plots
```

## 🚀 Quick Start with Docker (Recommended)

### Prerequisites
- **Docker** (Docker Desktop recommended)
- **Git** for cloning the repository

### 1. Clone and Build
```bash
git clone https://github.com/mazenbuk/mnist-cicd-gradio.git
cd nmist

# Build Docker image (includes training and app)
docker build -t mnist-app .
```

### 2. Run Application
```bash
# Run container
docker run -p 7860:7860 mnist-app

# Or run in background
docker run -d --name mnist-container -p 7860:7860 mnist-app
```

The application will be available at `http://localhost:7860`

### 3. Docker Management
```bash
# View logs
docker logs mnist-container

# Stop container
docker stop mnist-container

# Remove container
docker rm mnist-container
```

## 🔧 Alternative: Manual Setup

If you prefer to run without Docker:

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Model
```bash
make train
# or manually: python Scripts/train.py
```

### 3. Run Web App
```bash
cd App
pip install -r requirements.txt
python app.py
```

## 🐳 Docker Architecture

The project uses a **multi-stage Docker build**:

- **Stage 1 (Builder)**: Trains the model using full training environment
- **Stage 2 (App)**: Creates lightweight production image with only runtime dependencies

**Benefits**:
- Reduced final image size
- Automated training during build
- Production-ready deployment
- Consistent environment across systems

## 🛠️ Available Commands

| Command | Description |
|---------|-------------|
| `make install` | Install dependencies |
| `make train` | Train the CNN model |
| `make eval` | Evaluate model and generate report |
| `make format` | Format code with Black |
| `make deploy` | Deploy to Hugging Face Spaces |

## 📊 Model Details

- **Architecture**: SimpleCNN with 2 convolutional layers
- **Framework**: PyTorch with TorchScript compilation
- **Dataset**: MNIST (28x28 grayscale images)
- **Training**: 5 epochs with Adam optimizer (lr=0.001)
- **Data Augmentation**: Random rotation, translation, and scaling
- **Model Format**: TorchScript (.pt file) for optimized inference
- **Performance**: Training accuracy printed per epoch during training

## 🌐 Cloud Deployment

Deploy to Hugging Face Spaces:

```bash
# Set environment variables
export HF=your_huggingface_token
export USER_NAME="Your Name"
export USER_EMAIL="your@email.com"

# Deploy
make deploy
```

## 📁 Key Files

- **`Scripts/train.py`**: Model training script with SimpleCNN implementation
- **`Scripts/eval.py`**: Model evaluation script
- **`App/app.py`**: Gradio web interface for predictions
- **`App/README.md`**: HuggingFace Space configuration
- **`Notebooks/notebook.ipynb`**: Jupyter notebook for experimentation
- **`Dockerfile`**: Multi-stage build for training and deployment
- **`requirements.txt`**: Training dependencies
- **`App/requirements.txt`**: Production app dependencies

## 🔧 Troubleshooting

### Common Issues

**Model not found error**:
```bash
# Ensure model is trained
make train
```

**Port already in use**:
```bash
# Use different port
docker run -p 8080:7860 mnist-app
```

**Docker build fails**:
```bash
# Clear Docker cache
docker system prune -a
```

## 📄 License

This project is for educational purposes as part of Machine Learning Operations coursework.

## 🔗 Links

- **Hugging Face Space**: [MNIST](https://huggingface.co/spaces/mazenbuk/MNIST)
- **Dataset**: [MNIST](http://yann.lecun.com/exdb/mnist/)
- **Framework**: [PyTorch](https://pytorch.org/)