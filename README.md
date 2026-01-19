# 🚀 Gemma LoRA Fine-Tuner - Production No-Code Platform

**A production-ready, no-code web application for fine-tuning Google Gemma models using LoRA, powered by Unsloth for fast and memory-efficient training.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app/)
[![Docker](https://img.shields.io/badge/Docker-GPU-blue.svg)](https://www.docker.com/)

## 🌟 Features

- **🎨 No-Code Interface** - User-friendly Gradio UI for non-technical users
- **📊 Multi-Format Dataset Support** - Upload CSV, JSON, TXT files
- **🔍 Auto Dataset Validation** - Intelligent preprocessing and conversion
- **⚡ Unsloth-Powered Training** - 2x faster training with 60% less memory
- **🎯 LoRA Fine-Tuning** - Efficient parameter-efficient fine-tuning
- **📈 Real-Time Progress** - Live training metrics and progress tracking
- **💾 Model Export** - Download LoRA adapters or merged models
- **🔐 Secure Backend** - FastAPI with production-grade security
- **🐳 Docker + GPU Ready** - Containerized deployment with NVIDIA GPU support
- **📦 Single GPU Optimized** - Runs on consumer-grade GPUs (RTX 3060+)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Gradio Frontend (UI)                     │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │   Dataset    │ │   Training   │ │    Model     │        │
│  │   Upload     │ │   Progress   │ │    Export    │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
└─────────────────────────────────────────────────────────────┘
                             ↕ HTTP/WebSocket
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Backend (API)                     │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │   Upload     │ │   Training   │ │    Export    │        │
│  │   Endpoint   │ │   Manager    │ │   Endpoint   │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
└─────────────────────────────────────────────────────────────┘
                             ↕
┌─────────────────────────────────────────────────────────────┐
│              Unsloth Training Engine (Core)                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │   Gemma      │ │     LoRA     │ │   Progress   │        │
│  │   Loader     │ │   Training   │ │   Tracker    │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
gemma-finetuner/
├── backend/                    # FastAPI backend
│   ├── __init__.py
│   ├── main.py                # FastAPI application entry point
│   ├── config.py              # Configuration and settings
│   ├── models.py              # Pydantic models
│   ├── training/              # Training logic
│   │   ├── __init__.py
│   │   ├── trainer.py         # Unsloth training core
│   │   ├── progress.py        # Progress tracking
│   │   └── callbacks.py       # Training callbacks
│   ├── preprocessing/         # Dataset preprocessing
│   │   ├── __init__.py
│   │   ├── loader.py          # Dataset loaders
│   │   └── validator.py       # Validation logic
│   └── utils/                 # Utilities
│       ├── __init__.py
│       └── gpu_utils.py       # GPU memory management
├── frontend/                  # Gradio frontend
│   ├── __init__.py
│   ├── app.py                 # Gradio UI
│   └── components/            # UI components
│       ├── __init__.py
│       ├── upload.py          # Upload interface
│       ├── training.py        # Training interface
│       └── export.py          # Export interface
├── datasets/                  # Uploaded datasets storage
├── models/                    # Model cache
├── exports/                   # Exported models
├── logs/                      # Training logs
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker Compose setup
├── .env.example              # Environment variables template
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **CUDA 11.8+** (for GPU support)
- **16GB+ RAM**
- **NVIDIA GPU** with 8GB+ VRAM (RTX 3060 12GB or better recommended)
- **Docker** (optional, for containerized deployment)

### 1. Local Installation

```bash
# Clone the repository
git clone <repository-url>
cd gemma-finetuner

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your settings
```

### 2. Run the Application

#### Option A: Run Backend and Frontend Separately

**Terminal 1 - Backend:**
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend
python app.py
```

#### Option B: Run with Docker (Recommended)

```bash
# Build and run with GPU support
docker-compose up --build

# Access the application
# Frontend: http://localhost:7860
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

## 📖 User Guide

### 1. Upload Dataset

1. Navigate to the **Upload** tab
2. Select your dataset file (CSV, JSON, or TXT)
3. Configure dataset format:
   - **CSV**: Specify text column and optional label column
   - **JSON**: Specify keys for text and labels
   - **TXT**: One sample per line
4. Click **Upload & Validate**

### 2. Configure Training

1. Go to the **Training** tab
2. Select uploaded dataset
3. Configure parameters:
   - **Model**: Choose Gemma variant (gemma-2b, gemma-7b)
   - **LoRA Rank**: 8, 16, 32 (higher = more parameters)
   - **LoRA Alpha**: Usually 16 or 32
   - **Epochs**: Number of training iterations
   - **Batch Size**: Adjust based on GPU memory
   - **Learning Rate**: Usually 2e-4 to 3e-4
4. Click **Start Training**

### 3. Monitor Progress

- Real-time progress bar
- Live loss metrics
- GPU memory usage
- Estimated time remaining
- Training logs

### 4. Export Model

1. Navigate to **Export** tab
2. Select completed training run
3. Choose export format:
   - **LoRA Adapters Only** (small, ~100MB)
   - **Merged Model** (full model, ~5GB+)
4. Click **Export & Download**

## ⚙️ Configuration

### Environment Variables (.env)

```env
# API Settings
API_HOST=0.0.0.0
API_PORT=8000
GRADIO_PORT=7860

# Storage Paths
DATASETS_DIR=./datasets
MODELS_DIR=./models
EXPORTS_DIR=./exports
LOGS_DIR=./logs

# Model Settings
DEFAULT_MODEL=unsloth/gemma-2b-bnb-4bit
MAX_SEQ_LENGTH=2048
LOAD_IN_4BIT=true

# Training Defaults
DEFAULT_LORA_R=16
DEFAULT_LORA_ALPHA=16
DEFAULT_EPOCHS=3
DEFAULT_BATCH_SIZE=2
DEFAULT_LEARNING_RATE=2e-4

# GPU Settings
CUDA_VISIBLE_DEVICES=0
MAX_MEMORY_GB=12
```

## 🐳 Docker Deployment

### Docker Compose (Recommended)

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Manual Docker Build

```bash
# Build image
docker build -t gemma-finetuner:latest .

# Run with GPU
docker run --gpus all \
  -p 8000:8000 \
  -p 7860:7860 \
  -v $(pwd)/datasets:/app/datasets \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/exports:/app/exports \
  gemma-finetuner:latest
```

## 📊 Supported Models

- **Gemma 2B** - `unsloth/gemma-2b-bnb-4bit` (Recommended for 8GB VRAM)
- **Gemma 7B** - `unsloth/gemma-7b-bnb-4bit` (Requires 12GB+ VRAM)
- **Gemma 2B Instruct** - `unsloth/gemma-2b-it-bnb-4bit`
- **Gemma 7B Instruct** - `unsloth/gemma-7b-it-bnb-4bit`

## 🎯 Dataset Format Examples

### CSV Format

```csv
text,label
"Sample text for training",category_a
"Another training example",category_b
```

### JSON Format

```json
[
  {"text": "Sample text for training", "label": "category_a"},
  {"text": "Another training example", "label": "category_b"}
]
```

### TXT Format

```
Sample text for training
Another training example
```

## 🔧 Troubleshooting

### Out of Memory (OOM) Errors

1. Reduce `batch_size` to 1
2. Reduce `max_seq_length` to 1024
3. Use smaller model (gemma-2b instead of gemma-7b)
4. Enable gradient checkpointing (already enabled)

### Slow Training

1. Increase `batch_size` if memory allows
2. Use mixed precision (FP16/BF16) - already enabled
3. Ensure CUDA is properly installed
4. Check GPU utilization with `nvidia-smi`

### Dataset Upload Fails

1. Check file format matches specification
2. Ensure file size < 500MB
3. Verify CSV has correct headers
4. Check JSON is valid format

## 📚 API Documentation

### Interactive API Docs

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

- `POST /api/upload` - Upload dataset
- `POST /api/train` - Start training
- `GET /api/progress/{job_id}` - Get training progress
- `GET /api/export/{job_id}` - Export fine-tuned model
- `GET /api/jobs` - List all training jobs

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **Unsloth AI** - For the amazing fast training library
- **Google** - For the Gemma models
- **Hugging Face** - For Transformers and PEFT
- **Gradio** - For the excellent UI framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/gemma-finetuner/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/gemma-finetuner/discussions)

---

**Built with ❤️ for the ML Community**
#   G E M M A - N O - C O D E - G e m m a - L o R A - F i n e - T u n e r 
 
 