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
│  │   Dataset    │ │   Training   │ │    Model     │        │
│  │  Processing  │ │   Manager    │ │   Manager    │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
└─────────────────────────────────────────────────────────────┘
                             ↕
┌─────────────────────────────────────────────────────────────┐
│              Unsloth Training Engine (GPU)                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │    Gemma     │ │     LoRA     │ │   Optimizer  │        │
│  │    Model     │ │   Adapters   │ │   (4-bit)    │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- CUDA 11.8+ or 12.1+
- 16GB+ System RAM
- Docker (optional, for containerized deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/skarthi369/GEMMA-NO-CODE-Gemma-LoRA-Fine-Tuner.git
cd GEMMA-NO-CODE-Gemma-LoRA-Fine-Tuner

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your Hugging Face token

# Run the application
python working_frontend.py
```

### Docker Deployment

```bash
# Build and run with GPU support
docker-compose up --build

# Access the application at http://localhost:7860
```

## 🎯 Usage

1. **Upload Dataset** - Drag and drop your CSV, JSON, or TXT file
2. **Configure Training** - Set epochs, batch size, learning rate
3. **Start Training** - Monitor real-time progress and metrics
4. **Export Model** - Download LoRA adapters or merged model

## 📊 Dataset Format

### CSV Format
```csv
instruction,input,output
"Translate to French","Hello","Bonjour"
"Summarize","Long text...","Summary..."
```

### JSON Format
```json
[
  {
    "instruction": "Translate to French",
    "input": "Hello",
    "output": "Bonjour"
  }
]
```

### TXT Format
```
### Instruction: Translate to French
### Input: Hello
### Output: Bonjour

### Instruction: Summarize
### Input: Long text...
### Output: Summary...
```

## 🔧 Configuration

Key environment variables in `.env`:

```bash
# Hugging Face
HF_TOKEN=your_huggingface_token_here

# Model Settings
MODEL_NAME=unsloth/gemma-2-2b-it-bnb-4bit
MAX_SEQ_LENGTH=2048

# Training Defaults
DEFAULT_EPOCHS=3
DEFAULT_BATCH_SIZE=2
DEFAULT_LEARNING_RATE=2e-4

# Server
BACKEND_PORT=8000
FRONTEND_PORT=7860
```

## 📈 Performance

| Metric | Standard | Unsloth Optimized |
|--------|----------|-------------------|
| Training Speed | 1x | 2x faster |
| Memory Usage | 100% | 40% (60% reduction) |
| Min GPU VRAM | 24GB | 8GB |
| Batch Size (8GB) | 1 | 4 |

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn
- **Frontend**: Gradio
- **ML Framework**: PyTorch, Transformers, Unsloth
- **Model**: Google Gemma 2B/7B
- **Fine-Tuning**: LoRA (Low-Rank Adaptation)
- **Optimization**: 4-bit quantization, Flash Attention 2
- **Containerization**: Docker, Docker Compose

## 📁 Project Structure

```
.
├── backend/                 # FastAPI backend modules
├── frontend/               # Gradio UI components
├── datasets/               # Training datasets
├── models/                 # Downloaded and fine-tuned models
├── exports/                # Exported LoRA adapters
├── logs/                   # Training logs
├── simple_backend.py       # Main backend server
├── working_frontend.py     # Main frontend application
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose setup
└── .env.example           # Environment template
```

## 🚀 Advanced Features

### Custom Model Support
Fine-tune any Gemma variant:
- `gemma-2-2b-it` (Consumer GPUs)
- `gemma-2-7b-it` (Professional GPUs)
- `gemma-2-9b-it` (High-end GPUs)

### LoRA Configuration
```python
lora_config = {
    "r": 16,              # LoRA rank
    "lora_alpha": 16,     # LoRA alpha
    "lora_dropout": 0,    # Dropout
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
}
```

## 🔒 Security

- Environment-based configuration
- No hardcoded credentials
- Rate limiting on API endpoints
- Input validation and sanitization
- Secure file upload handling

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size in .env
DEFAULT_BATCH_SIZE=1

# Use gradient checkpointing
USE_GRADIENT_CHECKPOINTING=true
```

### Slow Training
```bash
# Enable Flash Attention 2
USE_FLASH_ATTENTION=true

# Increase batch size if memory allows
DEFAULT_BATCH_SIZE=4
```

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Contact

- **Author**: Karthikeyan S
- **GitHub**: [@skarthi369](https://github.com/skarthi369)
- **Repository**: [GEMMA-NO-CODE-Gemma-LoRA-Fine-Tuner](https://github.com/skarthi369/GEMMA-NO-CODE-Gemma-LoRA-Fine-Tuner)

## 🙏 Acknowledgments

- Google for Gemma models
- Unsloth for optimization framework
- Hugging Face for model hosting
- FastAPI and Gradio communities

---

**⭐ Star this repository if you find it useful!**