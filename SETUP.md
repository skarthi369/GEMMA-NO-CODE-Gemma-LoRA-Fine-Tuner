# Gemma LoRA Fine-Tuner - Setup & Installation Guide

## 🚀 Quick Start Guide

This guide will help you set up and run the Gemma LoRA Fine-Tuner application on your local machine or server.

---

## 📋 Prerequisites

### Hardware Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3060 12GB, RTX 3080, RTX 4070, or better)
- **RAM**: 16GB+ system RAM
- **Storage**: 50GB+ free disk space (for models and datasets)
- **CPU**: Modern multi-core processor

### Software Requirements

- **Operating System**: Linux (Ubuntu 20.04+), Windows 10/11 with WSL2, or macOS
- **Python**: 3.10 or 3.11
- **CUDA**: 11.8 or 12.1 (for GPU support)
- **Docker**: Latest version (optional, for containerized deployment)
- **Git**: For cloning the repository

---

## 🔧 Installation Methods

### Method 1: Local Installation (Recommended for Development)

#### Step 1: Clone or Navigate to the Project

```bash
cd gemma-finetuner
```

#### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

#### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

**Note:** The installation may take 10-20 minutes depending on your internet speed, as it downloads large packages like PyTorch and Transformers.

#### Step 4: Setup Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env file with your settings (optional)
nano .env  # or use your preferred editor
```

Key environment variables to configure:
- `API_HOST`: Default is `0.0.0.0` (all interfaces)
- `API_PORT`: Default is `8000`
- `GRADIO_PORT`: Default is `7860`
- `CUDA_VISIBLE_DEVICES`: GPU to use (default is `0`)
- `MAX_SEQ_LENGTH`: Max sequence length (default is `2048`)

#### Step 5: Verify GPU Setup

```bash
# Test CUDA availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}')"
```

Expected output:
```
CUDA Available: True
GPU Count: 1
GPU Name: NVIDIA GeForce RTX 3060
```

---

### Method 2: Docker Installation (Recommended for Production)

#### Step 1: Install Docker and NVIDIA Container Toolkit

**On Ubuntu:**

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### Step 2: Build and Run with Docker Compose

```bash
# Build and start services
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

#### Step 3: Verify Containers are Running

```bash
docker-compose ps
```

#### Step 4: View Logs

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f gemma-finetuner
```

---

## 🎯 Running the Application

### Option A: Run Backend and Frontend Separately (Development)

#### Terminal 1 - Start Backend API:

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### Terminal 2 - Start Frontend UI:

```bash
cd frontend
python app.py
```

### Option B: Run with Docker (Production):

```bash
docker-compose up
```

---

## 🌐 Accessing the Application

Once running, access the application at:

- **Gradio Frontend**: http://localhost:7860
- **FastAPI Backend**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **Alternative API Docs**: http://localhost:8000/redoc (ReDoc)

---

## 📖 Usage Tutorial

### Step 1: Upload Dataset

1. Navigate to http://localhost:7860
2. Go to the **"📤 Upload Dataset"** tab
3. Click **"Select Dataset File"** and choose your file (CSV, JSON, JSONL, or TXT)
4. Click **"📤 Upload Dataset"**
5. Wait for upload confirmation

### Step 2: Configure Training

1. Go to the **"⚙️ Configure & Train"** tab
2. Select your **Model Variant**:
   - **gemma-2b**: Best for 8GB VRAM GPUs
   - **gemma-7b**: Requires 12GB+ VRAM
   - **gemma-2b-it**: Instruction-tuned 2B model
   - **gemma-7b-it**: Instruction-tuned 7B model

3. Configure **LoRA Parameters**:
   - **LoRA Rank (r)**: 8-32 (16 is recommended)
   - **LoRA Alpha**: Usually same as rank (16)

4. Set **Training Hyperparameters**:
   - **Epochs**: 3-5 for most tasks
   - **Batch Size**: 2 (reduce to 1 if OOM)
   - **Learning Rate**: 2e-4 (recommended)

5. Click **"🚀 Start Training"**

### Step 3: Monitor Progress

1. Go to the **"📊 Training Progress"** tab
2. Click **"🔄 Refresh Progress"** to see real-time updates
3. Monitor:
   - Training status
   - Progress percentage
   - Current loss
   - ETA (estimated time remaining)

### Step 4: Export Model

1. Once training completes, go to the **"💾 Export Model"** tab
2. Your fine-tuned model will be saved in `models/{job_id}/`
3. LoRA adapters are automatically saved (small, ~100MB)

---

## 🧪 Testing & Validation

### Test 1: Health Check

```bash
curl http://localhost:8000/
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "gpu_available": true,
  "gpu_count": 1,
  "cuda_version": "11.8"
}
```

### Test 2: System Information

```bash
curl http://localhost:8000/api/system
```

### Test 3: Upload Sample Dataset

Create a test CSV file:

```bash
echo "text,label" > test_dataset.csv
echo "This is a sample text,positive" >> test_dataset.csv
echo "Another example sentence,negative" >> test_dataset.csv
```

Upload via API:

```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@test_dataset.csv"
```

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory (OOM)

**Solutions:**
1. Reduce `batch_size` to 1
2. Reduce `max_seq_length` from 2048 to 1024
3. Use `gemma-2b` instead of `gemma-7b`
4. Close other GPU-intensive applications

### Issue: CUDA Not Available

**Solutions:**
1. Verify NVIDIA drivers: `nvidia-smi`
2. Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Reinstall PyTorch with correct CUDA version:
   ```bash
   pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
   ```

### Issue: Slow Training

**Solutions:**
1. Increase `batch_size` if memory allows
2. Reduce `logging_steps` to decrease overhead
3. Ensure GPU is being utilized: `nvidia-smi` (should show high GPU usage)

### Issue: Import Errors

**Solutions:**
1. Reinstall dependencies:
   ```bash
   pip install --upgrade --force-reinstall -r requirements.txt
   ```
2. Check Python version: `python --version` (should be 3.10 or 3.11)

---

## 📊 Performance Tips

### Maximize Training Speed:

1. **Use BF16 Precision**: Already enabled by default (2x faster than FP32)
2. **Enable Gradient Checkpointing**: Already enabled (reduces memory)
3. **Optimize Batch Size**: Use largest batch size that fits in VRAM
4. **Use Fast Model Variant**: `gemma-2b` trains 3x faster than `gemma-7b`

### Expected Training Times:

On RTX 3060 12GB:
- **Gemma-2B**: ~30 min for 1000 samples (3 epochs)
- **Gemma-7B**: ~90 min for 1000 samples (3 epochs)

On RTX 4090:
- **Gemma-2B**: ~10 min for 1000 samples (3 epochs)
- **Gemma-7B**: ~25 min for 1000 samples (3 epochs)

---

## 🔐 Security Best Practices

1. **Change Secret Key**: Update `SECRET_KEY` in `.env` for production
2. **Enable HTTPS**: Use reverse proxy (nginx) with SSL certificate
3. **Limit CORS Origins**: Update `CORS_ORIGINS` to specific domains
4. **Use Authentication**: Add API key authentication for production
5. **Firewall Rules**: Only expose necessary ports

---

## 📚 Additional Resources

- **Unsloth Documentation**: https://github.com/unslothai/unsloth
- **Gemma Model Card**: https://huggingface.co/google/gemma-2b
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Gradio Docs**: https://gradio.app/

---

## 🤝 Support

If you encounter issues:

1. Check the logs: `tail -f logs/*.log`
2. Review error messages in browser console
3. Check GPU memory: `nvidia-smi`
4. Verify all dependencies are installed correctly

---

## ✅ Setup Checklist

Before starting training, ensure:

- [ ] GPU is detected and CUDA is available
- [ ] All Python dependencies are installed
- [ ] Backend API is running on port 8000
- [ ] Frontend UI is accessible on port 7860
- [ ] Dataset is uploaded successfully
- [ ] Training parameters are configured
- [ ] Sufficient disk space (50GB+) is available
- [ ] GPU has sufficient VRAM for selected model

---

**Happy Fine-Tuning! 🚀**
