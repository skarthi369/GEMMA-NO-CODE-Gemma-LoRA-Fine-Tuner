# 🚀 COMPLETE SETUP INSTRUCTIONS FOR REAL TRAINING

## ✅ CURRENT STATUS (What's Working Now):

- ✅ Python 3.11.9 installed
- ✅ Virtual environment created
- ✅ Backend API running on http://localhost:8000 (TEST MODE)
- ✅ Project structure complete
- ⚠️ Full ML dependencies NOT installed yet

---

## 📋 TO ENABLE REAL TRAINING - TWO OPTIONS:

### **OPTION 1: Quick Test (What's Running Now)**

**Currently Active:**
- Backend API: http://localhost:8000
- Gradio not started yet

**What This Can Do:**
- Test dataset upload
- Test API endpoints
- Verify backend works
- Cannot actually train models (no ML libraries)

**Next Step:**
```bash
# In a NEW terminal:
cd c:\Users\skarthi\Downloads\agentic-nids-main\gemma-finetuner
.\venv\Scripts\activate
pip install requests
python frontend/app.py
```

Then visit: http://localhost:7860

---

### **OPTION 2: Full Installation for Real Training** ⭐ RECOMMENDED

This installs everything needed for actual model training.

#### Step 1: Install Full ML Dependencies (~15-20 minutes, ~5GB download)

```powershell
cd c:\Users\skarthi\Downloads\agentic-nids-main\gemma-finetuner
.\venv\Scripts\activate

# Install PyTorch with CUDA 11.8 (for GPU)
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Install Transformers and related
pip install transformers==4.37.2 tokenizers==0.15.0 datasets==2.16.1

# Install PEFT (LoRA)
pip install peft==0.7.1

# Install acceleration and quantization
pip install accelerate==0.26.1 bitsandbytes==0.41.3.post2

# Install Unsloth (may take time)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install remaining dependencies
pip install scipy==1.11.4 scikit-learn xgboost lightgbm
pip install loguru tqdm tensorboard wandb
pip install psutil GPUtil py3nvml
pip install chardet humanize tenacity
```

#### Step 2: Stop Test Backend and Start Real Backend

```powershell
# Stop test backend (Ctrl+C in that terminal)

# Start real backend
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### Step 3: Start Frontend (New Terminal)

```powershell
cd c:\Users\skarthi\Downloads\agentic-nids-main\gemma-finetuner
.\venv\Scripts\activate
cd frontend
python app.py
```

#### Step 4: Access Application

- **Frontend**: http://localhost:7860
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

---

## 🎯 WHAT YOU CAN DO RIGHT NOW (Test Mode):

### Backend API is running at: http://localhost:8000

Test it:
```powershell
# Test health check
curl http://localhost:8000/

# View API documentation
# Open in browser: http://localhost:8000/docs
```

### To start frontend:
```powershell
# New terminal
cd c:\Users\skarthi\Downloads\agentic-nids-main\gemma-finetuner
.\venv\Scripts\activate
pip install requests
python frontend/app.py
```

---

## 📊 INSTALLATION SIZE ESTIMATES:

**Current (Test Mode):**
- Installed: ~500MB
- Can: Test UI, API, workflow
- Cannot: Train models

**Full Installation:**
- Total: ~8GB (5GB PyTorch + 2GB models + 1GB other)
- Time: 15-20 minutes
- Can: Everything including real training!

---

## 🚨 IMPORTANT NOTES:

### For CUDA 12.1 (if you have newer GPU drivers):
```powershell
pip install torch==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### First Training Run:
- Will download ~2-5GB Gemma model
- Saved to: C:\Users\skarthi\.cache\huggingface\
- Subsequent runs reuse cached model

### GPU Memory:
- **Gemma-2B**: Needs 6-8GB VRAM
- **Gemma-7B**: Needs 12GB+ VRAM

Check your GPU:
```powershell
nvidia-smi
```

---

## ✅ RECOMMENDED PATH:

**For Quick UI Test (5 minutes):**
1. Leave backend running (http://localhost:8000)
2. Open new terminal
3. Install requests: `pip install requests`
4. Run frontend: `python frontend/app.py`
5. Visit: http://localhost:7860

**For Real Training (20 minutes):**
1. Follow "OPTION 2" above
2. Install all ML dependencies
3. Replace test backend with real backend
4. Start training with actual models!

---

## 🎓 CURRENT RUNNING PROCESSES:

- ✅ Backend (TEST MODE): Running on port 8000
- ⏳ Frontend: Not started (needs `requests` package)

---

**Your Choice:**
- **Quick Test**: Install `requests` and start frontend now (5 min)
- **Full Setup**: Install all ML deps for real training (20 min)

What would you like to do? 🚀
