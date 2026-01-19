"""
Simplified Backend for Testing (No ML Dependencies Required)

This version works without PyTorch/Transformers/Unsloth for initial testing.
Use this to verify the API works before installing 5GB+ of ML libraries.
"""

import os
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create necessary directories
Path("./datasets").mkdir(exist_ok=True)
Path("./models").mkdir(exist_ok=True)
Path("./exports").mkdir(exist_ok=True)
Path("./logs").mkdir(exist_ok=True)

# Mock storage
training_jobs = {}

app = FastAPI(
    title="Gemma LoRA Fine-Tuner API (Test Mode)",
    description="Backend API for testing - Install full requirements.txt for actual training",
    version="1.0.0-test"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def health_check():
    """API health check"""
    return {
        "status": "healthy",
        "version": "1.0.0-test",
        "gpu_available": False,
        "message": "Backend running in TEST MODE. Install full requirements.txt for training.",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/system")
async def get_system_info():
    """Get system information"""
    return {
        "gpu_available": False,
        "gpu_count": 0,
        "cuda_version": None,
        "pytorch_version": "Not installed",
        "transformers_version": "Not installed",
        "unsloth_available": False,
        "available_models": ["gemma-2b", "gemma-7b"],
        "max_concurrent_trainings": 1,
        "current_active_trainings": 0,
        "mode": "TEST - Install full dependencies for real training"
    }


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload and validate a dataset file"""
    logger.info(f"Dataset upload request: {file.filename}")
    
    # Validate file size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    max_size = 500 * 1024 * 1024  # 500MB
    if file_size > max_size:
        raise HTTPException(status_code=413, detail="File too large. Max size: 500MB")
    
    # Validate file format
    file_ext = Path(file.filename).suffix.lower()
    if file_ext.replace('.', '') not in ['csv', 'json', 'jsonl', 'txt']:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {file_ext}")
    
    # Generate dataset ID
    dataset_id = str(uuid.uuid4())
    
    # Save file
    dataset_path = Path(f"./datasets/{dataset_id}{file_ext}")
    try:
        with open(dataset_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        
        logger.info(f"Dataset saved: {dataset_path}")
        
        return {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "file_size_bytes": file_size,
            "file_size_human": f"{file_size / 1024**2:.2f} MB",
            "format": file_ext.replace('.', ''),
            "uploaded_at": datetime.now().isoformat(),
            "status": "uploaded",
            "message": "Dataset uploaded successfully"
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/api/train")
async def start_training(request: dict, background_tasks: BackgroundTasks):
    """Start a training job (simulated in test mode)"""
    logger.info(f"Training request received")
    
    job_id = str(uuid.uuid4())
    
    job_info = {
        "job_id": job_id,
        "job_name": request.get("job_name", f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        "status": "pending",
        "dataset_id": request.get("dataset_id"),
        "model_variant": request.get("model_variant", "gemma-2b"),
        "created_at": datetime.now().isoformat(),
        "message": "TEST MODE: Install full requirements.txt for real training"
    }
    
    training_jobs[job_id] = job_info
    
    logger.info(f"Training job created: {job_id} (TEST MODE)")
    
    return {
        "job_id": job_id,
        "job_name": job_info["job_name"],
        "status": "pending",
        "created_at": job_info["created_at"],
        "message": "Job created in TEST MODE. Install full dependencies for real training."
    }


@app.get("/api/progress/{job_id}")
async def get_training_progress(job_id: str):
    """Get training progress"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = training_jobs[job_id]
    
    return {
        "job_id": job_id,
        "status": job.get("status", "pending"),
        "current_step": 0,
        "total_steps": 0,
        "current_epoch": 0.0,
        "total_epochs": 3,
        "progress_percent": 0.0,
        "current_loss": None,
        "message": "TEST MODE: Real training requires full installation",
        "updated_at": datetime.now().isoformat()
    }


@app.get("/api/jobs")
async def list_jobs(page: int = 1, page_size: int = 10):
    """List all training jobs"""
    jobs_list = list(training_jobs.values())
    jobs_list.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    
    start = (page - 1) * page_size
    end = start + page_size
    
    return {
        "jobs": jobs_list[start:end],
        "total": len(jobs_list),
        "page": page,
        "page_size": page_size
    }


if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("🚀 Starting Gemma Fine-Tuner Backend (TEST MODE)")
    print("=" * 70)
    print("Backend running at: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print("")
    print("⚠️  TEST MODE: For real training, install full requirements.txt")
    print("=" * 70)
    
    uvicorn.run(
        "simple_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
