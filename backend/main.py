"""
Gemma LoRA Fine-Tuner - FastAPI Backend

Main FastAPI application providing REST API for:
- Dataset upload and validation
- Training job management
- Real-time progress tracking
- Model export and download
"""

import os
import uuid
import asyncio
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import logging

# Local imports
from backend.config import settings, AVAILABLE_MODELS
from backend.models import (
    DatasetUploadResponse, TrainingRequest, TrainingJobResponse,
    TrainingProgress, ExportRequest, ExportResponse, JobListResponse,
    HealthCheck, SystemInfo, ErrorResponse, JobStatus, JobInfo
)
from backend.preprocessing.loader import DatasetLoader
from backend.training.trainer import UnslothTrainer
from backend.training.progress import ProgressTracker, get_gpu_metrics

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global storage for jobs and progress trackers
training_jobs: Dict[str, JobInfo] = {}
progress_trackers: Dict[str, ProgressTracker] = {}
active_trainers: Dict[str, UnslothTrainer] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Gemma Fine-Tuner API...")
    logger.info(f"  API Host: {settings.api_host}:{settings.api_port}")
    logger.info(f"  Datasets: {settings.datasets_dir}")
    logger.info(f"  Models: {settings.models_dir}")
    logger.info(f"  Exports: {settings.exports_dir}")
    
    # Ensure directories exist
    settings.datasets_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.exports_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    
    yield
    
    logger.info("Shutting down Gemma Fine-Tuner API...")


# Create FastAPI app
app = FastAPI(
    title="Gemma LoRA Fine-Tuner API",
    description="Production-ready API for fine-tuning Gemma models with LoRA using Unsloth",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)


# ============================================================================
# Health & System Endpoints
# ============================================================================

@app.get("/", response_model=HealthCheck)
async def health_check():
    """API health check"""
    import torch
    import transformers
    
    return HealthCheck(
        status="healthy",
        version="1.0.0",
        gpu_available=torch.cuda.is_available(),
        gpu_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        cuda_version=torch.version.cuda if hasattr(torch.version, 'cuda') else None,
        timestamp=datetime.now()
    )


@app.get("/api/system", response_model=SystemInfo)
async def get_system_info():
    """Get system information and capabilities"""
    import torch
    import transformers
    
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_metrics = get_gpu_metrics()
            if gpu_metrics:
                gpus.append(gpu_metrics)
    
    return SystemInfo(
        gpu_available=torch.cuda.is_available(),
        gpu_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        gpus=gpus,
        cuda_version=torch.version.cuda if hasattr(torch.version, 'cuda') else None,
        pytorch_version=torch.__version__,
        transformers_version=transformers.__version__,
        unsloth_available=True,  # Verified during import
        available_models=list(AVAILABLE_MODELS.keys()),
        max_concurrent_trainings=settings.max_concurrent_trainings,
        current_active_trainings=len([j for j in training_jobs.values() if j.status == JobStatus.TRAINING])
    )


# ============================================================================
# Dataset Endpoints
# ============================================================================

@app.post("/api/upload", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload and validate a dataset file
    
    Supported formats: CSV, JSON, JSONL, TXT
    Max size: 500MB (configurable)
    """
    logger.info(f"Dataset upload request: {file.filename}")
    
    # Validate file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset
    
    max_size = settings.max_upload_size_mb * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.max_upload_size_mb}MB"
        )
    
    # Validate file format
    file_ext = Path(file.filename).suffix.lower()
    if file_ext.replace('.', '') not in settings.supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Supported: {', '.join(settings.supported_formats)}"
        )
    
    # Generate dataset ID
    dataset_id = str(uuid.uuid4())
    
    # Save file
    dataset_path = settings.datasets_dir / f"{dataset_id}{file_ext}"
    try:
        with open(dataset_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        
        logger.info(f"Dataset saved: {dataset_path}")
        
        return DatasetUploadResponse(
            dataset_id=dataset_id,
            filename=file.filename,
            file_size_bytes=file_size,
            file_size_human=f"{file_size / 1024**2:.2f} MB",
            format=file_ext.replace('.', ''),
            uploaded_at=datetime.now(),
            status="uploaded",
            message="Dataset uploaded successfully"
        )
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# ============================================================================
# Training Endpoints
# ============================================================================

@app.post("/api/train", response_model=TrainingJobResponse)
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Start a fine-tuning job
    
    The training runs in the background while progress can be tracked via /api/progress/{job_id}
    """
    logger.info(f"Training request: {request.job_name}")
    
    # Check concurrent training limit
    active_count = len([j for j in training_jobs.values() if j.status == JobStatus.TRAINING])
    if active_count >= settings.max_concurrent_trainings:
        raise HTTPException(
            status_code=429,
            detail=f"Maximum concurrent trainings ({settings.max_concurrent_trainings}) reached"
        )
    
    # Validate dataset exists
    dataset_files = list(settings.datasets_dir.glob(f"{request.dataset_id}.*"))
    if not dataset_files:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset_path = dataset_files[0]
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Create job info
    job_info = JobInfo(
        job_id=job_id,
        job_name=request.job_name,
        status=JobStatus.PENDING,
        dataset_id=request.dataset_id,
        dataset_name=dataset_path.name,
        model_variant=request.model_variant,
        lora_config=request.lora_config,
        training_config=request.training_config,
        created_at=datetime.now()
    )
    
    training_jobs[job_id] = job_info
    
    # Start training in background
    background_tasks.add_task(run_training, job_id, request, str(dataset_path))
    
    logger.info(f"Training job created: {job_id}")
    
    return TrainingJobResponse(
        job_id=job_id,
        job_name=request.job_name,
        status=JobStatus.PENDING,
        created_at=datetime.now(),
        message="Training job started"
    )


async def run_training(job_id: str, request: TrainingRequest, dataset_path: str):
    """Background task to run training"""
    logger.info(f"[{job_id}] Starting training task...")
    
    try:
        # Update status
        training_jobs[job_id].status = JobStatus.PREPROCESSING
        training_jobs[job_id].started_at = datetime.now()
        
        # Get model ID from config
        model_id = AVAILABLE_MODELS[request.model_variant.value]["model_id"]
        
        # Create trainer
        trainer = UnslothTrainer(
            job_id=job_id,
            model_name=model_id,
            max_seq_length=request.max_seq_length,
            load_in_4bit=settings.load_in_4bit
        )
        
        active_trainers[job_id] = trainer
        
        # Load model
        trainer.load_model()
        
        # Setup LoRA
        trainer.setup_lora(request.lora_config)
        
        # Prepare dataset
        dataset = trainer.prepare_dataset(dataset_path)
        
        # Create progress tracker
        total_steps = len(dataset) // request.training_config.per_device_train_batch_size * request.training_config.num_train_epochs
        tracker = ProgressTracker(job_id, total_steps, request.training_config.num_train_epochs)
        progress_trackers[job_id] = tracker
        tracker.start()
        
        # Train
        training_jobs[job_id].status = JobStatus.TRAINING
        result = trainer.train(dataset, training_config=request.training_config)
        
        # Save model
        output_dir = settings.models_dir / job_id
        trainer.save_model(str(output_dir), save_method="lora")
        
        # Update job info
        training_jobs[job_id].status = JobStatus.COMPLETED
        training_jobs[job_id].completed_at = datetime.now()
        training_jobs[job_id].final_loss = result.training_loss
        tracker.complete()
        
        logger.info(f"[{job_id}] Training completed successfully")
        
    except Exception as e:
        logger.error(f"[{job_id}] Training failed: {str(e)}")
        training_jobs[job_id].status = JobStatus.FAILED
        training_jobs[job_id].error_message = str(e)
        if job_id in progress_trackers:
            progress_trackers[job_id].fail(str(e))
    
    finally:
        # Cleanup
        if job_id in active_trainers:
            active_trainers[job_id].cleanup()
            del active_trainers[job_id]


@app.get("/api/progress/{job_id}", response_model=TrainingProgress)
async def get_training_progress(job_id: str):
    """Get real-time training progress"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = training_jobs[job_id]
    tracker = progress_trackers.get(job_id)
    
    if tracker:
        progress_dict = tracker.to_dict()
        progress_dict["gpu_metrics"] = get_gpu_metrics()
        return TrainingProgress(**progress_dict)
    
    # Return basic status if no tracker yet
    return TrainingProgress(
        job_id=job_id,
        status=job.status,
        current_step=0,
        total_steps=0,
        current_epoch=0.0,
        total_epochs=job.training_config.num_train_epochs,
        progress_percent=0.0,
        updated_at=datetime.now()
    )


@app.get("/api/jobs", response_model=JobListResponse)
async def list_jobs(page: int = 1, page_size: int = 10):
    """List all training jobs"""
    jobs_list = list(training_jobs.values())
    jobs_list.sort(key=lambda x: x.created_at, reverse=True)
    
    start = (page - 1) * page_size
    end = start + page_size
    
    return JobListResponse(
        jobs=jobs_list[start:end],
        total=len(jobs_list),
        page=page,
        page_size=page_size
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
