"""
Pydantic Models for API Request/Response Validation

Defines all data models used in the FastAPI application for
type safety, validation, and automatic documentation.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field, validator


# ============================================================================
# Enums
# ============================================================================

class JobStatus(str, Enum):
    """Training job status enumeration"""
    PENDING = "pending"
    VALIDATING = "validating"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DatasetFormat(str, Enum):
    """Supported dataset file formats"""
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    TXT = "txt"


class ModelVariant(str, Enum):
    """Available Gemma model variants"""
    GEMMA_2B = "gemma-2b"
    GEMMA_7B = "gemma-7b"
    GEMMA_2B_IT = "gemma-2b-it"
    GEMMA_7B_IT = "gemma-7b-it"


# ============================================================================
# Dataset Models
# ============================================================================

class DatasetUploadResponse(BaseModel):
    """Response after dataset upload"""
    dataset_id: str
    filename: str
    file_size_bytes: int
    file_size_human: str
    format: DatasetFormat
    uploaded_at: datetime
    status: str
    message: str


class DatasetInfo(BaseModel):
    """Dataset information and statistics"""
    dataset_id: str
    filename: str
    format: DatasetFormat
    num_samples: int
    num_train: int
    num_val: int
    text_column: str
    label_column: Optional[str] = None
    sample_preview: List[Dict[str, Any]] = Field(default_factory=list)
    uploaded_at: datetime
    validated: bool = False


class DatasetValidationRequest(BaseModel):
    """Request for dataset validation"""
    dataset_id: str
    text_column: str = "text"
    label_column: Optional[str] = None
    max_seq_length: int = 2048
    train_test_split: float = 0.1


# ============================================================================
# Training Models
# ============================================================================

class LoRAConfig(BaseModel):
    """LoRA configuration parameters"""
    r: int = Field(default=16, ge=1, le=64, description="LoRA rank")
    alpha: int = Field(default=16, ge=1, le=128, description="LoRA alpha")
    dropout: float = Field(default=0.0, ge=0.0, le=0.5, description="LoRA dropout")
    bias: str = Field(default="none", description="Bias type: none, all, lora_only")
    task_type: str = Field(default="CAUSAL_LM", description="Task type")
    target_modules: Optional[List[str]] = None
    
    @validator("alpha")
    def alpha_should_match_r(cls, v, values):
        """Typically alpha should be equal to or 2x the rank"""
        if "r" in values and v < values["r"]:
            raise ValueError("alpha should be >= r")
        return v


class TrainingConfig(BaseModel):
    """Training hyperparameters configuration"""
    num_train_epochs: int = Field(default=3, ge=1, le=100, description="Number of epochs")
    per_device_train_batch_size: int = Field(default=2, ge=1, le=32, description="Batch size")
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=64)
    learning_rate: float = Field(default=2e-4, gt=0, lt=1)
    warmup_steps: int = Field(default=5, ge=0)
    max_steps: int = Field(default=-1, description="-1 for full training")
    weight_decay: float = Field(default=0.01, ge=0, le=1)
    lr_scheduler_type: str = Field(default="linear")
    logging_steps: int = Field(default=10, ge=1)
    save_steps: int = Field(default=100, ge=1)
    eval_steps: int = Field(default=50, ge=1)
    fp16: bool = Field(default=False)
    bf16: bool = Field(default=True)
    optim: str = Field(default="adamw_8bit")
    gradient_checkpointing: bool = Field(default=True)
    max_grad_norm: float = Field(default=1.0)


class TrainingRequest(BaseModel):
    """Request to start a training job"""
    dataset_id: str
    model_variant: ModelVariant = ModelVariant.GEMMA_2B
    job_name: Optional[str] = None
    lora_config: LoRAConfig = Field(default_factory=LoRAConfig)
    training_config: TrainingConfig = Field(default_factory=TrainingConfig)
    max_seq_length: int = Field(default=2048, ge=128, le=4096)
    prompt_template: Optional[str] = None
    
    @validator("job_name", always=True)
    def generate_job_name(cls, v, values):
        """Generate job name if not provided"""
        if not v:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model = values.get("model_variant", "gemma").value
            return f"{model}_training_{timestamp}"
        return v


class TrainingJobResponse(BaseModel):
    """Response after starting a training job"""
    job_id: str
    job_name: str
    status: JobStatus
    created_at: datetime
    message: str


# ============================================================================
# Progress Tracking Models
# ============================================================================

class TrainingMetrics(BaseModel):
    """Training metrics at a specific step"""
    step: int
    epoch: float
    loss: float
    learning_rate: float
    grad_norm: Optional[float] = None
    timestamp: datetime


class GPUMetrics(BaseModel):
    """GPU utilization metrics"""
    gpu_id: int
    name: str
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    utilization_percent: float
    temperature_c: Optional[float] = None


class TrainingProgress(BaseModel):
    """Real-time training progress"""
    job_id: str
    status: JobStatus
    current_step: int
    total_steps: int
    current_epoch: float
    total_epochs: int
    progress_percent: float
    estimated_time_remaining_seconds: Optional[int] = None
    current_loss: Optional[float] = None
    best_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    samples_per_second: Optional[float] = None
    gpu_metrics: Optional[GPUMetrics] = None
    recent_metrics: List[TrainingMetrics] = Field(default_factory=list)
    error_message: Optional[str] = None
    updated_at: datetime


# ============================================================================
# Export Models
# ============================================================================

class ExportRequest(BaseModel):
    """Request to export a trained model"""
    job_id: str
    export_lora_only: bool = True
    export_merged_model: bool = False
    quantize_merged: bool = False
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None


class ExportResponse(BaseModel):
    """Response after model export"""
    job_id: str
    export_id: str
    lora_adapters_path: Optional[str] = None
    merged_model_path: Optional[str] = None
    download_urls: Dict[str, str] = Field(default_factory=dict)
    file_sizes: Dict[str, str] = Field(default_factory=dict)
    hub_url: Optional[str] = None
    created_at: datetime
    message: str


# ============================================================================
# Job Management Models
# ============================================================================

class JobInfo(BaseModel):
    """Complete job information"""
    job_id: str
    job_name: str
    status: JobStatus
    dataset_id: str
    dataset_name: str
    model_variant: ModelVariant
    lora_config: LoRAConfig
    training_config: TrainingConfig
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    final_loss: Optional[float] = None
    best_loss: Optional[float] = None
    total_steps: int = 0
    error_message: Optional[str] = None


class JobListResponse(BaseModel):
    """List of training jobs"""
    jobs: List[JobInfo]
    total: int
    page: int = 1
    page_size: int = 10


# ============================================================================
# System Models
# ============================================================================

class HealthCheck(BaseModel):
    """API health check response"""
    status: str
    version: str
    gpu_available: bool
    gpu_count: int
    cuda_version: Optional[str] = None
    timestamp: datetime


class SystemInfo(BaseModel):
    """System information and capabilities"""
    gpu_available: bool
    gpu_count: int
    gpus: List[GPUMetrics] = Field(default_factory=list)
    cuda_version: Optional[str] = None
    pytorch_version: str
    transformers_version: str
    unsloth_available: bool
    available_models: List[str]
    max_concurrent_trainings: int
    current_active_trainings: int


# ============================================================================
# Error Models
# ============================================================================

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
