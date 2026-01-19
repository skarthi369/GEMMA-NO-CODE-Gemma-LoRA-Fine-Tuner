"""
Configuration Module for Gemma LoRA Fine-Tuner

This module handles all configuration settings for the application,
including environment variables, paths, and default parameters.
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    
    Uses Pydantic for validation and type safety
    """
    
    # ========================================================================
    # API Configuration
    # ========================================================================
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    gradio_port: int = Field(default=7860, env="GRADIO_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")
    
    # ========================================================================
    # CORS Settings
    # ========================================================================
    cors_origins: List[str] = Field(
        default=["http://localhost:7860", "http://localhost:3000"],
        env="CORS_ORIGINS"
    )
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: List[str] = Field(default=["*"], env="CORS_ALLOW_METHODS")
    cors_allow_headers: List[str] = Field(default=["*"], env="CORS_ALLOW_HEADERS")
    
    # ========================================================================
    # Storage Paths
    # ========================================================================
    project_root: Path = Path(__file__).parent.parent
    datasets_dir: Path = Field(default=Path("./datasets"), env="DATASETS_DIR")
    models_dir: Path = Field(default=Path("./models"), env="MODELS_DIR")
    exports_dir: Path = Field(default=Path("./exports"), env="EXPORTS_DIR")
    logs_dir: Path = Field(default=Path("./logs"), env="LOGS_DIR")
    temp_dir: Path = Field(default=Path("./temp"), env="TEMP_DIR")
    
    # ========================================================================
    # Model Settings
    # ========================================================================
    default_model: str = Field(
        default="unsloth/gemma-2b-bnb-4bit",
        env="DEFAULT_MODEL"
    )
    max_seq_length: int = Field(default=2048, env="MAX_SEQ_LENGTH")
    load_in_4bit: bool = Field(default=True, env="LOAD_IN_4BIT")
    load_in_8bit: bool = Field(default=False, env="LOAD_IN_8BIT")
    
    # ========================================================================
    # LoRA Training Defaults
    # ========================================================================
    default_lora_r: int = Field(default=16, env="DEFAULT_LORA_R")
    default_lora_alpha: int = Field(default=16, env="DEFAULT_LORA_ALPHA")
    default_lora_dropout: float = Field(default=0.0, env="DEFAULT_LORA_DROPOUT")
    default_epochs: int = Field(default=3, env="DEFAULT_EPOCHS")
    default_batch_size: int = Field(default=2, env="DEFAULT_BATCH_SIZE")
    default_gradient_accumulation_steps: int = Field(
        default=4, env="DEFAULT_GRADIENT_ACCUMULATION_STEPS"
    )
    default_learning_rate: float = Field(default=2e-4, env="DEFAULT_LEARNING_RATE")
    default_warmup_steps: int = Field(default=5, env="DEFAULT_WARMUP_STEPS")
    default_max_steps: int = Field(default=-1, env="DEFAULT_MAX_STEPS")
    default_weight_decay: float = Field(default=0.01, env="DEFAULT_WEIGHT_DECAY")
    default_lr_scheduler_type: str = Field(default="linear", env="DEFAULT_LR_SCHEDULER_TYPE")
    default_logging_steps: int = Field(default=10, env="DEFAULT_LOGGING_STEPS")
    default_save_steps: int = Field(default=100, env="DEFAULT_SAVE_STEPS")
    default_eval_steps: int = Field(default=50, env="DEFAULT_EVAL_STEPS")
    
    # ========================================================================
    # GPU & Memory Settings
    # ========================================================================
    cuda_visible_devices: str = Field(default="0", env="CUDA_VISIBLE_DEVICES")
    max_memory_gb: int = Field(default=12, env="MAX_MEMORY_GB")
    gradient_checkpointing: bool = Field(default=True, env="GRADIENT_CHECKPOINTING")
    fp16: bool = Field(default=False, env="FP16")
    bf16: bool = Field(default=True, env="BF16")
    optim: str = Field(default="adamw_8bit", env="OPTIM")
    
    # ========================================================================
    # Dataset Settings
    # ========================================================================
    max_dataset_size_mb: int = Field(default=500, env="MAX_DATASET_SIZE_MB")
    supported_formats: List[str] = Field(
        default=["csv", "json", "txt", "jsonl"],
        env="SUPPORTED_FORMATS"
    )
    default_text_column: str = Field(default="text", env="DEFAULT_TEXT_COLUMN")
    default_label_column: str = Field(default="label", env="DEFAULT_LABEL_COLUMN")
    train_test_split: float = Field(default=0.1, env="TRAIN_TEST_SPLIT")
    shuffle_dataset: bool = Field(default=True, env="SHUFFLE_DATASET")
    seed: int = Field(default=42, env="SEED")
    
    # ========================================================================
    # Export Settings
    # ========================================================================
    export_lora_only: bool = Field(default=True, env="EXPORT_LORA_ONLY")
    export_merged_model: bool = Field(default=False, env="EXPORT_MERGED_MODEL")
    export_quantized: bool = Field(default=False, env="EXPORT_QUANTIZED")
    push_to_hub: bool = Field(default=False, env="PUSH_TO_HUB")
    hf_token: Optional[str] = Field(default=None, env="HF_TOKEN")
    
    # ========================================================================
    # Logging & Monitoring
    # ========================================================================
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    enable_tensorboard: bool = Field(default=True, env="ENABLE_TENSORBOARD")
    enable_wandb: bool = Field(default=False, env="ENABLE_WANDB")
    wandb_project: str = Field(default="gemma-finetuner", env="WANDB_PROJECT")
    wandb_entity: Optional[str] = Field(default=None, env="WANDB_ENTITY")
    wandb_api_key: Optional[str] = Field(default=None, env="WANDB_API_KEY")
    
    # ========================================================================
    # Security
    # ========================================================================
    secret_key: str = Field(
        default="change-this-secret-key-in-production",
        env="SECRET_KEY"
    )
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(
        default=30,
        env="ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    
    # ========================================================================
    # Database (Optional)
    # ========================================================================
    database_url: str = Field(
        default="sqlite+aiosqlite:///./gemma_finetuner.db",
        env="DATABASE_URL"
    )
    db_echo: bool = Field(default=False, env="DB_ECHO")
    
    # ========================================================================
    # Rate Limiting
    # ========================================================================
    max_concurrent_trainings: int = Field(default=1, env="MAX_CONCURRENT_TRAININGS")
    max_upload_size_mb: int = Field(default=500, env="MAX_UPLOAD_SIZE_MB")
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    
    @validator("datasets_dir", "models_dir", "exports_dir", "logs_dir", "temp_dir")
    def create_directories(cls, v):
        """Ensure all required directories exist"""
        if not v.exists():
            v.mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Setup CUDA environment
os.environ["CUDA_VISIBLE_DEVICES"] = settings.cuda_visible_devices


# ============================================================================
# Available Models Configuration
# ============================================================================
AVAILABLE_MODELS = {
    "gemma-2b": {
        "model_id": "unsloth/gemma-2b-bnb-4bit",
        "display_name": "Gemma 2B (4-bit)",
        "vram_required_gb": 6,
        "description": "Lightweight model, great for consumer GPUs",
    },
    "gemma-7b": {
        "model_id": "unsloth/gemma-7b-bnb-4bit",
        "display_name": "Gemma 7B (4-bit)",
        "vram_required_gb": 12,
        "description": "More capable model, requires 12GB+ VRAM",
    },
    "gemma-2b-it": {
        "model_id": "unsloth/gemma-2b-it-bnb-4bit",
        "display_name": "Gemma 2B Instruct (4-bit)",
        "vram_required_gb": 6,
        "description": "Instruction-tuned variant for chat/QA tasks",
    },
    "gemma-7b-it": {
        "model_id": "unsloth/gemma-7b-it-bnb-4bit",
        "display_name": "Gemma 7B Instruct (4-bit)",
        "vram_required_gb": 12,
        "description": "Instruction-tuned, more capable variant",
    },
}


# ============================================================================
# LoRA Target Modules (Gemma-specific)
# ============================================================================
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


# ============================================================================
# Prompt Templates
# ============================================================================
INSTRUCTION_PROMPT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

CHAT_PROMPT_TEMPLATE = """<start_of_turn>user
{}
<end_of_turn>
<start_of_turn>model
{}
<end_of_turn>"""
