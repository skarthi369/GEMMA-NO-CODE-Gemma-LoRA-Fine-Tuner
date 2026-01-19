"""
Training Progress Tracking and Callbacks

This module provides real-time progress tracking for training jobs,
including metrics collection, GPU monitoring, and callback integration.
"""

import time
import torch
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from transformers import TrainerCallback, TrainerState, TrainerControl
import logging

from backend.models import TrainingMetrics, GPUMetrics, JobStatus

logger = logging.getLogger(__name__)


class ProgressTracker:
    """
    Track training progress and metrics in real-time
    
    Maintains state for a single training job including:
    - Current step and epoch
    - Loss and metrics history
    - GPU utilization
    - Time estimates
    """
    
    def __init__(self, job_id: str, total_steps: int, total_epochs: int):
        """
        Initialize progress tracker
        
        Args:
            job_id: Unique job identifier
            total_steps: Total training steps
            total_epochs: Total training epochs
        """
        self.job_id = job_id
        self.total_steps = total_steps
        self.total_epochs = total_epochs
        
        # Current state
        self.current_step = 0
        self.current_epoch = 0.0
        self.current_loss = None
        self.best_loss = float('inf')
        self.learning_rate = None
        
        # Timing
        self.start_time = None
        self.last_update_time = None
        self.steps_per_second = 0.0
        
        # Metrics history
        self.metrics_history: list[TrainingMetrics] = []
        self.status = JobStatus.PENDING
        self.error_message = None
        
        logger.info(f"[{job_id}] ProgressTracker initialized")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Total epochs: {total_epochs}")
    
    def start(self):
        """Mark training as started"""
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.status = JobStatus.TRAINING
        logger.info(f"[{self.job_id}] Training started at {datetime.fromtimestamp(self.start_time)}")
    
    def update(
        self,
        step: int,
        epoch: float,
        loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        **kwargs
    ):
        """
        Update progress with new metrics
        
        Args:
            step: Current training step
            epoch: Current epoch (can be fractional)
            loss: Current loss value
            learning_rate: Current learning rate
            **kwargs: Additional metrics
        """
        current_time = time.time()
        
        # Update state
        steps_delta = step - self.current_step
        time_delta = current_time - (self.last_update_time or current_time)
        
        self.current_step = step
        self.current_epoch = epoch
        self.current_loss = loss
        self.learning_rate = learning_rate
        
        # Update timing metrics
        if time_delta > 0 and steps_delta > 0:
            self.steps_per_second = steps_delta / time_delta
        
        self.last_update_time = current_time
        
        # Track best loss
        if loss is not None and loss < self.best_loss:
            self.best_loss = loss
        
        # Add to metrics history (keep last 100)
        metric = TrainingMetrics(
            step=step,
            epoch=epoch,
            loss=loss or 0.0,
            learning_rate=learning_rate or 0.0,
            timestamp=datetime.fromtimestamp(current_time)
        )
        self.metrics_history.append(metric)
        if len(self.metrics_history) > 100:
            self.metrics_history.pop(0)
        
        logger.debug(
            f"[{self.job_id}] Step {step}/{self.total_steps} | "
            f"Epoch {epoch:.2f}/{self.total_epochs} | "
            f"Loss: {loss:.4f if loss else 'N/A'}"
        )
    
    def get_progress_percent(self) -> float:
        """Calculate overall progress percentage"""
        if self.total_steps <= 0:
            return 0.0
        return min(100.0, (self.current_step / self.total_steps) * 100)
    
    def get_estimated_time_remaining(self) -> Optional[int]:
        """
        Calculate estimated time remaining in seconds
        
        Returns:
            Estimated seconds remaining, or None if can't estimate
        """
        if self.steps_per_second <= 0 or self.current_step >= self.total_steps:
            return None
        
        remaining_steps = self.total_steps - self.current_step
        return int(remaining_steps / self.steps_per_second)
    
    def get_samples_per_second(self, batch_size: int) -> float:
        """
        Calculate samples processed per second
        
        Args:
            batch_size: Training batch size
            
        Returns:
            Samples per second
        """
        return self.steps_per_second * batch_size
    
    def complete(self):
        """Mark training as completed"""
        self.status = JobStatus.COMPLETED
        elapsed = time.time() - (self.start_time or time.time())
        logger.info(f"[{self.job_id}] Training completed")
        logger.info(f"  Total time: {elapsed:.2f}s ({elapsed/60:.2f} minutes)")
        logger.info(f"  Final loss: {self.current_loss:.4f if self.current_loss else 'N/A'}")
        logger.info(f"  Best loss: {self.best_loss:.4f}")
    
    def fail(self, error_message: str):
        """Mark training as failed"""
        self.status = JobStatus.FAILED
        self.error_message = error_message
        logger.error(f"[{self.job_id}] Training failed: {error_message}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tracker state to dictionary"""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "current_epoch": round(self.current_epoch, 2),
            "total_epochs": self.total_epochs,
            "progress_percent": round(self.get_progress_percent(), 2),
            "estimated_time_remaining_seconds": self.get_estimated_time_remaining(),
            "current_loss": round(self.current_loss, 6) if self.current_loss else None,
            "best_loss": round(self.best_loss, 6) if self.best_loss != float('inf') else None,
            "learning_rate": self.learning_rate,
            "samples_per_second": self.steps_per_second,  # Approximate
            "recent_metrics": [
                {
                    "step": m.step,
                    "epoch": round(m.epoch, 2),
                    "loss": round(m.loss, 6),
                    "learning_rate": m.learning_rate,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in self.metrics_history[-10:]  # Last 10 metrics
            ],
            "error_message": self.error_message,
            "updated_at": datetime.now().isoformat()
        }


class ProgressCallback(TrainerCallback):
    """
    Hugging Face Trainer callback for real-time progress updates
    
    Integrates with ProgressTracker to capture training events
    and forward them to the progress tracking system.
    """
    
    def __init__(self, job_id: str, callback_fn: Optional[Callable] = None):
        """
        Initialize progress callback
        
        Args:
            job_id: Unique job identifier
            callback_fn: Optional function to call with progress updates
        """
        self.job_id = job_id
        self.callback_fn = callback_fn
        self.start_time = None
        
        logger.info(f"[{job_id}] ProgressCallback initialized")
    
    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the beginning of training"""
        self.start_time = time.time()
        logger.info(f"[{self.job_id}] Training begin callback triggered")
        
        if self.callback_fn:
            self.callback_fn({
                "event": "train_begin",
                "job_id": self.job_id,
                "max_steps": state.max_steps,
                "total_epochs": args.num_train_epochs,
                "timestamp": datetime.now().isoformat()
            })
    
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of each training step"""
        # Calculate current epoch (fractional)
        current_epoch = state.epoch if state.epoch else 0.0
        
        # Get current loss from logs
        current_loss = None
        if state.log_history:
            for log_entry in reversed(state.log_history):
                if "loss" in log_entry:
                    current_loss = log_entry["loss"]
                    break
        
        # Get learning rate
        learning_rate = None
        if hasattr(state, 'learning_rate'):
            learning_rate = state.learning_rate
        elif state.log_history:
            for log_entry in reversed(state.log_history):
                if "learning_rate" in log_entry:
                    learning_rate = log_entry["learning_rate"]
                    break
        
        # Send update via callback
        if self.callback_fn:
            self.callback_fn({
                "event": "step_end",
                "job_id": self.job_id,
                "step": state.global_step,
                "max_steps": state.max_steps,
                "epoch": current_epoch,
                "loss": current_loss,
                "learning_rate": learning_rate,
                "timestamp": datetime.now().isoformat()
            })
    
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Called when logging occurs"""
        if logs and self.callback_fn:
            self.callback_fn({
                "event": "log",
                "job_id": self.job_id,
                "step": state.global_step,
                "logs": logs,
                "timestamp": datetime.now().isoformat()
            })
    
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of training"""
        elapsed_time = time.time() - (self.start_time or time.time())
        
        logger.info(f"[{self.job_id}] Training end callback triggered")
        logger.info(f"  Total steps: {state.global_step}")
        logger.info(f"  Elapsed time: {elapsed_time:.2f}s")
        
        if self.callback_fn:
            self.callback_fn({
                "event": "train_end",
                "job_id": self.job_id,
                "total_steps": state.global_step,
                "elapsed_time": elapsed_time,
                "timestamp": datetime.now().isoformat()
            })


def get_gpu_metrics() -> Optional[GPUMetrics]:
    """
    Get current GPU utilization metrics
    
    Returns:
        GPUMetrics object with current GPU stats, or None if unavailable
    """
    if not torch.cuda.is_available():
        return None
    
    try:
        import py3nvml.py3nvml as nvml
        
        # Initialize NVML
        nvml.nvmlInit()
        
        # Get first GPU (index 0)
        handle = nvml.nvmlDeviceGetHandleByIndex(0)
        
        # Get memory info
        mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
        
        # Get utilization
        utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
        
        # Get temperature
        try:
            temperature = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
        except:
            temperature = None
        
        # Get name
        name = nvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode('utf-8')
        
        # Cleanup
        nvml.nvmlShutdown()
        
        return GPUMetrics(
            gpu_id=0,
            name=name,
            memory_used_mb=mem_info.used / 1024**2,
            memory_total_mb=mem_info.total / 1024**2,
            memory_percent=(mem_info.used / mem_info.total) * 100,
            utilization_percent=utilization.gpu,
            temperature_c=temperature
        )
        
    except Exception as e:
        logger.warning(f"Failed to get GPU metrics: {e}")
        
        # Fallback to PyTorch metrics
        try:
            return GPUMetrics(
                gpu_id=0,
                name=torch.cuda.get_device_name(0),
                memory_used_mb=torch.cuda.memory_allocated(0) / 1024**2,
                memory_total_mb=torch.cuda.get_device_properties(0).total_memory / 1024**2,
                memory_percent=(torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory) * 100,
                utilization_percent=0.0,  # Not available via PyTorch
                temperature_c=None
            )
        except:
            return None
