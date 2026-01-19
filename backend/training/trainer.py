"""
Unsloth Training Engine Core

This module implements the core training logic using Unsloth for fast,
memory-efficient fine-tuning of Gemma models with LoRA.

Key features:
- Unsloth FastLanguageModel for 2x faster training
- 4-bit quantization for memory efficiency
- LoRA/QLoRA parameter-efficient fine-tuning
- Gradient checkpointing for reduced memory
- Real-time progress tracking via callbacks
"""

import os
import gc
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from datetime import datetime
import logging

# Unsloth imports
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments

# Transformers imports
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset, load_dataset

# PEFT imports
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Local imports
from backend.config import settings, LORA_TARGET_MODULES
from backend.models import LoRAConfig, TrainingConfig, JobStatus
from backend.training.callbacks import ProgressCallback

logger = logging.getLogger(__name__)


class UnslothTrainer:
    """
    Unsloth-powered trainer for Gemma models with LoRA fine-tuning
    
    This class handles the entire training pipeline:
    1. Model loading with Unsloth optimizations
    2. LoRA configuration and application
    3. Dataset preparation and tokenization
    4. Training with progress tracking
    5. Model saving and export
    """
    
    def __init__(
        self,
        job_id: str,
        model_name: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        progress_callback: Optional[Callable] = None
    ):
        """
        Initialize the Unsloth trainer
        
        Args:
            job_id: Unique identifier for this training job
            model_name: Hugging Face model identifier (e.g., 'unsloth/gemma-2b-bnb-4bit')
            max_seq_length: Maximum sequence length for training
            load_in_4bit: Whether to load model in 4-bit quantization
            progress_callback: Callback function for progress updates
        """
        self.job_id = job_id
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.progress_callback = progress_callback
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Training state
        self.is_training = False
        self.should_stop = False
        
        logger.info(f"[{job_id}] Initialized UnslothTrainer")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Max sequence length: {max_seq_length}")
        logger.info(f"  4-bit quantization: {load_in_4bit}")
    
    def load_model(self):
        """
        Load Gemma model using Unsloth with optimizations
        
        Unsloth provides:
        - Fast loading with automatic 4-bit quantization
        - Pre-configured for LoRA training
        - Optimized kernels for 2x speedup
        """
        logger.info(f"[{self.job_id}] Loading model: {self.model_name}")
        
        try:
            # Clear GPU cache before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Load model with Unsloth optimizations
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                dtype=None,  # Auto-detect (bf16 if supported)
                load_in_4bit=self.load_in_4bit,
                # Additional Unsloth optimizations
                trust_remote_code=True,
            )
            
            logger.info(f"[{self.job_id}] ✓ Model loaded successfully")
            logger.info(f"  Device: {next(self.model.parameters()).device}")
            logger.info(f"  Dtype: {next(self.model.parameters()).dtype}")
            
            # Log GPU memory if available
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"  GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
            return True
            
        except Exception as e:
            logger.error(f"[{self.job_id}] ✗ Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def setup_lora(self, lora_config: LoRAConfig):
        """
        Configure LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
        
        Args:
            lora_config: LoRA configuration parameters
        """
        logger.info(f"[{self.job_id}] Setting up LoRA configuration")
        logger.info(f"  Rank (r): {lora_config.r}")
        logger.info(f"  Alpha: {lora_config.alpha}")
        logger.info(f"  Dropout: {lora_config.dropout}")
        
        try:
            # Apply LoRA using Unsloth's optimized method
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=lora_config.r,
                target_modules=lora_config.target_modules or LORA_TARGET_MODULES,
                lora_alpha=lora_config.alpha,
                lora_dropout=lora_config.dropout,
                bias=lora_config.bias,
                use_gradient_checkpointing=settings.gradient_checkpointing,
                random_state=settings.seed,
                use_rslora=False,  # Rank-Stabilized LoRA (optional)
                loftq_config=None,  # LoftQ initialization (optional)
            )
            
            # Print trainable parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_percent = 100 * trainable_params / total_params
            
            logger.info(f"[{self.job_id}] ✓ LoRA configured successfully")
            logger.info(f"  Trainable parameters: {trainable_params:,} ({trainable_percent:.2f}%)")
            logger.info(f"  Total parameters: {total_params:,}")
            
            return True
            
        except Exception as e:
            logger.error(f"[{self.job_id}] ✗ Failed to setup LoRA: {str(e)}")
            raise RuntimeError(f"LoRA setup failed: {str(e)}")
    
    def prepare_dataset(
        self,
        dataset_path: str,
        text_column: str = "text",
        prompt_template: Optional[str] = None
    ) -> Dataset:
        """
        Load and prepare dataset for training
        
        Args:
            dataset_path: Path to dataset file or directory
            text_column: Column name containing text data
            prompt_template: Optional prompt template for formatting
            
        Returns:
            Prepared Hugging Face dataset
        """
        logger.info(f"[{self.job_id}] Preparing dataset: {dataset_path}")
        
        try:
            # Load dataset based on file extension
            dataset_path_obj = Path(dataset_path)
            
            if dataset_path_obj.suffix == '.json':
                dataset = load_dataset('json', data_files=str(dataset_path))['train']
            elif dataset_path_obj.suffix == '.jsonl':
                dataset = load_dataset('json', data_files=str(dataset_path))['train']
            elif dataset_path_obj.suffix == '.csv':
                dataset = load_dataset('csv', data_files=str(dataset_path))['train']
            else:
                raise ValueError(f"Unsupported dataset format: {dataset_path_obj.suffix}")
            
            logger.info(f"  Loaded {len(dataset)} samples")
            
            # Tokenization function
            def tokenize_function(examples):
                """Tokenize text with proper formatting"""
                texts = examples[text_column]
                
                # Apply prompt template if provided
                if prompt_template:
                    texts = [prompt_template.format(text) for text in texts]
                
                # Tokenize with padding and truncation
                tokenized = self.tokenizer(
                    texts,
                    truncation=True,
                    max_length=self.max_seq_length,
                    padding="max_length",
                    return_tensors=None,  # Return lists, not tensors
                )
                
                # For causal LM, labels are the same as input_ids
                tokenized["labels"] = tokenized["input_ids"].copy()
                
                return tokenized
            
            # Apply tokenization
            logger.info(f"[{self.job_id}] Tokenizing dataset...")
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names,
                desc="Tokenizing",
            )
            
            logger.info(f"[{self.job_id}] ✓ Dataset prepared successfully")
            logger.info(f"  Total samples: {len(tokenized_dataset)}")
            
            return tokenized_dataset
            
        except Exception as e:
            logger.error(f"[{self.job_id}] ✗ Failed to prepare dataset: {str(e)}")
            raise RuntimeError(f"Dataset preparation failed: {str(e)}")
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        training_config: TrainingConfig = None,
        output_dir: Optional[str] = None
    ):
        """
        Execute training with Unsloth-optimized trainer
        
        Args:
            train_dataset: Prepared training dataset
            eval_dataset: Optional validation dataset
            training_config: Training hyperparameters
            output_dir: Directory to save checkpoints
        """
        if training_config is None:
            training_config = TrainingConfig()
        
        if output_dir is None:
            output_dir = str(settings.models_dir / self.job_id)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[{self.job_id}] Starting training")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Training samples: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"  Validation samples: {len(eval_dataset)}")
        
        try:
            # Create Unsloth training arguments (compatible with Transformers)
            training_args = UnslothTrainingArguments(
                # Output
                output_dir=output_dir,
                overwrite_output_dir=True,
                
                # Training hyperparameters
                num_train_epochs=training_config.num_train_epochs,
                per_device_train_batch_size=training_config.per_device_train_batch_size,
                gradient_accumulation_steps=training_config.gradient_accumulation_steps,
                learning_rate=training_config.learning_rate,
                weight_decay=training_config.weight_decay,
                warmup_steps=training_config.warmup_steps,
                max_steps=training_config.max_steps,
                max_grad_norm=training_config.max_grad_norm,
                
                # Optimizer
                optim=training_config.optim,
                lr_scheduler_type=training_config.lr_scheduler_type,
                
                # Mixed precision
                fp16=training_config.fp16 and not is_bfloat16_supported(),
                bf16=training_config.bf16 and is_bfloat16_supported(),
                
                # Logging
                logging_steps=training_config.logging_steps,
                logging_dir=str(settings.logs_dir / self.job_id),
                report_to=["tensorboard"] if settings.enable_tensorboard else None,
                
                # Checkpointing
                save_strategy="steps",
                save_steps=training_config.save_steps,
                save_total_limit=3,
                
                # Evaluation
                evaluation_strategy="steps" if eval_dataset else "no",
                eval_steps=training_config.eval_steps if eval_dataset else None,
                
                # Performance
                gradient_checkpointing=training_config.gradient_checkpointing,
                dataloader_num_workers=0,  # Use 0 for GPU training
                remove_unused_columns=False,
                
                # Misc
                seed=settings.seed,
                data_seed=settings.seed,
            )
            
            # Create data collator for language modeling
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,  # Causal LM, not masked LM
            )
            
            # Create progress callback
            callbacks = []
            if self.progress_callback:
                callbacks.append(
                    ProgressCallback(
                        job_id=self.job_id,
                        callback_fn=self.progress_callback
                    )
                )
            
            # Create Unsloth trainer (optimized Transformers Trainer)
            self.trainer = UnslothTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                callbacks=callbacks,
            )
            
            logger.info(f"[{self.job_id}] 🚀 Training started...")
            self.is_training = True
            
            # Start training
            train_result = self.trainer.train()
            
            logger.info(f"[{self.job_id}] ✓ Training completed successfully")
            logger.info(f"  Final loss: {train_result.training_loss:.4f}")
            
            # Save final metrics
            metrics = train_result.metrics
            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)
            self.trainer.save_state()
            
            self.is_training = False
            
            return train_result
            
        except Exception as e:
            self.is_training = False
            logger.error(f"[{self.job_id}] ✗ Training failed: {str(e)}")
            raise RuntimeError(f"Training failed: {str(e)}")
    
    def save_model(self, output_dir: str, save_method: str = "lora"):
        """
        Save the fine-tuned model
        
        Args:
            output_dir: Directory to save the model
            save_method: 'lora' for LoRA adapters only, 'merged' for full model
        """
        logger.info(f"[{self.job_id}] Saving model to: {output_dir}")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            if save_method == "lora":
                # Save only LoRA adapters (small, ~100MB)
                logger.info("  Saving LoRA adapters only...")
                self.model.save_pretrained(str(output_path))
                self.tokenizer.save_pretrained(str(output_path))
                logger.info(f"  ✓ LoRA adapters saved")
                
            elif save_method == "merged":
                # Merge LoRA weights with base model and save
                logger.info("  Merging LoRA with base model...")
                merged_model = self.model.merge_and_unload()
                merged_model.save_pretrained(str(output_path))
                self.tokenizer.save_pretrained(str(output_path))
                logger.info(f"  ✓ Merged model saved")
                
            else:
                raise ValueError(f"Unknown save method: {save_method}")
            
            logger.info(f"[{self.job_id}] ✓ Model saved successfully")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"[{self.job_id}] ✗ Failed to save model: {str(e)}")
            raise RuntimeError(f"Model saving failed: {str(e)}")
    
    def cleanup(self):
        """Clean up GPU memory after training"""
        logger.info(f"[{self.job_id}] Cleaning up resources...")
        
        try:
            # Delete model and trainer
            del self.model
            del self.trainer
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info(f"[{self.job_id}] ✓ Cleanup completed")
            
        except Exception as e:
            logger.warning(f"[{self.job_id}] Cleanup warning: {str(e)}")
    
    def stop_training(self):
        """Request to stop training gracefully"""
        logger.info(f"[{self.job_id}] Stop requested")
        self.should_stop = True
        if self.trainer:
            self.trainer.control.should_training_stop = True
