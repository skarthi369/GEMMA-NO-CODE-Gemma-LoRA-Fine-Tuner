# ============================================================================
# COMPLETE ML DEPENDENCIES INSTALLATION SCRIPT
# Run this after PyTorch is installed
# ============================================================================

# Step 1: Core ML Libraries (Already installing: PyTorch)
# torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Step 2: Transformers Ecosystem
pip install transformers==4.37.2
pip install tokenizers==0.15.0
pip install datasets==2.16.1

# Step 3: PEFT (LoRA) and Acceleration
pip install peft==0.7.1
pip install accelerate==0.26.1
pip install bitsandbytes==0.41.3.post2

# Step 4: Additional ML Dependencies
pip install scipy==1.11.4
pip install scikit-learn==1.3.0
pip install xgboost==2.0.0
pip install lightgbm==4.0.0

# Step 5: Monitoring and Logging
pip install tensorboard==2.15.1
pip install wandb==0.16.2
pip install loguru==0.7.2
pip install tqdm==4.66.1

# Step 6: GPU Monitoring
pip install psutil==5.9.7
pip install GPUtil==1.4.0
pip install py3nvml==0.2.7

# Step 7: Utilities
pip install humanize==4.9.0
pip install tenacity==8.2.3
pip install chardet==5.2.0

# Step 8: Unsloth (Last - may take time)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# ============================================================================
# TOTAL DOWNLOAD SIZE: ~5GB
# INSTALLATION TIME: 15-20 minutes
# ============================================================================
