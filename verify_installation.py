"""
Verify that all ML dependencies are installed correctly
"""

print("=" * 70)
print("🔍 Verifying ML Dependencies Installation")
print("=" * 70)

# Test 1: PyTorch + CUDA
try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"❌ PyTorch: {e}")

# Test 2: Transformers
try:
    import transformers
    print(f"✅ Transformers {transformers.__version__}")
except Exception as e:
    print(f"❌ Transformers: {e}")

# Test 3: PEFT (LoRA)
try:
    import peft
    print(f"✅ PEFT {peft.__version__}")
except Exception as e:
    print(f"❌ PEFT: {e}")

# Test 4: Accelerate
try:
    import accelerate
    print(f"✅ Accelerate {accelerate.__version__}")
except Exception as e:
    print(f"❌ Accelerate: {e}")

# Test 5: BitsAndBytes
try:
    import bitsandbytes
    print(f"✅ BitsAndBytes {bitsandbytes.__version__}")
except Exception as e:
    print(f"❌ BitsAndBytes: {e}")

# Test 6: Unsloth (CRITICAL!)
try:
    from unsloth import FastLanguageModel
    print(f"✅ Unsloth INSTALLED!")
except Exception as e:
    print(f"❌ Unsloth: {e}")

# Test 7: Datasets
try:
    import datasets
    print(f"✅ Datasets {datasets.__version__}")
except Exception as e:
    print(f"❌ Datasets: {e}")

print("=" * 70)
print("🎯 Installation Status")
print("=" * 70)

try:
    import torch
    from unsloth import FastLanguageModel
    import peft
    import transformers
    
    if torch.cuda.is_available():
        print("✅ ALL DEPENDENCIES INSTALLED SUCCESSFULLY!")
        print("✅ GPU Detected and Ready!")
        print("")
        print("🚀 You can now run REAL MODEL TRAINING!")
        print("")
        print("Next steps:")
        print("1. Start backend: cd backend && python -m uvicorn main:app --reload")
        print("2. Start frontend: cd frontend && python app.py")
        print("3. Visit: http://localhost:7860")
    else:
        print("⚠️  Dependencies installed but no GPU detected")
        print("   Training will be VERY slow without GPU")
except:
    print("❌ Some dependencies missing. Re-run installation.")

print("=" * 70)
