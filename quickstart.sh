#!/bin/bash

# ============================================================================
# Gemma LoRA Fine-Tuner - Quick Start Script
# ============================================================================

echo "=========================================="
echo "Gemma LoRA Fine-Tuner - Quick Start"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "⚠️  Please do not run this script as root"
    exit 1
fi

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version detected"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip -q
echo "✓ Pip upgraded"

# Install dependencies
echo ""
echo "Installing dependencies (this may take 10-15 minutes)..."
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Setup environment file
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created"
fi

# Create directories
echo ""
echo "Creating necessary directories..."
mkdir -p datasets models exports logs temp
echo "✓ Directories created"

# Check CUDA
echo ""
echo "Checking CUDA availability..."
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "To start the application:"
echo ""
echo "1. Start Backend (Terminal 1):"
echo "   cd backend"
echo "   uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "2. Start Frontend (Terminal 2):"
echo "   cd frontend"
echo "   python app.py"
echo ""
echo "3. Open your browser:"
echo "   Frontend: http://localhost:7860"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "=========================================="
