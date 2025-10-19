#!/bin/bash
# Setup script for MVP Diabetes Project

echo "=========================================="
echo "MVP Diabetes Project Setup"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo ""
echo "[1/6] Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo ""
echo "[2/6] Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "[3/6] Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "[4/6] Installing dependencies..."
pip install -r requirements.txt

# Create directory structure
echo ""
echo "[5/6] Creating directory structure..."
python3 create_structure.py

# Setup environment file
echo ""
echo "[6/6] Setting up environment configuration..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✓ Created .env file from template"
    echo "⚠️  Please edit .env and add your API keys"
else
    echo "✓ .env file already exists"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env and add your OpenAI API key"
echo "2. Add nutrition databases to data/db/"
echo "3. Implement model wrappers in src/models/"
echo "4. Test with: python main.py data/raw/sample_meal.jpg"
echo ""
echo "To activate the environment in the future:"
echo "  source venv/bin/activate"
echo ""
