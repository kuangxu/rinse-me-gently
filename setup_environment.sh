#!/bin/bash

# LLM Fine-Tuning Workshop Setup Script
# This script sets up the environment for the MBA workshop

echo "🚀 Setting up LLM Fine-Tuning Workshop Environment"
echo "=================================================="

# Function to check if conda is available
check_conda() {
    if command -v conda &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to install conda via Homebrew
install_conda_brew() {
    echo "📦 Installing Anaconda via Homebrew..."
    if command -v brew &> /dev/null; then
        echo "Installing Anaconda (this may require sudo password)..."
        brew install --cask anaconda
        if [ $? -eq 0 ]; then
            echo "✅ Anaconda installed successfully!"
            echo "Please restart your terminal or run: source ~/.zshrc"
            echo "Then run this script again."
            return 0
        else
            echo "❌ Failed to install Anaconda via Homebrew"
            return 1
        fi
    else
        echo "❌ Homebrew not found. Please install Homebrew first:"
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        return 1
    fi
}

# Function to install conda manually
install_conda_manual() {
    echo "📦 Manual Anaconda installation required:"
    echo ""
    echo "Option 1: Download Anaconda"
    echo "   1. Go to: https://www.anaconda.com/download"
    echo "   2. Download Anaconda for macOS"
    echo "   3. Run the installer"
    echo "   4. Restart your terminal"
    echo "   5. Run this script again"
    echo ""
    echo "Option 2: Install Miniconda (lighter version)"
    echo "   1. Go to: https://docs.conda.io/en/latest/miniconda.html"
    echo "   2. Download Miniconda for macOS"
    echo "   3. Run: bash Miniconda3-latest-MacOSX-arm64.sh"
    echo "   4. Restart your terminal"
    echo "   5. Run this script again"
    echo ""
    echo "Option 3: Use Python virtual environment (alternative)"
    echo "   This script will fall back to using Python venv if conda is not available."
    echo ""
    read -p "Press Enter to continue with Python venv fallback, or Ctrl+C to install Anaconda first..."
}

# Function to setup with Python venv
setup_python_venv() {
    echo "🐍 Setting up with Python virtual environment..."
    
    # Check if Python 3.10+ is available
    if ! python3 --version | grep -E "Python 3\.(1[0-9]|[2-9][0-9])" &> /dev/null; then
        echo "❌ Python 3.10+ required. Please install Python 3.10 or later."
        echo "   Download from: https://www.python.org/downloads/"
        exit 1
    fi
    
    echo "✅ Python 3.10+ found"
    
    # Create virtual environment
    echo "📦 Creating virtual environment 'llm-workshop'..."
    python3 -m venv llm-workshop-env
    
    # Activate environment
    echo "🔄 Activating environment..."
    source llm-workshop-env/bin/activate
    
    # Upgrade pip
    echo "⬆️ Upgrading pip..."
    pip install --upgrade pip
    
    # Install PyTorch (CPU version)
    echo "🔥 Installing PyTorch (CPU version)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    # Install other requirements
    echo "📚 Installing other requirements..."
    pip install -r requirements.txt
    
    echo ""
    echo "✅ Setup complete!"
    echo ""
    echo "To start the workshop:"
    echo "1. source llm-workshop-env/bin/activate"
    echo "2. jupyter notebook"
    echo "3. Open the 'LLM_Fine_Tuning_Demo.ipynb' notebook"
    echo ""
    echo "To deactivate the environment later: deactivate"
    echo "Happy learning! 🎓"
}

# Main setup logic
if check_conda; then
    echo "✅ Conda found"
    
    # Create virtual environment
    echo "📦 Creating virtual environment 'llm-workshop'..."
    conda create -n llm-workshop python=3.10 -y
    
    # Activate environment
    echo "🔄 Activating environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate llm-workshop
    
    # Install PyTorch (CPU version)
    echo "🔥 Installing PyTorch (CPU version)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    # Install other requirements
    echo "📚 Installing other requirements..."
    pip install -r requirements.txt
    
    echo ""
    echo "✅ Setup complete!"
    echo ""
    echo "To start the workshop:"
    echo "1. conda activate llm-workshop"
    echo "2. jupyter notebook"
    echo "3. Open the 'LLM_Fine_Tuning_Demo.ipynb' notebook"
    echo ""
    echo "Happy learning! 🎓"
    
else
    echo "❌ Conda not found"
    echo ""
    echo "Choose an installation method:"
    echo "1. Install Anaconda via Homebrew (recommended)"
    echo "2. Manual Anaconda installation"
    echo "3. Use Python virtual environment (fallback)"
    echo ""
    read -p "Enter your choice (1-3): " choice
    
    case $choice in
        1)
            install_conda_brew
            ;;
        2)
            install_conda_manual
            ;;
        3)
            setup_python_venv
            ;;
        *)
            echo "Invalid choice. Using Python virtual environment fallback..."
            setup_python_venv
            ;;
    esac
fi
