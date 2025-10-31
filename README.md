# Rince Me Gently 

## LLM Fine-Tuning Demo - Setup Guide

A focused demonstration of fine-tuning language models using LoRA (Low-Rank Adaptation). This project provides a clean, simplified interface for training and testing language models, specifically optimized for Shakespeare text generation.

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** installed on your system
- **Hardware**: Midrange laptop (Intel i7 or equivalent, 8-16GB RAM recommended). No GPU required.
- **Operating System**: Mac (macOS) or Windows

### Quick Reference (TL;DR)

**Mac (3 steps):**
```bash
python3 -m venv llm-workshop-env
source llm-workshop-env/bin/activate
pip install --upgrade pip && pip install torch torchvision torchaudio && pip install -r requirements.txt
python test_training_pipeline.py
```

**Windows (3 steps):**
```powershell
python -m venv llm-workshop-env
llm-workshop-env\Scripts\activate
python -m pip install --upgrade pip; pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; pip install -r requirements.txt
python test_training_pipeline.py
```

After setup, test models with `python demo_model.py --use-raw` or run the training pipeline tests. For detailed setup instructions and troubleshooting, see sections below.

### 1. Environment Setup

#### For Mac Users

**Option A: Automated Setup Script (Recommended)**
```bash
# Make script executable (if needed)
chmod +x setup_environment.sh

# Run the setup script
./setup_environment.sh
```

The script will:
- Detect if you have conda, Homebrew, or Python venv
- Automatically create a virtual environment
- Install all required packages
- Provide instructions for activation

**Option B: Manual Setup with Python venv**
```bash
# Create virtual environment
python3 -m venv llm-workshop-env

# Activate virtual environment
source llm-workshop-env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU version for Mac)
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt
```

**Option C: Using Conda (if you have Anaconda/Miniconda)**
```bash
# Create conda environment
conda create -n llm-workshop python=3.10 -y

# Activate environment
conda activate llm-workshop

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

#### For Windows Users

**Option A: Using Python venv (Recommended)**
```powershell
# Create virtual environment
python -m venv llm-workshop-env

# Activate virtual environment
llm-workshop-env\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch (CPU version for Windows)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

**Option B: Using Conda (if you have Anaconda/Miniconda)**
```powershell
# Create conda environment
conda create -n llm-workshop python=3.10 -y

# Activate environment
conda activate llm-workshop

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

### 2. Activate Environment

**Mac:**
```bash
source llm-workshop-env/bin/activate
```

**Windows:**
```powershell
llm-workshop-env\Scripts\activate
```

### 3. Test Models

Once your environment is activated, you can test models:

```bash
# Test with prompts
python demo_model.py --model-path ./my_fine_tuned_model

# Test + interactive chat
python demo_model.py --model-path ./my_fine_tuned_model --interactive

# Use the raw (unfine-tuned) base model
python demo_model.py --use-raw
```

## 📁 What's Included

- **`demo_model.py`** - Test and chat with saved models
- **`test_training_pipeline.py`** - Test training pipeline and all modules
- **`data/`** - Data files folder
  - **`shakespeare_data.txt`** - Shakespeare text dataset for training
  - **`test_prompts.json`** - Sample prompts for testing
- **`util/`** - Utility modules folder
  - **`config.py`** - Configuration settings
  - **`model_utils.py`** - Model loading and LoRA setup
  - **`data_utils.py`** - Dataset handling
  - **`training_utils.py`** - Training execution
  - **`evaluation_utils.py`** - Model testing
- **`requirements.txt`** - All Python dependencies
- **`setup_environment.sh`** - Automated setup script (Mac/Linux)

## 📚 Usage Options

### Option A: Test Existing Model

```bash
# Test with prompts
python demo_model.py --model-path ./my_fine_tuned_model

# Test + interactive chat
python demo_model.py --model-path ./my_fine_tuned_model --interactive

# Use the raw (unfine-tuned) base model
python demo_model.py --use-raw
```

### Option B: Test Training Pipeline First

```bash
python test_training_pipeline.py
```

## ⚙️ Customization

Edit `util/config.py` to change settings:

```python
# Model settings
config.model.model_name = "distilgpt2"  # Change model
config.model.max_length = 128           # Adjust text length

# Training settings  
config.training.num_train_epochs = 1    # More epochs = better results
config.training.learning_rate = 5e-4    # Learning speed
config.training.output_dir = "./my_fine_tuned_model"  # Save location

# Data settings
config.data.max_samples = 5000          # Use more/fewer examples
config.data.min_length = 20             # Filter short lines
config.data.max_length = 256            # Max text length
```

## 🔧 Model Options

The default model is `distilgpt2` (82M parameters), which is fast and works well on laptops. If you want to experiment with larger models:

**Smaller/Faster Models:**
- `distilgpt2` (default, 82M params) - ~1-3 minutes for 50 examples
- `gpt2` (124M params) - ~2-5 minutes

**Larger Models (require more RAM/time):**
- `openai-community/gpt2-medium` (355M params) - ~5-10 minutes with quantization
- `HuggingFaceTB/SmolLM-360M` (360M params) - ~3-7 minutes

**Very Large Models (may require 16GB+ RAM):**
- `HuggingFaceH4/zephyr-7b-beta` (7B params, quantized) - ~20-40 minutes

**Note**: For models larger than 1B parameters, consider using Google Colab with free GPU for faster training.

## 🐛 Troubleshooting

### Package Installation Issues

**Mac:**
- If pip fails, try: `pip install --upgrade pip` first
- For permission errors, use: `pip install --user <package>`
- If using M1/M2 Mac, PyTorch should work natively (MPS support available)

**Windows:**
- Ensure Python is added to PATH during installation
- Use `python -m pip` instead of just `pip` if you get command not found errors
- If you see "pip is not recognized", reinstall Python with "Add to PATH" option

### Out of Memory Errors

- Reduce batch size in `config.py`: `config.training.per_device_train_batch_size = 1`
- Use fewer examples: reduce `config.data.max_samples`
- Reduce text length: `config.model.max_length = 64`
- For Windows with low RAM, close other applications

### Slow Training

- Reduce examples: `config.data.max_samples = 20` or lower
- Reduce text length: `config.model.max_length = 64`
- Use fewer epochs: `config.training.num_train_epochs = 1`

### Network Issues / Download Failures

- Models download from Hugging Face automatically on first use
- If downloads fail, check internet connection
- Large models may take time - be patient
- For offline use, download models manually and place in local cache

### Environment Activation Issues

**Mac:**
- If activation fails: `source llm-workshop-env/bin/activate`
- Check Python version: `python3 --version` (needs 3.10+)

**Windows:**
- Use PowerShell or Command Prompt (not Git Bash for activation)
- Path: `llm-workshop-env\Scripts\activate`
- If blocked by execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Import Errors

- Ensure virtual environment is activated (you should see `(llm-workshop-env)` in terminal prompt)
- Reinstall dependencies: `pip install -r requirements.txt`
- Check Python version matches: `python --version`

## 📝 Expected Training Times

Times are estimates for 50 examples, 1 epoch on midrange CPU (i7, 16GB RAM):

- **distilgpt2**: 30 seconds - 3 minutes
- **gpt2**: 2-5 minutes
- **SmolLM-360M**: 3-7 minutes
- **gpt2-medium (quantized)**: 5-10 minutes

If training is slow, reduce the number of examples or text length.

## 🎓 Getting Started Checklist

- [ ] Python 3.10+ installed and working
- [ ] Virtual environment created and activated
- [ ] All packages installed (`pip install -r requirements.txt`)
- [ ] Test training pipeline runs: `python test_training_pipeline.py`
- [ ] Ready to test models: `python demo_model.py --use-raw`

## 📖 Next Steps

1. **Test models**: Use `python demo_model.py --model-path ./my_fine_tuned_model` to test fine-tuned models
2. **Interactive chat**: Run `python demo_model.py --model-path ./my_fine_tuned_model --interactive` for interactive testing
3. **Experiment**: Try different models, hyperparameters, or data by editing `util/config.py`
4. **Customize**: Modify the data files in `data/` or config settings in `util/config.py` to use your own data

---

**Need more help?** Check the code comments in the Python scripts for detailed explanations.
