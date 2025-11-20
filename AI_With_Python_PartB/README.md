# Rinse Me Gently 

## LLM Fine-Tuning Demo - Setup Guide

A focused demonstration of fine-tuning language models using LoRA (Low-Rank Adaptation). This project provides a clean, simplified interface for training and testing language models with various text datasets.

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** installed on your system
- **Hardware**: Midrange laptop (Intel i7 or equivalent, 8-16GB RAM recommended). No GPU required.
- **Operating System**: Mac (macOS) or Windows

### Quick Reference (TL;DR)

**Mac (3 steps):**
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip && pip install torch torchvision torchaudio && pip install -r requirements.txt
python test_training_pipeline.py
```

**Windows (3 steps):**
```powershell
python -m venv venv
venv\Scripts\activate
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
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

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
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

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
source venv/bin/activate
```

**Windows:**
```powershell
venv\Scripts\activate
```

### 3. Test Models

Once your environment is activated, you can test models:

```bash
# Test with prompts (folder name auto-generated from data file)
python demo_model.py --model-path ./fine_tuned_shakespeare_data_model

# Test + interactive chat
python demo_model.py --model-path ./fine_tuned_shakespeare_data_model --interactive

# Use the raw (unfine-tuned) base model
python demo_model.py --use-raw

# Test with washing machine model (if you trained on that data)
python demo_model.py --model-path ./fine_tuned_washingmachine_data_model
```

## 📁 What's Included

- **`config.py`** - Main configuration file (edit this to customize training!)
- **`demo_model.py`** - Test and chat with saved models
- **`test_training_pipeline.py`** - Test training pipeline and all modules
- **`data/`** - Data files folder
  - **`shakespeare_data.txt`** - Shakespeare text dataset for training
  - **`washingmachine_data.txt`** - Washing machine manual dataset
  - **`eminen.txt`** - Eminem lyrics dataset
  - **`letranger.txt`** - Additional text dataset
  - **`test_prompts.json`** - Sample prompts for testing
- **`util/`** - Utility modules folder
  - **`model_utils.py`** - Model loading and LoRA setup
  - **`data_utils.py`** - Dataset handling
  - **`training_utils.py`** - Training execution
  - **`evaluation_utils.py`** - Model testing
- **`requirements.txt`** - All Python dependencies
- **`setup_environment.sh`** - Automated setup script (Mac/Linux)

## 📚 Usage Options

### Option A: Test Existing Model

```bash
# Test with prompts (folder name auto-generated from data file)
python demo_model.py --model-path ./fine_tuned_shakespeare_data_model

# Test + interactive chat
python demo_model.py --model-path ./fine_tuned_shakespeare_data_model --interactive

# Use the raw (unfine-tuned) base model
python demo_model.py --use-raw
```

### Option B: Test Training Pipeline First

```bash
python test_training_pipeline.py
```

**Note**: After training completes, the fine-tuned model is automatically saved to the folder specified in `config.training.output_dir` (auto-generated from your data file name).

## ⚙️ Customization

### Quick Configuration (Edit `config.py`)

**Change Training Data File** (line ~100):
```python
# Easy way to switch between data files
# Available data files:
data_file_name = "data/shakespeare_data.txt"  # Shakespeare text
data_file_name = "data/washingmachine_data.txt"  # Washing machine manual
data_file_name = "data/eminen.txt"  # Eminem lyrics
data_file_name = "data/letranger.txt"  # Additional text
# Or your own:
data_file_name = "data/my_custom_data.txt"  # Your own data
```

**Training Duration** (line ~53-61):
```python
# Option 1: Use epochs (full passes through dataset)
num_train_epochs: int = 1

# Option 2: Use exact number of steps
max_steps: Optional[int] = 100  # Train for exactly 100 steps
```

**Model Output Location**:
- Auto-generated from data file name (default): `./fine_tuned_{data_filename}_model`
- Example: `data/shakespeare_data.txt` → `./fine_tuned_shakespeare_data_model`
- Or set explicitly: `config.training.output_dir = "./my_custom_folder"`

### Advanced Configuration

Edit `config.py` directly or modify at runtime:

```python
from config import config

# Model settings
config.model.model_name = "distilgpt2"  # Change model
config.model.max_length = 128           # Adjust text length

# Training settings  
config.training.num_train_epochs = 1    # More epochs = better results
config.training.max_steps = 100         # Or use exact step count
config.training.learning_rate = 5e-4    # Learning speed
config.training.per_device_train_batch_size = 8  # Batch size
# output_dir is auto-generated from data file name (or set explicitly)

# Data settings
config.data.data_file = "data/washingmachine_data.txt"  # Change data file
# Available: shakespeare_data.txt, washingmachine_data.txt, eminen.txt, letranger.txt
config.data.max_samples = 5000          # Use more/fewer examples
config.data.min_length = 20             # Filter short lines
config.data.max_length = 256            # Max text length (characters)
```

### Key Features

- **Auto-generated model folders**: Model saving folder automatically matches your data file name
- **Easy data switching**: Just change `data_file_name` at the top of `config.py` to use different training data
- **Flexible training duration**: Use epochs or exact step counts (`max_steps`)
- **Clear configuration**: All settings in one easy-to-edit file (`config.py`)
- **Optimized for Apple Silicon**: Automatically disables unsupported features (e.g., pin_memory on MPS)
- **Stable training**: Configured to prevent common issues like NaN gradients and padding token loss

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
- Reduce training steps: `config.training.max_steps = 50` (fewer steps)
- For Windows with low RAM, close other applications

### Slow Training

- Reduce examples: `config.data.max_samples = 20` or lower
- Reduce text length: `config.model.max_length = 64`
- Use fewer steps: `config.training.max_steps = 50` (faster than epochs)
- Or use fewer epochs: `config.training.num_train_epochs = 1`

### Network Issues / Download Failures

- Models download from Hugging Face automatically on first use
- If downloads fail, check internet connection
- Large models may take time - be patient
- For offline use, download models manually and place in local cache

### Environment Activation Issues

**Mac:**
- If activation fails: `source venv/bin/activate`
- Check Python version: `python3 --version` (needs 3.10+)

**Windows:**
- Use PowerShell or Command Prompt (not Git Bash for activation)
- Path: `venv\Scripts\activate`
- If blocked by execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Import Errors

- Ensure virtual environment is activated (you should see `(venv)` in terminal prompt)
- Reinstall dependencies: `pip install -r requirements.txt`
- Check Python version matches: `python --version`

### Training Warnings (Fixed Automatically)

- **MPS pin_memory warning**: The code automatically disables `pin_memory` on Apple Silicon (MPS) devices - this is normal and safe to ignore if you see it in older code
- **PEFT loss_type warning**: The LoRA configuration now explicitly sets `task_type="CAUSAL_LM"` to prevent warnings about unrecognized loss types
- Both warnings have been fixed in the latest version of the code

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

1. **Test models**: Use `python demo_model.py --model-path ./fine_tuned_shakespeare_data_model` to test fine-tuned models (or your auto-generated folder name based on the data file used)
2. **Interactive chat**: Run `python demo_model.py --model-path ./fine_tuned_shakespeare_data_model --interactive` for interactive testing
3. **Change training data**: Edit `config.py` top line: `data_file_name = "data/washingmachine_data.txt"` to use different data
4. **Experiment**: Try different models, hyperparameters, or data by editing `config.py`
5. **Customize**: Add your own data files to `data/` directory or modify config settings in `config.py`

---

**Need more help?** Check the code comments in the Python scripts for detailed explanations.
