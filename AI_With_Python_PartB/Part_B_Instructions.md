# Rinse Me Gently 

## LLM Fine-Tuning Demo - Setup Guide

A focused demonstration of fine-tuning language models using LoRA (Low-Rank Adaptation). This project provides a clean, simplified interface for training and testing language models with various text datasets.

### Transitioning from Part A

If you completed **Part A**, you're already set up! Here's what you need to know:

- **You already have a virtual environment** at the root of the project (the folder containing both `AI_With_Python_PartA` and `AI_With_Python_PartB`)
- **You're already using Cursor** (or VS Code) - keep using the same setup
- **You just need to** activate your existing venv and install Part B's additional requirements

**Quick start for Part A students:**
1. Open Cursor with the **root folder** (same as Part A - should contain both `AI_With_Python_PartA` and `AI_With_Python_PartB`)
2. Open Terminal in Cursor
3. Activate your existing venv: `source venv/bin/activate` (Mac) or `.\venv\Scripts\activate` (Windows)
4. Navigate to Part B: `cd AI_With_Python_PartB`
5. Install requirements: `pip install -r requirements.txt` (after upgrading pip and installing PyTorch - see detailed instructions below)

If you're starting fresh (skipped Part A), follow the full setup instructions below.

## Quick Start

### Prerequisites

- **Python 3.10+** installed on your system
- **Hardware**: Midrange laptop (Intel i7 or equivalent, 8-16GB RAM recommended). No GPU required.
- **Operating System**: Mac (macOS) or Windows

### Quick Reference (TL;DR)

**If you already completed Part A:**
You should already have a virtual environment set up at the root of the project (the folder containing both `AI_With_Python_PartA` and `AI_With_Python_PartB`). Simply activate it and install Part B requirements:

**Mac:**
```bash
# Make sure you're in the root folder (where venv folder is located)
source venv/bin/activate
cd AI_With_Python_PartB
pip install --upgrade pip && pip install torch torchvision torchaudio && pip install -r requirements.txt
python test_training_pipeline.py
```

**Windows:**
```powershell
# Make sure you're in the root folder (where venv folder is located)
.\venv\Scripts\activate
cd AI_With_Python_PartB
python -m pip install --upgrade pip; pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; pip install -r requirements.txt
python test_training_pipeline.py
```

**If starting fresh (skipping Part A):**
**Mac (3 steps):**
```bash
python3 -m venv venv
source venv/bin/activate
cd AI_With_Python_PartB
pip install --upgrade pip && pip install torch torchvision torchaudio && pip install -r requirements.txt
python test_training_pipeline.py
```

**Windows (3 steps):**
```powershell
python -m venv venv
.\venv\Scripts\activate
cd AI_With_Python_PartB
python -m pip install --upgrade pip; pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; pip install -r requirements.txt
python test_training_pipeline.py
```

After setup, test models with `python demo_model.py --use-raw` or run the training pipeline tests. For detailed setup instructions and troubleshooting, see sections below.

### 1. Environment Setup

**Important:** If you completed Part A, you should already have a virtual environment set up at the root of the project (the folder containing both `AI_With_Python_PartA` and `AI_With_Python_PartB`). The venv folder should be at the same level as these two folders, not inside either of them. Skip to "If you completed Part A" below. If you're starting fresh, follow the setup instructions.

**Note:** You should still be using Cursor (or VS Code) from Part A. Make sure you have the root folder open in Cursor, just like you did in Part A.

#### For Mac Users

**If you completed Part A (Recommended):**
```bash
# Make sure you're in the root folder (where venv folder is located)
# If you're currently in AI_With_Python_PartA or AI_With_Python_PartB, navigate up:
cd ..

# Verify you're in the right place - you should see both AI_With_Python_PartA and AI_With_Python_PartB folders
# You should also see the venv folder here

# Activate the existing virtual environment
source venv/bin/activate

# You should see (venv) at the start of your terminal prompt
# If not, the activation didn't work - check that you're in the root folder

# Navigate to Part B folder
cd AI_With_Python_PartB

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU version for Mac)
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt
```

**If starting fresh:**

**Option A: Automated Setup Script**
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
# Navigate to root folder (parent directory)
cd ..

# Create virtual environment at root (shared for both parts)
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Navigate to Part B folder
cd AI_With_Python_PartB

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU version for Mac)
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt
```

**Option C: Using Conda (if you have Anaconda/Miniconda)**
*Note: If you completed Part A with a venv, we recommend using that same venv for consistency. Only use conda if you're starting fresh and prefer conda.*

```bash
# Create conda environment
conda create -n llm-workshop python=3.10 -y

# Activate environment
conda activate llm-workshop

# Navigate to Part B folder
cd AI_With_Python_PartB

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

#### For Windows Users

**If you completed Part A (Recommended):**
```powershell
# Make sure you're in the root folder (where venv folder is located)
# If you're currently in AI_With_Python_PartA or AI_With_Python_PartB, navigate up:
cd ..

# Verify you're in the right place - you should see both AI_With_Python_PartA and AI_With_Python_PartB folders
# You should also see the venv folder here

# Activate the existing virtual environment
.\venv\Scripts\activate

# You should see (venv) at the start of your terminal prompt
# If not, the activation didn't work - check that you're in the root folder

# Navigate to Part B folder
cd AI_With_Python_PartB

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch (CPU version for Windows)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

**If starting fresh:**

**Option A: Using Python venv (Recommended)**
```powershell
# Navigate to root folder (parent directory)
cd ..

# Create virtual environment at root (shared for both parts)
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Navigate to Part B folder
cd AI_With_Python_PartB

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch (CPU version for Windows)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

**Option B: Using Conda (if you have Anaconda/Miniconda)**
*Note: If you completed Part A with a venv, we recommend using that same venv for consistency. Only use conda if you're starting fresh and prefer conda.*

```powershell
# Create conda environment
conda create -n llm-workshop python=3.10 -y

# Activate environment
conda activate llm-workshop

# Navigate to Part B folder
cd AI_With_Python_PartB

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

### 2. Activate Environment

**Important:** The virtual environment should be at the root of the project (not inside `AI_With_Python_PartB`). If you completed Part A, the venv is already set up there at the root level, alongside both `AI_With_Python_PartA` and `AI_With_Python_PartB` folders.

**How to verify you're in the right place:**
- The root folder should contain: `venv/`, `AI_With_Python_PartA/`, and `AI_With_Python_PartB/`
- If you only see `AI_With_Python_PartB/` and no `venv/` folder, you need to navigate up one level

**Mac:**
```bash
# From root folder (where venv folder is located)
source venv/bin/activate

# You should see (venv) at the start of your prompt
# Then navigate to Part B folder
cd AI_With_Python_PartB

# Or if you're already in Part B folder:
cd ..
source venv/bin/activate
cd AI_With_Python_PartB
```

**Windows:**
```powershell
# From root folder (where venv folder is located)
.\venv\Scripts\activate

# You should see (venv) at the start of your prompt
# Then navigate to Part B folder
cd AI_With_Python_PartB

# Or if you're already in Part B folder:
cd ..
.\venv\Scripts\activate
cd AI_With_Python_PartB
```

### 3. Verify Setup (Optional but Recommended)

After activating your environment and installing packages, you can verify everything is working:

```bash
# Make sure you're in AI_With_Python_PartB folder and venv is activated
# Check Python version (should show Python 3.10+)
python --version

# Verify PyTorch is installed
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

If you see the Python version and PyTorch version printed, you're ready to proceed!

### 4. Test Models

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

## What's Included

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

## Usage Options

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

## Customization

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

## Model Options

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

## Troubleshooting

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
- If activation fails, make sure you're in the root folder (where `venv` folder is located)
- Try: `source venv/bin/activate`
- Check Python version: `python3 --version` (needs 3.10+)
- If you don't see `(venv)` in your prompt, the activation didn't work - navigate to root folder first

**Windows:**
- Use PowerShell or Command Prompt (not Git Bash for activation)
- Make sure you're in the root folder (where `venv` folder is located)
- Path: `.\venv\Scripts\activate` (note the `.\` prefix)
- If blocked by execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- If you don't see `(venv)` in your prompt, the activation didn't work - navigate to root folder first

### Import Errors

- Ensure virtual environment is activated (you should see `(venv)` in terminal prompt)
- Reinstall dependencies: `pip install -r requirements.txt`
- Check Python version matches: `python --version`

### Training Warnings (Fixed Automatically)

- **MPS pin_memory warning**: The code automatically disables `pin_memory` on Apple Silicon (MPS) devices - this is normal and safe to ignore if you see it in older code
- **PEFT loss_type warning**: The LoRA configuration now explicitly sets `task_type="CAUSAL_LM"` to prevent warnings about unrecognized loss types
- Both warnings have been fixed in the latest version of the code

## Expected Training Times

Times are estimates for 50 examples, 1 epoch on midrange CPU (i7, 16GB RAM):

- **distilgpt2**: 30 seconds - 3 minutes
- **gpt2**: 2-5 minutes
- **SmolLM-360M**: 3-7 minutes
- **gpt2-medium (quantized)**: 5-10 minutes

If training is slow, reduce the number of examples or text length.

## Getting Started Checklist

**If you completed Part A:**
- [ ] Virtual environment activated (you see `(venv)` in terminal prompt)
- [ ] Navigated to `AI_With_Python_PartB` folder
- [ ] Installed Part B requirements (`pip install -r requirements.txt`)
- [ ] PyTorch installed successfully
- [ ] Test training pipeline runs: `python test_training_pipeline.py`
- [ ] Ready to test models: `python demo_model.py --use-raw`

**If starting fresh:**
- [ ] Python 3.10+ installed and working
- [ ] Virtual environment created at root folder and activated
- [ ] All packages installed (`pip install -r requirements.txt`)
- [ ] Test training pipeline runs: `python test_training_pipeline.py`
- [ ] Ready to test models: `python demo_model.py --use-raw`

## Next Steps

1. **Test models**: Use `python demo_model.py --model-path ./fine_tuned_shakespeare_data_model` to test fine-tuned models (or your auto-generated folder name based on the data file used)
2. **Interactive chat**: Run `python demo_model.py --model-path ./fine_tuned_shakespeare_data_model --interactive` for interactive testing
3. **Change training data**: Edit `config.py` top line: `data_file_name = "data/washingmachine_data.txt"` to use different data
4. **Experiment**: Try different models, hyperparameters, or data by editing `config.py`
5. **Customize**: Add your own data files to `data/` directory or modify config settings in `config.py`

---

**Need more help?** Check the code comments in the Python scripts for detailed explanations.
