# Data Directory

This directory contains datasets and test data for the LLM fine-tuning demo.

## Data Files

Available training datasets:

- **`shakespeare_data.txt`** - Shakespeare text dataset for training
- **`washingmachine_data.txt`** - Washing machine manual dataset
- **`eminen.txt`** - Eminem lyrics dataset
- **`letranger.txt`** - Additional text dataset

## Test Files

- **`test_prompts.json`** - Sample prompts for testing models

## Usage

To switch between datasets, edit the `data_file_name` variable at the top of `config.py`:

```python
# Change this line in config.py
data_file_name = "data/shakespeare_data.txt"  # or any other file in this directory
```

The model output folder will automatically be named based on your data file (e.g., `fine_tuned_shakespeare_data_model` for `shakespeare_data.txt`).

