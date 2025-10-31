"""
Configuration module for LLM Fine-Tuning Demo
Contains all hyperparameters and settings for the fine-tuning process

QUICK START - Change Training Data:
==================================
To use a different training data file, edit line 74 in this file:
    data_file: str = "data/shakespeare_data.txt"
    
Change to:
    data_file: str = "data/washingmachine_data.txt"  # or your own file
    
Or modify at runtime:
    from config import config
    config.data.data_file = "data/your_file.txt"
"""

from dataclasses import dataclass
from typing import List, Optional
import os


data_file_name = "data/washingmachine_data.txt"


@dataclass
class ModelConfig:
    """Model configuration parameters"""
    model_name: str = "distilgpt2"
    max_length: int = 128
    pad_token: Optional[str] = None  # Will be set to eos_token


@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration parameters"""
    r: int = 8  # Rank - how many parameters to update
    lora_alpha: int = 16  # Scaling factor
    target_modules: List[str] = None  # Which parts to update
    lora_dropout: float = 0.1  # Regularization
    bias: str = "none"  # Don't update bias terms
    task_type: str = "CAUSAL_LM"  # Task type for PEFT - prevents loss_type warning
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["c_attn", "c_proj"]


@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    # Model output directory - will be auto-generated from data file name if None
    # Format: "./fine_tuned_{data_filename}_model" (e.g., "./fine_tuned_shakespeare_data_model")
    output_dir: Optional[str] = None  # Auto-generated from data file if None
    
    # Prefix for auto-generated model folder names
    model_folder_prefix: str = "fine_tuned_"
    model_folder_suffix: str = "_model"
    
    # ============================================
    # TRAINING DURATION CONTROL
    # ============================================
    # Option 1: Use epochs (number of full passes through dataset)
    num_train_epochs: int = 1  # Single epoch for 3-minute training
    
    # Option 2: Use max_steps (maximum number of training steps/iterations)
    # If max_steps is set (not None), it will override num_train_epochs
    # Note: This is DIFFERENT from max_samples in DataConfig!
    #   - max_samples: Limits dataset size (how much data to load)
    #   - max_steps: Limits training duration (how many iterations to run)
    # Example: max_steps = 100  # Train for exactly 100 steps
    max_steps: Optional[int] = 200  # Set to a number to limit training steps
    
    # ============================================
    # TRAINING HYPERPARAMETERS
    # ============================================
    per_device_train_batch_size: int = 8  # Larger batch size for faster training
    learning_rate: float = 4e-4  # Much lower LR to prevent gradient explosion
    logging_steps: int = 10  # Less frequent logging to reduce overhead
    save_strategy: str = "epoch"  # Save model at end of training
    remove_unused_columns: bool = False
    warmup_steps: int = 10  # Reduced warmup with lower LR
    weight_decay: float = 0.01
    max_grad_norm: float = 5.0  # Higher gradient clipping threshold
    evaluation_strategy: str = "no"  # No evaluation for demo
    save_total_limit: int = 1  # Keep only the final model
    dataloader_num_workers: int = 0  # Disable multiprocessing for stability
    fp16: bool = False  # Disable fp16 - causing NaN gradients with short sequences


@dataclass
class DataConfig:
    """Data configuration parameters
    
    To use a different training data file, simply change the `data_file` parameter:
    
    Examples:
        # Use Shakespeare data (default)
        data_file = "data/shakespeare_data.txt"
        
        # Use washing machine data
        data_file = "data/washingmachine_data.txt"
        
        # Use your own custom file
        data_file = "data/my_custom_data.txt"
    """
    # ============================================
    # TRAINING DATA FILE CONFIGURATION
    # ============================================
    # 🔧 CHANGE THIS to use a different training data file
    # Available preset files in data/ directory:
    #   - "data/shakespeare_data.txt" (default)
    #   - "data/washingmachine_data.txt"
    #   - "data/your_custom_file.txt" (add your own)
    data_file: str = data_file_name
    
    # Legacy parameter name (for backward compatibility with existing code)
    # This will automatically mirror data_file
    custom_data_file: str = data_file_name
    
    # ============================================
    # DATA PROCESSING PARAMETERS
    # ============================================
    data_source: str = "custom"  # Auto-detected from filename (shakespeare, washingmachine, custom)
    data_format: str = "plain_text"  # Format: "plain_text" or "instruction_response"
    
    # Dataset size limit (controls how much data is loaded from file)
    # Note: This is DIFFERENT from max_steps in TrainingConfig!
    #   - max_samples: Limits dataset size (e.g., use only first 5000 lines from file)
    #   - max_steps: Limits training duration (e.g., train for exactly 100 steps)
    # Example: max_samples=5000, batch_size=10 → 500 steps per epoch
    #          If max_steps=100, training stops after 100 steps (0.2 epochs)
    max_samples: int = 5000  # Limit dataset size for faster training (set to None for all)
    
    min_length: int = 5  # Minimum text length to filter out short lines
    max_length: int = 512  # Maximum text length for data filtering (chars, not tokens!)
    
    def __post_init__(self):
        """Initialize and auto-detect data source from filename"""
        # Sync custom_data_file with data_file for backward compatibility
        self.custom_data_file = self.data_file
        
        # Auto-detect data_source based on file name for convenience
        if "shakespeare" in self.data_file.lower():
            self.data_source = "shakespeare"
        elif "washingmachine" in self.data_file.lower() or "washing" in self.data_file.lower():
            self.data_source = "washingmachine"
        else:
            self.data_source = "custom"


@dataclass
class EvaluationConfig:
    """Evaluation configuration parameters"""
    test_prompts: List[str] = None
    max_new_tokens: int = 50
    temperature: float = 0.7
    do_sample: bool = True
    
    def __post_init__(self):
        if self.test_prompts is None:
            self.test_prompts = [
                "To be or not to be, that is the ",
                "Romeo, Romeo, wherefore art thou ",
                "All the world's a stage, and all the men and women merely ",
                "What light through yonder window breaks? It is the ",
                "Double, double toil and trouble; fire burn and ",
            ]


@dataclass
class VisualizationConfig:
    """Visualization configuration parameters"""
    figure_size: tuple = (10, 6)
    line_width: int = 2
    grid_alpha: float = 0.3
    show_plots: bool = True


# Global configuration instance
class Config:
    """Main configuration class containing all sub-configurations"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.lora = LoRAConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.evaluation = EvaluationConfig()
        self.visualization = VisualizationConfig()
        
        # Auto-generate output_dir from data file name if not set
        if self.training.output_dir is None:
            self.training.output_dir = self._generate_output_dir()
    
    def _generate_output_dir(self) -> str:
        """Generate output directory name from data file name"""
        # Extract filename from path (e.g., "data/shakespeare_data.txt" -> "shakespeare_data")
        data_file = self.data.data_file
        filename = os.path.basename(data_file)  # Get just the filename
        filename_no_ext = os.path.splitext(filename)[0]  # Remove extension
        
        # Generate folder name: prefix + filename + suffix
        # Example: "fine_tuned_" + "shakespeare_data" + "_model" = "fine_tuned_shakespeare_data_model"
        folder_name = f"{self.training.model_folder_prefix}{filename_no_ext}{self.training.model_folder_suffix}"
        
        # Return with "./" prefix for relative path
        return f"./{folder_name}"
    
    def get_output_dir(self) -> str:
        """Get the output directory, generating it if needed"""
        if self.training.output_dir is None:
            self.training.output_dir = self._generate_output_dir()
        return self.training.output_dir
    
    def print_config(self):
        """Print all configuration parameters in a readable format"""
        print("🔧 Configuration Summary:")
        print("=" * 50)
        
        print(f"\n📊 Model Configuration:")
        print(f"  Model: {self.model.model_name}")
        print(f"  Max Length: {self.model.max_length}")
        
        print(f"\n🔧 LoRA Configuration:")
        print(f"  Rank (r): {self.lora.r}")
        print(f"  Alpha: {self.lora.lora_alpha}")
        print(f"  Target Modules: {self.lora.target_modules}")
        print(f"  Dropout: {self.lora.lora_dropout}")
        
        print(f"\n🚀 Training Configuration:")
        print(f"  Output Directory: {self.get_output_dir()}")
        if self.training.max_steps is not None:
            print(f"  Max Steps: {self.training.max_steps} (overrides epochs)")
        else:
            print(f"  Epochs: {self.training.num_train_epochs}")
        print(f"  Batch Size: {self.training.per_device_train_batch_size}")
        print(f"  Learning Rate: {self.training.learning_rate}")
        print(f"  Logging Steps: {self.training.logging_steps}")
        print(f"  Warmup Steps: {self.training.warmup_steps}")
        
        print(f"\n📁 Data Configuration:")
        print(f"  Data File: {self.data.data_file}")  # 🔧 Change this to use different data!
        print(f"  Data Source: {self.data.data_source}")
        print(f"  Format: {self.data.data_format}")
        print(f"  Max Samples: {self.data.max_samples}")
        print(f"  Min Length: {self.data.min_length}")
        print(f"  Max Length: {self.data.max_length}")
        
        print(f"\n🧪 Evaluation Configuration:")
        print(f"  Test Prompts: {len(self.evaluation.test_prompts)}")
        print(f"  Max Tokens: {self.evaluation.max_new_tokens}")
        print(f"  Temperature: {self.evaluation.temperature}")


# Create global config instance
config = Config()

