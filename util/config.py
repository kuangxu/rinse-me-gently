"""
Configuration module for LLM Fine-Tuning Demo
Contains all hyperparameters and settings for the fine-tuning process
"""

from dataclasses import dataclass
from typing import List, Optional


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
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["c_attn", "c_proj"]


@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    output_dir: str = "./shakespeare_model"
    num_train_epochs: int = 1  # Single epoch for 3-minute training
    per_device_train_batch_size: int = 8  # Larger batch size for faster training
    learning_rate: float = 5e-4  # Higher learning rate for faster convergence
    logging_steps: int = 10  # Less frequent logging to reduce overhead
    save_strategy: str = "no"  # Don't save checkpoints for demo
    remove_unused_columns: bool = False
    warmup_steps: int = 50  # More warmup steps for stability with high LR
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0  # Gradient clipping
    evaluation_strategy: str = "no"  # No evaluation for demo
    dataloader_num_workers: int = 0  # Disable multiprocessing for stability
    fp16: bool = True  # Enable mixed precision for faster training


@dataclass
class DataConfig:
    """Data configuration parameters"""
    # Shakespeare dataset settings
    data_source: str = "shakespeare"  # Fixed to Shakespeare only
    custom_data_file: str = "data/shakespeare_data.txt"  # Fixed Shakespeare data file
    data_format: str = "plain_text"  # Plain text format for Shakespeare
    max_samples: int = 5000  # Limit samples for faster training
    min_length: int = 20  # Minimum text length to filter out short lines
    max_length: int = 256  # Maximum text length for efficient training


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
        print(f"  Epochs: {self.training.num_train_epochs}")
        print(f"  Batch Size: {self.training.per_device_train_batch_size}")
        print(f"  Learning Rate: {self.training.learning_rate}")
        print(f"  Logging Steps: {self.training.logging_steps}")
        
        print(f"\n📁 Data Configuration:")
        print(f"  Data Source: {self.data.data_source}")
        print(f"  Custom File: {self.data.custom_data_file}")
        print(f"  Max Samples: {self.data.max_samples}")
        print(f"  Min Length: {self.data.min_length}")
        print(f"  Max Length: {self.data.max_length}")
        
        print(f"\n🧪 Evaluation Configuration:")
        print(f"  Test Prompts: {len(self.evaluation.test_prompts)}")
        print(f"  Max Tokens: {self.evaluation.max_new_tokens}")
        print(f"  Temperature: {self.evaluation.temperature}")


# Create global config instance
config = Config()
