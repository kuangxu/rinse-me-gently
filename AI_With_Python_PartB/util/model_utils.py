"""
Model utilities for LLM Fine-Tuning Demo
Handles model loading, LoRA configuration, and model setup
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, PeftModel
from config import config


class ModelManager:
    """Manages model loading, configuration, and setup"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.data_collator = None
        self.lora_config = None
        
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer"""
        print(f"🔄 Loading {config.model.model_name} model...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
        
        # Set padding token
        if config.model.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer.pad_token = config.model.pad_token
            
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(config.model.model_name)
        
        # Move model to appropriate device
        import torch
        if torch.backends.mps.is_available():
            self.model = self.model.to("mps")
        elif torch.cuda.is_available():
            self.model = self.model.to("cuda")
        
        print(f"✅ Model loaded! Parameters: {self.model.num_parameters():,}")
        print(f"Model size: ~{self.model.num_parameters() / 1_000_000:.1f}M parameters")
        
        return self.model, self.tokenizer
    
    def setup_lora(self):
        """Configure LoRA for efficient fine-tuning"""
        print("🔧 Applying LoRA configuration...")
        
        # Create LoRA configuration
        self.lora_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            target_modules=config.lora.target_modules,
            lora_dropout=config.lora.lora_dropout,
            bias=config.lora.bias,
            task_type=config.lora.task_type  # Explicitly set task type to avoid warning
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, self.lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        print("✅ LoRA configuration applied!")
        
        return self.model
    
    def setup_data_collator(self):
        """Setup data collator for training"""
        print("🔧 Setting up data collator...")
        
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=False  # We're doing causal language modeling, not masked
        )
        
        print("✅ Data collator ready!")
        
        return self.data_collator
    
    def get_model_info(self):
        """Get detailed information about the model"""
        if self.model is None:
            return "Model not loaded yet"
        
        info = {
            "total_parameters": self.model.num_parameters(),
            "model_name": config.model.model_name,
            "lora_enabled": hasattr(self.model, 'peft_config'),
            "device": next(self.model.parameters()).device if self.model.parameters() else "unknown"
        }
        
        if hasattr(self.model, 'peft_config'):
            info["lora_config"] = {
                "r": config.lora.r,
                "alpha": config.lora.lora_alpha,
                "target_modules": config.lora.target_modules,
                "dropout": config.lora.lora_dropout
            }
        
        return info
    
    def print_model_summary(self):
        """Print a comprehensive model summary"""
        print("\n📊 Model Summary:")
        print("=" * 50)
        
        info = self.get_model_info()
        print(f"Model: {info['model_name']}")
        print(f"Total Parameters: {info['total_parameters']:,}")
        print(f"LoRA Enabled: {info['lora_enabled']}")
        print(f"Device: {info['device']}")
        
        if info['lora_enabled']:
            print(f"\nLoRA Configuration:")
            lora_config = info['lora_config']
            print(f"  Rank (r): {lora_config['r']}")
            print(f"  Alpha: {lora_config['alpha']}")
            print(f"  Target Modules: {lora_config['target_modules']}")
            print(f"  Dropout: {lora_config['dropout']}")
    
    def save_model(self, save_path: str):
        """Save the fine-tuned model and tokenizer"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before saving")
        
        print(f"💾 Saving model to {save_path}...")
        
        # Save model and tokenizer
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        print("✅ Model saved successfully!")
        print(f"You can now use this model for your business tasks!")
        
        return save_path
    
    def load_saved_model(self, model_path: str, use_raw: bool = False):
        """Load a previously saved fine-tuned model
        
        Args:
            model_path: Path to the saved model (LoRA adapter directory)
            use_raw: If True, load only the base model without adapters
        """
        if use_raw:
            print("🔄 Loading raw base model (no adapters)...")
            # Load base model directly
            self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(config.model.model_name)
        else:
            print(f"🔄 Loading saved model from {model_path}...")
            
            # Load tokenizer from saved path or fall back to base model name
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            except:
                # If tokenizer not in saved path, use base model tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
            
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model first
            base_model = AutoModelForCausalLM.from_pretrained(config.model.model_name)
            
            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(base_model, model_path)
        
        # Move model to appropriate device
        if torch.backends.mps.is_available():
            self.model = self.model.to("mps")
        elif torch.cuda.is_available():
            self.model = self.model.to("cuda")
        
        print("✅ Saved model loaded successfully!")
        
        return self.model, self.tokenizer


def create_model_manager():
    """Factory function to create a ModelManager instance"""
    return ModelManager()
