"""
Data utilities for LLM Fine-Tuning Demo
Handles dataset loading, preprocessing, and tokenization
"""

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional
from .config import config


class DataManager:
    """Manages dataset loading, preprocessing, and tokenization"""
    
    def __init__(self, tokenizer: Optional[AutoTokenizer] = None):
        self.tokenizer = tokenizer
        self.raw_dataset = None
        self.tokenized_dataset = None
        
    def load_demo_dataset(self):
        """Load the Shakespeare dataset from custom file"""
        print("📁 Loading Shakespeare dataset...")
        
        try:
            # Always use the Shakespeare data file
            print(f"Using Shakespeare data file: {config.data.custom_data_file}")
            return self.load_custom_dataset(config.data.custom_data_file)
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            raise
    
    def load_custom_dataset(self, file_path: str):
        """Load custom dataset from a text file"""
        print(f"📁 Loading custom dataset from {file_path}...")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            # Filter and process lines for Shakespeare text
            processed_lines = []
            for line in lines:
                line = line.strip()
                # Skip empty lines, headers, and very short lines
                if (len(line) >= config.data.min_length and 
                    len(line) <= config.data.max_length and
                    not line.isdigit() and  # Skip line numbers
                    not line.startswith("by ") and  # Skip author lines
                    not line.startswith("THE ")):  # Skip title lines
                    processed_lines.append(line)
            
            # Limit the number of samples for faster training
            if len(processed_lines) > config.data.max_samples:
                print(f"📊 Limiting dataset from {len(processed_lines)} to {config.data.max_samples} samples for faster training")
                processed_lines = processed_lines[:config.data.max_samples]
            
            # Format data according to expected structure
            if config.data.data_format == "instruction_response":
                # Each line should be: instruction: ... response: ...
                data = {"text": [line + self.tokenizer.eos_token for line in processed_lines]}
            else:
                # Plain text format - add EOS token
                data = {"text": [line + self.tokenizer.eos_token for line in processed_lines]}
            
            self.raw_dataset = Dataset.from_dict(data)
            print(f"✅ Loaded {len(self.raw_dataset)} examples from custom file")
            
            # Show sample data
            self._print_sample_data()
            
            return self.raw_dataset
            
        except Exception as e:
            print(f"❌ Error loading custom dataset: {e}")
            raise
    
    def load_dataset(self, custom_file: Optional[str] = None):
        """Load dataset (demo or custom)"""
        if custom_file:
            return self.load_custom_dataset(custom_file)
        else:
            return self.load_demo_dataset()
    
    def _print_sample_data(self):
        """Print sample data for inspection"""
        if self.raw_dataset is None:
            print("No dataset loaded")
            return
        
        print("\n📋 Sample data:")
        print("-" * 50)
        sample = self.raw_dataset[0]
        
        if isinstance(sample, dict):
            for key, value in sample.items():
                print(f"{key}: {value}")
        else:
            print(sample)
        
        print("\n" + "=" * 50)
    
    def tokenize_dataset(self):
        """Tokenize the dataset for training"""
        if self.raw_dataset is None:
            raise ValueError("No dataset loaded. Call load_dataset() first.")
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer not provided. Set tokenizer before tokenizing.")
        
        print("🔄 Converting text to tokens...")
        
        def tokenize_function(examples):
            """Convert text to numbers (tokens) that the model can understand"""
            tokenized = self.tokenizer(
                examples["text"], 
                truncation=True, 
                padding="max_length", 
                max_length=config.model.max_length
            )
            # For causal language modeling, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        # Apply tokenization
        self.tokenized_dataset = self.raw_dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=self.raw_dataset.column_names
        )
        
        print("✅ Tokenization complete!")
        
        # Show tokenization details
        self._print_tokenization_details()
        
        return self.tokenized_dataset
    
    def _print_tokenization_details(self):
        """Print details about the tokenization process"""
        if self.tokenized_dataset is None:
            return
        
        print("\n🔍 Tokenization Details:")
        print("-" * 30)
        
        # Show sample tokens
        sample_tokens = self.tokenized_dataset[0]["input_ids"][:20]
        print(f"Sample tokens (first 20): {sample_tokens}")
        
        # Decode back to text
        if self.tokenizer:
            decoded_text = self.tokenizer.decode(
                self.tokenized_dataset[0]["input_ids"][:50]
            )
            print(f"\nDecoded back to text:")
            print(f"{decoded_text}...")
        
        print(f"\nDataset size: {len(self.tokenized_dataset)} examples")
        print(f"Max length: {config.model.max_length} tokens")
    
    def get_dataset_info(self):
        """Get information about the current dataset"""
        info = {
            "raw_dataset_loaded": self.raw_dataset is not None,
            "tokenized_dataset_loaded": self.tokenized_dataset is not None,
            "raw_size": len(self.raw_dataset) if self.raw_dataset else 0,
            "tokenized_size": len(self.tokenized_dataset) if self.tokenized_dataset else 0,
            "max_length": config.model.max_length,
            "tokenizer_available": self.tokenizer is not None
        }
        
        return info
    
    def print_dataset_summary(self):
        """Print a comprehensive dataset summary"""
        print("\n📊 Dataset Summary:")
        print("=" * 50)
        
        info = self.get_dataset_info()
        print(f"Raw Dataset Loaded: {info['raw_dataset_loaded']}")
        print(f"Tokenized Dataset Loaded: {info['tokenized_dataset_loaded']}")
        print(f"Raw Dataset Size: {info['raw_size']} examples")
        print(f"Tokenized Dataset Size: {info['tokenized_size']} examples")
        print(f"Max Length: {info['max_length']} tokens")
        print(f"Tokenizer Available: {info['tokenizer_available']}")
    
    def validate_dataset(self):
        """Validate the dataset for training readiness"""
        if self.tokenized_dataset is None:
            print("❌ No tokenized dataset available")
            return False
        
        if len(self.tokenized_dataset) == 0:
            print("❌ Dataset is empty")
            return False
        
        # Check if all required fields are present
        required_fields = ["input_ids", "attention_mask"]
        sample = self.tokenized_dataset[0]
        
        for field in required_fields:
            if field not in sample:
                print(f"❌ Missing required field: {field}")
                return False
        
        print("✅ Dataset validation passed!")
        return True


def create_data_manager(tokenizer: Optional[AutoTokenizer] = None):
    """Factory function to create a DataManager instance"""
    return DataManager(tokenizer)


def format_instruction_response(instruction: str, response: str) -> str:
    """Format instruction and response into the expected format"""
    return f"instruction: {instruction} response: {response}"


def create_sample_dataset(instructions: List[str], responses: List[str]) -> Dataset:
    """Create a sample dataset from lists of instructions and responses"""
    if len(instructions) != len(responses):
        raise ValueError("Instructions and responses must have the same length")
    
    formatted_texts = [
        format_instruction_response(inst, resp) 
        for inst, resp in zip(instructions, responses)
    ]
    
    data = {"text": formatted_texts}
    return Dataset.from_dict(data)
