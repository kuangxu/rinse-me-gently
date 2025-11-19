"""
Training utilities for LLM Fine-Tuning Demo
Handles training configuration, execution, and monitoring
"""

import time
from transformers import Trainer, TrainingArguments
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import numpy as np
from config import config


class TrainingManager:
    """Manages the training process and monitoring"""
    
    def __init__(self, model, tokenized_dataset, data_collator):
        self.model = model
        self.tokenized_dataset = tokenized_dataset
        self.data_collator = data_collator
        self.trainer = None
        self.training_args = None
        self.training_start_time = None
        self.training_end_time = None
        
    def setup_training(self):
        """Setup training arguments and trainer"""
        print("🔧 Setting up training configuration...")
        
        # Disable pin_memory for MPS (Apple Silicon) - not supported and causes warnings
        import torch
        pin_memory = not torch.backends.mps.is_available()  # False for MPS, True for CUDA/CPU
        
        # Create training arguments
        training_args_dict = {
            "output_dir": config.get_output_dir(),  # Use auto-generated path from data file name
            "per_device_train_batch_size": config.training.per_device_train_batch_size,
            "learning_rate": config.training.learning_rate,
            "logging_steps": config.training.logging_steps,
            "save_strategy": config.training.save_strategy,
            "remove_unused_columns": config.training.remove_unused_columns,
            "warmup_steps": config.training.warmup_steps,
            "weight_decay": config.training.weight_decay,
            "max_grad_norm": config.training.max_grad_norm,
            "eval_strategy": config.training.evaluation_strategy,
            "dataloader_num_workers": config.training.dataloader_num_workers,
            "fp16": config.training.fp16,
            "dataloader_pin_memory": pin_memory,  # Disable for MPS to prevent warnings
        }
        
        # Use max_steps if set, otherwise use num_train_epochs
        if config.training.max_steps is not None:
            training_args_dict["max_steps"] = config.training.max_steps
        else:
            training_args_dict["num_train_epochs"] = config.training.num_train_epochs
        
        self.training_args = TrainingArguments(**training_args_dict)
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_dataset,
            data_collator=self.data_collator,
        )
        
        print("✅ Training setup complete!")
        self._print_training_info()
        
        return self.trainer
    
    def _print_training_info(self):
        """Print training configuration information"""
        print(f"\n📊 Training Configuration:")
        print(f"  Training examples: {len(self.tokenized_dataset)}")
        print(f"  Batch size: {self.training_args.per_device_train_batch_size}")
        print(f"  Learning rate: {self.training_args.learning_rate}")
        if hasattr(self.training_args, 'max_steps') and self.training_args.max_steps is not None:
            print(f"  Max steps: {self.training_args.max_steps}")
        else:
            print(f"  Epochs: {self.training_args.num_train_epochs}")
        print(f"  Logging steps: {self.training_args.logging_steps}")
        print(f"  Output directory: {self.training_args.output_dir}")
    
    def train(self, show_progress: bool = True):
        """Execute the training process"""
        if self.trainer is None:
            raise ValueError("Trainer not setup. Call setup_training() first.")
        
        print("🚀 Starting training...")
        if show_progress:
            print("Watch the loss decrease - this means the model is learning!")
            print("\nTraining progress:")
            print("-" * 50)
        
        # Record start time
        self.training_start_time = time.time()
        
        try:
            # Execute training
            self.trainer.train()
            
            # Record end time
            self.training_end_time = time.time()
            
            # Calculate training time
            training_time = self.training_end_time - self.training_start_time
            
            print(f"\n✅ Training complete!")
            print(f"⏱️  Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

            # Save the fine-tuned model explicitly
            # For PEFT/LoRA models, we need to use model.save_pretrained() instead of trainer.save_model()
            # This ensures the adapter weights (adapter_config.json and adapter_model.safetensors) are saved correctly
            try:
                save_path = self.trainer.state.best_model_checkpoint or self.trainer.args.output_dir
                if save_path:
                    print(f"💾 Saving fine-tuned model to: {save_path}")
                    
                    # Get the model from trainer (might be wrapped)
                    model_to_save = self.trainer.model if hasattr(self.trainer, 'model') else self.model
                    
                    # Check if it's a PEFT model
                    from peft import PeftModel
                    is_peft = isinstance(model_to_save, PeftModel) or hasattr(model_to_save, 'peft_config')
                    
                    if is_peft:
                        print(f"   [INFO] Detected PEFT model - saving adapter weights only")
                        # Save PEFT adapter (saves adapter_config.json and adapter_model.safetensors)
                        model_to_save.save_pretrained(save_path)
                        
                        # Verify adapter files were created
                        import os
                        adapter_config_path = os.path.join(save_path, "adapter_config.json")
                        if os.path.exists(adapter_config_path):
                            print("✅ Model saved successfully! (PEFT adapter)")
                        else:
                            print(f"⚠️  Warning: adapter_config.json not found at {adapter_config_path}")
                    else:
                        print(f"   [INFO] Not a PEFT model - saving full model")
                        # For non-PEFT models, save normally
                        model_to_save.save_pretrained(save_path)
                        print("✅ Model saved successfully!")
                else:
                    print("⚠️  Could not determine save path")
            except Exception as e:
                print(f"⚠️  Warning: Could not save model: {e}")
                import traceback
                traceback.print_exc()

            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def get_training_losses(self) -> List[float]:
        """Extract training losses from trainer state"""
        if self.trainer is None:
            return []
        
        losses = [
            log["loss"] 
            for log in self.trainer.state.log_history 
            if "loss" in log
        ]
        
        return losses
    
    def plot_training_curve(self, save_path: Optional[str] = None):
        """Plot the training loss curve"""
        losses = self.get_training_losses()
        
        if not losses:
            print("❌ No training losses available to plot")
            return
        
        # Create the plot
        plt.figure(figsize=config.visualization.figure_size)
        plt.plot(np.arange(len(losses)), losses, 'b-', linewidth=config.visualization.line_width)
        plt.xlabel("Training Steps")
        plt.ylabel("Loss (Error)")
        plt.title("Training Loss Curve - Model Learning Progress")
        plt.grid(True, alpha=config.visualization.grid_alpha)
        
        # Add final loss annotation
        final_loss = losses[-1]
        plt.annotate(f'Final Loss: {final_loss:.4f}', 
                    xy=(len(losses)-1, final_loss), 
                    xytext=(len(losses)*0.7, max(losses)*0.8),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                    fontsize=10, color='red')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training curve saved to {save_path}")
        
        if config.visualization.show_plots:
            plt.show()
        
        return losses
    
    def print_training_summary(self):
        """Print a comprehensive training summary"""
        losses = self.get_training_losses()
        
        if not losses:
            print("❌ No training data available")
            return
        
        print("\n📊 Training Summary:")
        print("=" * 50)
        
        initial_loss = losses[0]
        final_loss = losses[-1]
        loss_reduction = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0
        
        print(f"Initial Loss: {initial_loss:.4f}")
        print(f"Final Loss: {final_loss:.4f}")
        print(f"Loss Reduction: {loss_reduction:.1f}%")
        print(f"Training Steps: {len(losses)}")
        
        if self.training_start_time and self.training_end_time:
            training_time = self.training_end_time - self.training_start_time
            print(f"Training Time: {training_time:.2f} seconds")
            print(f"Steps per Second: {len(losses) / training_time:.2f}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        losses = self.get_training_losses()
        
        stats = {
            "training_completed": self.training_end_time is not None,
            "num_steps": len(losses),
            "initial_loss": losses[0] if losses else None,
            "final_loss": losses[-1] if losses else None,
            "loss_reduction_percent": None,
            "training_time_seconds": None,
            "steps_per_second": None
        }
        
        if losses:
            initial_loss = losses[0]
            final_loss = losses[-1]
            stats["loss_reduction_percent"] = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0
        
        if self.training_start_time and self.training_end_time:
            training_time = self.training_end_time - self.training_start_time
            stats["training_time_seconds"] = training_time
            stats["steps_per_second"] = len(losses) / training_time if training_time > 0 else 0
        
        return stats


def create_training_manager(model, tokenized_dataset, data_collator):
    """Factory function to create a TrainingManager instance"""
    return TrainingManager(model, tokenized_dataset, data_collator)


def quick_train_test(model, tokenized_dataset, data_collator, epochs: int = 1):
    """Quick training test with minimal configuration"""
    print("🧪 Running quick training test...")
    
    # Create minimal training args
    training_args = TrainingArguments(
        output_dir="./test_training",
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        learning_rate=1e-4,
        logging_steps=1,
        save_strategy="no",
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Train
    start_time = time.time()
    try:
        trainer.train()
        end_time = time.time()
        print(f"✅ Quick test completed in {end_time - start_time:.2f} seconds")
        return trainer
    except Exception as e:
        print(f"⚠️  Quick test encountered an issue: {e}")
        # Return trainer anyway for testing purposes
        return trainer
