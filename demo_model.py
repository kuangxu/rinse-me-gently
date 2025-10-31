#!/usr/bin/env python3
"""
Demo script for testing fine-tuned models
Loads a saved model and runs test prompts with expected responses
"""

import argparse
import json
import sys
import os
from typing import List, Dict, Any, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch


class ModelDemo:
    """Demo class for testing fine-tuned models"""
    
    def __init__(self, model_path: Optional[str], use_raw: bool = False):
        self.model_path = model_path
        self.use_raw = use_raw
        self.model = None
        self.tokenizer = None
        self.test_prompts = []
        
    def load_model(self):
        """Load the model and tokenizer (raw or fine-tuned)"""
        if self.use_raw:
            print("\n[STATUS] Loading raw base model...")
        else:
            print(f"\n[STATUS] Loading fine-tuned model from {self.model_path}...")
        
        try:
            # Load tokenizer and set padding token
            print("[STEP 1/4] Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"[STEP 1/4] ✓ Tokenizer loaded (vocab size: {len(self.tokenizer)})")
            
            # Load base model
            print("[STEP 2/4] Loading base model...")
            base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
            print("[STEP 2/4] ✓ Base model loaded")
            
            # Move to appropriate device (GPU/MPS if available, else CPU)
            print("[STEP 3/4] Moving model to device...")
            device = "cpu"
            if torch.backends.mps.is_available():
                base_model = base_model.to("mps")  # Apple Silicon GPU
                device = "mps"
                print("[STEP 3/4] ✓ Model moved to Apple Silicon GPU (MPS)")
            elif torch.cuda.is_available():
                base_model = base_model.to("cuda")  # NVIDIA GPU
                device = "cuda"
                print(f"[STEP 3/4] ✓ Model moved to CUDA device")
            else:
                print("[STEP 3/4] ✓ Model running on CPU")
            
            # Use raw model or load fine-tuned LoRA adapter
            if self.use_raw:
                self.model = base_model
                print("[STEP 4/4] ✓ Using raw base model (no fine-tuning)")
            else:
                print("[STEP 4/4] Loading LoRA adapter weights...")
                self.model = PeftModel.from_pretrained(base_model, self.model_path)
                print("[STEP 4/4] ✓ Fine-tuned LoRA adapter loaded")
            
            print(f"\n[SUCCESS] Model loaded successfully on {device}")
            return True
            
        except Exception as e:
            print(f"\n[ERROR] Error loading model: {e}")
            return False
    
    def load_test_prompts(self, prompts_file: str = "data/test_prompts.json"):
        """Load test prompts from JSON file"""
        try:
            print(f"\n[STATUS] Loading test prompts from {prompts_file}...")
            with open(prompts_file, 'r', encoding='utf-8') as f:
                self.test_prompts = json.load(f)
            
            print(f"[SUCCESS] Loaded {len(self.test_prompts)} test prompts")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error loading test prompts: {e}")
            return False
    
    def load_training_text(self, training_file: str):
        """Load training text from a file and create simple prompts"""
        try:
            print(f"\n[STATUS] Loading training text from {training_file}...")
            with open(training_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            print(f"[INFO] Read {len(lines)} lines from file")
            print("[STATUS] Creating prompts from training text...")
            
            # Create prompts from first few lines of training text
            prompts = []
            for i, line in enumerate(lines[:10]):  # Limit to first 10 lines
                line = line.strip()
                if len(line) > 20:  # Only use substantial lines
                    words = line.split()
                    if len(words) > 3:
                        # Use first 3 words as prompt, full line as expected
                        prompt = ' '.join(words[:3]) + ' '
                        prompts.append({
                            'prompt': prompt,
                            'expected': line,
                            'source': f'Line {i+1}',
                            'context': 'Training text sample'
                        })
            
            self.test_prompts = prompts
            print(f"[SUCCESS] Created {len(self.test_prompts)} prompts from training text")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error loading training text: {e}")
            return False
    
    def generate_response(self, prompt: str, max_tokens: int = 50, temperature: float = 0.7) -> str:
        """Generate response for a given prompt"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move inputs to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode generated tokens to text
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt from response if it's included (model returns prompt + generation)
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response
    
    def run_test_prompts(self, max_tokens: int = 50, temperature: float = 0.7):
        """Run all test prompts and compare with expected responses"""
        if not self.test_prompts:
            print("[WARNING] No test prompts loaded. Call load_test_prompts() first.")
            return
        
        print("\n" + "=" * 80)
        print("[STATUS] Running Test Prompts")
        print(f"[INFO] Total prompts: {len(self.test_prompts)}")
        print(f"[INFO] Generation settings: max_tokens={max_tokens}, temperature={temperature}")
        print("=" * 80)
        
        results = []
        successful = 0
        failed = 0
        
        for i, test in enumerate(self.test_prompts, 1):
            print(f"\n[TEST {i}/{len(self.test_prompts)}] Processing prompt...")
            print(f"   Prompt: \"{test['prompt']}\"")
            if 'expected' in test and test['expected']:
                print(f"   Expected: \"{test['expected']}\"")
            if 'source' in test and test['source']:
                print(f"   Source: {test['source']}")
            
            try:
                print(f"   [STATUS] Generating response...")
                response = self.generate_response(test['prompt'], max_tokens, temperature)
                print(f"   [SUCCESS] Response: \"{response}\"")
                successful += 1
                
                result_item = {
                    'prompt': test['prompt'],
                    'response': response,
                }
                if 'expected' in test and test['expected']:
                    result_item['expected'] = test['expected']
                if 'source' in test and test['source']:
                    result_item['source'] = test['source']
                if 'context' in test and test['context']:
                    result_item['context'] = test['context']
                results.append(result_item)
                
            except Exception as e:
                print(f"   [ERROR] Generation failed: {e}")
                failed += 1
                error_item = {
                    'prompt': test['prompt'],
                    'response': f"Error: {e}",
                }
                if 'expected' in test and test['expected']:
                    error_item['expected'] = test['expected']
                if 'source' in test and test['source']:
                    error_item['source'] = test['source']
                if 'context' in test and test['context']:
                    error_item['context'] = test['context']
                results.append(error_item)
        
        print("\n" + "=" * 80)
        print("[SUMMARY] Test Execution Complete")
        print(f"   Total Tests: {len(self.test_prompts)}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        print("=" * 80)
        
        return results
    
    def interactive_mode(self):
        """Enter interactive mode for free-form testing"""
        print("\n" + "=" * 80)
        print("[STATUS] Entering Interactive Mode")
        print("=" * 80)
        print("\n[INFO] Type 'quit', 'exit', or 'bye' to end the conversation")
        print("[INFO] Type 'help' for commands")
        print("-" * 80)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\nGoodbye!")
                    break
                
                if user_input.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  - Type your message to chat with the model")
                    print("  - 'quit', 'exit', 'bye', or 'q' to end chat")
                    print("  - 'help' to show this message")
                    print("  - 'clear' to clear the conversation context")
                    continue
                
                if user_input.lower() == 'clear':
                    print("Conversation context cleared!")
                    continue
                
                if not user_input:
                    continue
                
                print("[STATUS] Processing input and generating response...")
                print("Model: ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response)
                print("[INFO] Response generated successfully")
                
            except KeyboardInterrupt:
                print("\n\nChat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Try again or type 'quit' to exit.\n")


def main():
    """Main function - entry point for model demo script"""
    print("=" * 80)
    print("Model Demo Script - Starting")
    print("=" * 80)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Demo script for testing fine-tuned models")
    parser.add_argument("--model-path", type=str, required=False, default=None,
                       help="Path to the saved fine-tuned model (ignored with --use-raw)")
    parser.add_argument("--prompts-file", type=str, default="data/test_prompts.json",
                       help="Path to the test prompts JSON file")
    parser.add_argument("--interactive", action="store_true",
                       help="Enter interactive mode after running tests")
    parser.add_argument("--max-tokens", type=int, default=50,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for text generation (higher = more creative)")
    parser.add_argument("--use-raw", action="store_true",
                       help="Use raw base model (no fine-tuned adapters)")
    parser.add_argument("--training-text", type=str,
                       help="Path to training text file to create prompts from")
    
    args = parser.parse_args()
    
    print("\n[STATUS] Validating arguments...")
    
    # Validate model path when not using raw model
    if not args.use_raw:
        if not args.model_path:
            print("[ERROR] --model-path is required unless --use-raw is specified")
            sys.exit(1)
        if not os.path.exists(args.model_path):
            print(f"[ERROR] Model path does not exist: {args.model_path}")
            sys.exit(1)
        print(f"[INFO] Model path validated: {args.model_path}")
    else:
        print("[INFO] Using raw base model (no fine-tuning)")
    
    # Validate prompts or training text file exists
    if args.training_text:
        if not os.path.exists(args.training_text):
            print(f"[ERROR] Training text file does not exist: {args.training_text}")
            sys.exit(1)
        print(f"[INFO] Training text file validated: {args.training_text}")
    else:
        if not os.path.exists(args.prompts_file):
            print(f"[ERROR] Prompts file does not exist: {args.prompts_file}")
            sys.exit(1)
        print(f"[INFO] Prompts file validated: {args.prompts_file}")
    
    print("[SUCCESS] All arguments validated")
    
    # Create demo instance
    print("\n[STATUS] Initializing ModelDemo instance...")
    demo = ModelDemo(args.model_path, use_raw=args.use_raw)
    print("[SUCCESS] ModelDemo initialized")
    
    try:
        # Load model
        if not demo.load_model():
            print("[ERROR] Failed to load model. Exiting.")
            sys.exit(1)
        
        # Load test prompts or training text
        if args.training_text:
            if not demo.load_training_text(args.training_text):
                print("[ERROR] Failed to load training text. Exiting.")
                sys.exit(1)
        else:
            if not demo.load_test_prompts(args.prompts_file):
                print("[ERROR] Failed to load test prompts. Exiting.")
                sys.exit(1)
        
        # Run test prompts
        results = demo.run_test_prompts(args.max_tokens, args.temperature)
        
        # Enter interactive mode if requested
        if args.interactive:
            demo.interactive_mode()
        else:
            print("\n[INFO] Interactive mode not requested. Use --interactive to enable.")
        
        print("\n" + "=" * 80)
        print("[SUCCESS] Demo script completed successfully")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n[WARNING] Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
