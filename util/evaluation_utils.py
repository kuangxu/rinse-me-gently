"""
Evaluation utilities for LLM Fine-Tuning Demo
Handles model testing, evaluation, and result visualization
"""

import torch
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
from .config import config


class EvaluationManager:
    """Manages model evaluation and testing"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.test_results = []
        
    def test_model(self, prompt: str, max_tokens: int = None, temperature: float = None, 
                   do_sample: bool = None) -> str:
        """Test the model with a single prompt"""
        if max_tokens is None:
            max_tokens = config.evaluation.max_new_tokens
        if temperature is None:
            temperature = config.evaluation.temperature
        if do_sample is None:
            do_sample = config.evaluation.do_sample
        
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
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response
    
    def test_multiple_prompts(self, prompts: List[str] = None, 
                            max_tokens: int = None, temperature: float = None) -> List[Dict[str, str]]:
        """Test the model with multiple prompts"""
        if prompts is None:
            prompts = config.evaluation.test_prompts
        
        if max_tokens is None:
            max_tokens = config.evaluation.max_new_tokens
        if temperature is None:
            temperature = config.evaluation.temperature
        
        results = []
        
        print("🧪 Testing the fine-tuned model:")
        print("=" * 60)
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\nTest {i}: {prompt}")
            
            try:
                response = self.test_model(prompt, max_tokens, temperature)
                
                result = {
                    "prompt": prompt,
                    "response": response,
                    "success": True,
                    "error": None
                }
                
                print(f"Response: {response}")
                print("-" * 40)
                
            except Exception as e:
                result = {
                    "prompt": prompt,
                    "response": None,
                    "success": False,
                    "error": str(e)
                }
                
                print(f"❌ Error: {e}")
                print("-" * 40)
            
            results.append(result)
        
        self.test_results = results
        return results
    
    def compare_with_original(self, original_model, prompts: List[str] = None):
        """Compare fine-tuned model with original model"""
        if prompts is None:
            prompts = config.evaluation.test_prompts
        
        print("🔄 Comparing fine-tuned vs original model:")
        print("=" * 60)
        
        comparison_results = []
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\nComparison {i}: {prompt}")
            print("-" * 40)
            
            # Test fine-tuned model
            try:
                fine_tuned_response = self.test_model(prompt)
                print(f"Fine-tuned: {fine_tuned_response}")
            except Exception as e:
                fine_tuned_response = f"Error: {e}"
                print(f"Fine-tuned: {fine_tuned_response}")
            
            # Test original model
            try:
                original_inputs = self.tokenizer(prompt, return_tensors="pt")
                with torch.no_grad():
                    original_outputs = original_model.generate(
                        **original_inputs,
                        max_new_tokens=config.evaluation.max_new_tokens,
                        temperature=config.evaluation.temperature,
                        do_sample=config.evaluation.do_sample,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                original_response = self.tokenizer.decode(original_outputs[0], skip_special_tokens=True)
                print(f"Original:   {original_response}")
            except Exception as e:
                original_response = f"Error: {e}"
                print(f"Original:   {original_response}")
            
            comparison_results.append({
                "prompt": prompt,
                "fine_tuned_response": fine_tuned_response,
                "original_response": original_response
            })
            
            print()
        
        return comparison_results
    
    def analyze_responses(self, results: List[Dict[str, str]] = None):
        """Analyze the quality of model responses"""
        if results is None:
            results = self.test_results
        
        if not results:
            print("❌ No test results available for analysis")
            return
        
        print("\n📊 Response Analysis:")
        print("=" * 50)
        
        successful_tests = [r for r in results if r["success"]]
        failed_tests = [r for r in results if not r["success"]]
        
        print(f"Total Tests: {len(results)}")
        print(f"Successful: {len(successful_tests)}")
        print(f"Failed: {len(failed_tests)}")
        print(f"Success Rate: {len(successful_tests)/len(results)*100:.1f}%")
        
        if successful_tests:
            # Analyze response lengths
            response_lengths = [len(r["response"]) for r in successful_tests]
            avg_length = np.mean(response_lengths)
            min_length = np.min(response_lengths)
            max_length = np.max(response_lengths)
            
            print(f"\nResponse Length Statistics:")
            print(f"  Average: {avg_length:.1f} characters")
            print(f"  Min: {min_length} characters")
            print(f"  Max: {max_length} characters")
        
        if failed_tests:
            print(f"\nFailed Tests:")
            for i, result in enumerate(failed_tests, 1):
                print(f"  {i}. {result['prompt'][:50]}... - {result['error']}")
    
    def interactive_test(self):
        """Interactive testing mode where user can input custom prompts"""
        print("\n🎮 Interactive Testing Mode")
        print("Enter prompts to test the model (type 'quit' to exit)")
        print("=" * 50)
        
        while True:
            try:
                prompt = input("\nEnter prompt: ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("Exiting interactive mode...")
                    break
                
                if not prompt:
                    print("Please enter a valid prompt")
                    continue
                
                print(f"\nTesting: {prompt}")
                response = self.test_model(prompt)
                print(f"Response: {response}")
                print("-" * 40)
                
            except KeyboardInterrupt:
                print("\nExiting interactive mode...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def save_test_results(self, filepath: str, results: List[Dict[str, str]] = None):
        """Save test results to a file"""
        if results is None:
            results = self.test_results
        
        if not results:
            print("❌ No test results to save")
            return
        
        import json
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Test results saved to {filepath}")
    
    def load_test_results(self, filepath: str) -> List[Dict[str, str]]:
        """Load test results from a file"""
        import json
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            self.test_results = results
            print(f"📁 Test results loaded from {filepath}")
            return results
            
        except Exception as e:
            print(f"❌ Error loading test results: {e}")
            return []


def create_evaluation_manager(model, tokenizer):
    """Factory function to create an EvaluationManager instance"""
    return EvaluationManager(model, tokenizer)


def quick_test(model, tokenizer, prompt: str = "Hello, how are you?") -> str:
    """Quick test function for a single prompt"""
    evaluator = EvaluationManager(model, tokenizer)
    return evaluator.test_model(prompt)
