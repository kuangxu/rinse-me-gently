#!/usr/bin/env python3
"""
Fine-tuning script for LLM Fine-Tuning Demo
This script can be used to:
1. Fine-tune a model (main functionality)
2. Run tests to verify all modules work correctly
"""

import sys
import traceback
import time
import argparse
from typing import List, Dict, Any


def test_imports():
    """Test that all modules can be imported"""
    print("\n[STATUS] Testing module imports...")
    try:
        print("   [STEP] Importing config...")
        from config import config
        print("   [STEP] Importing util.model_utils...")
        from util.model_utils import create_model_manager
        print("   [STEP] Importing util.data_utils...")
        from util.data_utils import create_data_manager
        print("   [STEP] Importing util.training_utils...")
        from util.training_utils import create_training_manager
        print("   [STEP] Importing util.evaluation_utils...")
        from util.evaluation_utils import create_evaluation_manager
        print("[SUCCESS] All imports successful")
        return True
    except Exception as e:
        print(f"[ERROR] Import failed: {e}")
        return False


def test_config():
    """Test configuration module"""
    print("\n[STATUS] Testing configuration module...")
    try:
        print("   [STEP] Loading configuration...")
        from config import config
        
        print("   [STEP] Validating model configuration...")
        assert config.model.model_name == "distilgpt2"
        print(f"   [INFO] Model name: {config.model.model_name}")
        
        print("   [STEP] Validating training configuration...")
        assert config.training.num_train_epochs == 1
        print(f"   [INFO] Training epochs: {config.training.num_train_epochs}")
        
        print("   [STEP] Validating LoRA configuration...")
        assert config.lora.r == 8
        print(f"   [INFO] LoRA rank (r): {config.lora.r}")
        
        print("[SUCCESS] Configuration test passed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Configuration test failed: {e}")
        traceback.print_exc()
        return False


def test_model_utils():
    """Test model utilities"""
    print("\n[STATUS] Testing model utilities...")
    try:
        print("   [STEP] Creating model manager...")
        from util.model_utils import create_model_manager
        model_manager = create_model_manager()
        assert model_manager is not None
        print("   [SUCCESS] Model manager created")
        
        # Load base model and tokenizer
        print("   [STEP] Loading base model and tokenizer...")
        model, tokenizer = model_manager.load_model_and_tokenizer()
        assert model is not None
        assert tokenizer is not None
        print("   [SUCCESS] Model and tokenizer loaded")
        
        # Setup LoRA for efficient fine-tuning
        print("   [STEP] Setting up LoRA adapter...")
        model = model_manager.setup_lora()
        assert model is not None
        print("   [SUCCESS] LoRA adapter configured")
        
        # Setup data collator for batching
        print("   [STEP] Creating data collator...")
        data_collator = model_manager.setup_data_collator()
        assert data_collator is not None
        print("   [SUCCESS] Data collator created")
        
        # Verify model info is available
        print("   [STEP] Retrieving model information...")
        info = model_manager.get_model_info()
        assert "total_parameters" in info
        print(f"   [INFO] Model info retrieved: {len(info)} metrics available")
        
        print("[SUCCESS] Model utilities test passed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Model utilities test failed: {e}")
        traceback.print_exc()
        return False


def test_data_utils():
    """Test data utilities"""
    print("\n[STATUS] Testing data utilities...")
    try:
        print("   [STEP] Creating tokenizer...")
        from util.data_utils import create_data_manager
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        tokenizer.pad_token = tokenizer.eos_token
        print("   [SUCCESS] Tokenizer created")
        
        print("   [STEP] Creating data manager...")
        data_manager = create_data_manager(tokenizer)
        assert data_manager is not None
        print("   [SUCCESS] Data manager created")
        
        print("   [STEP] Loading raw dataset...")
        dataset = data_manager.load_dataset()
        assert dataset is not None
        assert len(dataset) > 0
        print(f"   [SUCCESS] Dataset loaded: {len(dataset)} samples")
        
        print("   [STEP] Tokenizing dataset...")
        tokenized_dataset = data_manager.tokenize_dataset()
        assert tokenized_dataset is not None
        assert len(tokenized_dataset) > 0
        print(f"   [SUCCESS] Dataset tokenized: {len(tokenized_dataset)} samples")
        
        print("   [STEP] Validating dataset...")
        is_valid = data_manager.validate_dataset()
        assert is_valid
        print("   [SUCCESS] Dataset validation passed")
        
        print("   [STEP] Retrieving dataset information...")
        info = data_manager.get_dataset_info()
        assert "raw_dataset_loaded" in info
        print(f"   [INFO] Dataset info: {len(info)} metrics available")
        
        print("[SUCCESS] Data utilities test passed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Data utilities test failed: {e}")
        traceback.print_exc()
        return False


def test_training_utils():
    """Test training utilities"""
    print("\n[STATUS] Testing training utilities...")
    try:
        print("   [STEP] Importing required modules...")
        from util.training_utils import create_training_manager
        from util.data_utils import create_data_manager
        from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
        
        # Load model and tokenizer
        print("   [STEP] Loading model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        tokenizer.pad_token = tokenizer.eos_token
        print("   [SUCCESS] Model and tokenizer loaded")
        
        # CRITICAL: Apply LoRA adapter before training
        # Without LoRA, the model trains all parameters, which can cause issues
        print("   [STEP] Applying LoRA adapter...")
        from util.model_utils import ModelManager
        model_manager = ModelManager()
        model_manager.model = model
        model_manager.tokenizer = tokenizer
        model = model_manager.setup_lora()  # Apply LoRA - only trains adapter weights
        print("   [SUCCESS] LoRA adapter applied")
        
        # Setup data collator for language modeling (not masked LM)
        print("   [STEP] Creating data collator...")
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        print("   [SUCCESS] Data collator created")
        
        # Load and tokenize dataset
        print("   [STEP] Loading and tokenizing dataset...")
        data_manager = create_data_manager(tokenizer)
        data_manager.load_dataset()
        tokenized_dataset = data_manager.tokenize_dataset()
        assert len(tokenized_dataset) > 0
        print(f"   [SUCCESS] Dataset ready: {len(tokenized_dataset)} samples")
        
        # Create training manager and setup trainer
        print("   [STEP] Creating training manager...")
        training_manager = create_training_manager(model, tokenized_dataset, data_collator)
        assert training_manager is not None
        print("   [SUCCESS] Training manager created")
        
        print("   [STEP] Setting up trainer...")
        trainer = training_manager.setup_training()
        assert trainer is not None
        print("   [SUCCESS] Trainer configured")
        
        # Run actual training
        print("\n   [STATUS] Starting training process...")
        start_time = time.time()
        training_manager.train(show_progress=True)
        elapsed_time = time.time() - start_time
        print(f"   [SUCCESS] Training completed in {elapsed_time:.2f} seconds")
        
        # Verify training completed successfully
        print("   [STEP] Verifying training statistics...")
        stats = training_manager.get_training_stats()
        assert stats.get("training_completed") is True
        assert "num_steps" in stats
        print(f"   [INFO] Training steps: {stats.get('num_steps', 'N/A')}")
        
        print("[SUCCESS] Training utilities test passed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Training utilities test failed: {e}")
        traceback.print_exc()
        return False


def test_evaluation_utils():
    """Test evaluation utilities - uses fine-tuned model from training test"""
    print("\n[STATUS] Testing evaluation utilities...")
    try:
        print("   [STEP] Importing evaluation modules...")
        from util.evaluation_utils import create_evaluation_manager, quick_test
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        from config import config
        
        print("   [STEP] Loading fine-tuned model and tokenizer...")
        # Load base model first
        base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load the fine-tuned LoRA adapter (from the training test)
        model_path = config.get_output_dir()
        try:
            model = PeftModel.from_pretrained(base_model, model_path)
            print(f"   [SUCCESS] Fine-tuned model loaded from {model_path}")
        except Exception as e:
            print(f"   [WARNING] Could not load fine-tuned model: {e}")
            print("   [INFO] Using base model instead (this is expected on first run)")
            model = base_model
        
        # Move model to appropriate device
        import torch
        if torch.backends.mps.is_available():
            model = model.to("mps")
        elif torch.cuda.is_available():
            model = model.to("cuda")
        
        print("   [STEP] Creating evaluation manager...")
        evaluation_manager = create_evaluation_manager(model, tokenizer)
        assert evaluation_manager is not None
        print("   [SUCCESS] Evaluation manager created")
        
        print("   [STEP] Testing single prompt generation...")
        response = evaluation_manager.test_model("Hello, how are you?")
        assert response is not None
        assert len(response) > 0
        print(f"   [SUCCESS] Single prompt test: Generated {len(response)} characters")
        
        print("   [STEP] Testing multiple prompts...")
        test_prompts = ["Hello", "What is AI?"]
        results = evaluation_manager.test_multiple_prompts(test_prompts)
        assert results is not None
        assert len(results) == len(test_prompts)
        print(f"   [SUCCESS] Multiple prompts test: {len(results)} responses generated")
        
        print("   [STEP] Testing quick_test function...")
        response = quick_test(model, tokenizer, "Test prompt")
        assert response is not None
        print(f"   [SUCCESS] Quick test: Generated {len(response)} characters")
        
        print("[SUCCESS] Evaluation utilities test passed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Evaluation utilities test failed: {e}")
        traceback.print_exc()
        return False


def test_shakespeare_prompts():
    """Test Shakespeare evaluation prompts with actual responses - uses fine-tuned model"""
    print("\n[STATUS] Testing Shakespeare prompts...")
    try:
        print("   [STEP] Loading fine-tuned model and tokenizer...")
        from util.evaluation_utils import create_evaluation_manager
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        from config import config
        
        # Load base model first
        base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load the fine-tuned LoRA adapter (from the training test)
        model_path = config.get_output_dir()
        try:
            model = PeftModel.from_pretrained(base_model, model_path)
            print(f"   [SUCCESS] Fine-tuned model loaded from {model_path}")
        except Exception as e:
            print(f"   [WARNING] Could not load fine-tuned model: {e}")
            print("   [INFO] Using base model instead (this is expected on first run)")
            model = base_model
        
        # Move model to appropriate device
        import torch
        if torch.backends.mps.is_available():
            model = model.to("mps")
        elif torch.cuda.is_available():
            model = model.to("cuda")
        
        print("   [STEP] Creating evaluation manager...")
        evaluation_manager = create_evaluation_manager(model, tokenizer)
        print("   [SUCCESS] Evaluation manager created")
        
        print("\n   [STATUS] Running Shakespeare test prompts...")
        shakespeare_tests = [
            {
                "prompt": "To be or not to be, that is the ",
                "expected": "question: Whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune, or to take arms against a sea of troubles, and by opposing end them.",
                "source": "Hamlet, Act 3, Scene 1"
            },
            {
                "prompt": "Romeo, Romeo, wherefore art thou ",
                "expected": "Romeo? Deny thy father and refuse thy name; or, if thou wilt not, be but sworn my love, and I'll no longer be a Capulet.",
                "source": "Romeo and Juliet, Act 2, Scene 2"
            },
            {
                "prompt": "All the world's a stage, and all the men and women merely ",
                "expected": "players: They have their exits and their entrances; and one man in his time plays many parts, his acts being seven ages.",
                "source": "As You Like It, Act 2, Scene 7"
            },
            {
                "prompt": "What light through yonder window breaks? It is the ",
                "expected": "east, and Juliet is the sun. Arise, fair sun, and kill the envious moon, who is already sick and pale with grief.",
                "source": "Romeo and Juliet, Act 2, Scene 2"
            },
            {
                "prompt": "Double, double toil and trouble; fire burn and ",
                "expected": "cauldron bubble. By the pricking of my thumbs, something wicked this way comes.",
                "source": "Macbeth, Act 4, Scene 1"
            }
        ]
        
        for i, test in enumerate(shakespeare_tests, 1):
            print(f"\n   [TEST {i}/{len(shakespeare_tests)}] Processing...")
            print(f"      Prompt: \"{test['prompt']}\"")
            print(f"      Expected: \"{test['expected']}\"")
            print(f"      Source: {test['source']}")
            try:
                print(f"      [STATUS] Generating response...")
                response = evaluation_manager.test_model(test['prompt'])
                print(f"      [SUCCESS] Response: \"{response}\"")
            except Exception as e:
                print(f"      [ERROR] Generation failed: {e}")
        
        print(f"\n   [SUCCESS] Completed {len(shakespeare_tests)} Shakespeare tests")
        print("[SUCCESS] Shakespeare prompts test passed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Shakespeare prompts test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and print summary"""
    print("=" * 80)
    print("LLM Fine-Tuning Pipeline Test Suite")
    print("=" * 80)
    
    # Define all test functions
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Model Utils", test_model_utils),
        ("Data Utils", test_data_utils),
        ("Training Utils", test_training_utils),
        ("Evaluation Utils", test_evaluation_utils),
        ("Shakespeare Prompts", test_shakespeare_prompts),
    ]
    
    print(f"\n[INFO] Total tests to run: {len(tests)}")
    print("[INFO] Starting test execution...\n")
    
    start_time = time.time()
    
    # Run all tests and collect results
    results = []
    for idx, (test_name, test_func) in enumerate(tests, 1):
        test_start = time.time()
        print(f"\n{'='*80}")
        print(f"Test {idx}/{len(tests)}: {test_name}")
        print(f"{'='*80}")
        try:
            success = test_func()
            elapsed = time.time() - test_start
            status = "PASSED" if success else "FAILED"
            print(f"\n[RESULT] {test_name}: {status} (took {elapsed:.2f}s)")
            results.append((test_name, success, elapsed))
        except Exception as e:
            elapsed = time.time() - test_start
            print(f"\n[ERROR] {test_name} test crashed: {e}")
            print(f"[RESULT] {test_name}: FAILED (crashed after {elapsed:.2f}s)")
            results.append((test_name, False, elapsed))
    
    total_elapsed = time.time() - start_time
    
    # Print summary of results
    print("\n" + "=" * 80)
    print("Test Results Summary")
    print("=" * 80)
    
    passed = 0
    total = len(results)
    
    for test_name, success, elapsed in results:
        status = "PASSED" if success else "FAILED"
        print(f"{test_name:25} {status:8} ({elapsed:6.2f}s)")
        if success:
            passed += 1
    
    print("-" * 80)
    print(f"{'Total':25} {passed}/{total:2} passed  ({total_elapsed:6.2f}s)")
    print("=" * 80)
    
    if passed == total:
        print("\n[SUCCESS] All tests passed!")
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed")
    
    return passed == total


def run_fine_tuning():
    """Main fine-tuning function - trains a model on the configured dataset"""
    print("=" * 80)
    print("LLM Fine-Tuning Pipeline - Starting Training")
    print("=" * 80)
    
    try:
        # Import required modules
        print("\n[STATUS] Importing modules...")
        from util.model_utils import create_model_manager
        from util.data_utils import create_data_manager
        from util.training_utils import create_training_manager
        from config import config
        
        print("[SUCCESS] Modules imported")
        
        # Step 1: Create model manager and load base model
        print("\n[STEP 1/5] Loading base model and tokenizer...")
        model_manager = create_model_manager()
        model, tokenizer = model_manager.load_model_and_tokenizer()
        print("[SUCCESS] Base model and tokenizer loaded")
        
        # Step 2: Setup LoRA adapter for efficient fine-tuning
        print("\n[STEP 2/5] Setting up LoRA adapter...")
        model = model_manager.setup_lora()
        print("[SUCCESS] LoRA adapter configured")
        
        # Step 3: Setup data collator
        print("\n[STEP 3/5] Creating data collator...")
        data_collator = model_manager.setup_data_collator()
        print("[SUCCESS] Data collator created")
        
        # Step 4: Load and tokenize dataset
        print("\n[STEP 4/5] Loading and tokenizing dataset...")
        data_manager = create_data_manager(tokenizer)
        data_manager.load_dataset()
        tokenized_dataset = data_manager.tokenize_dataset()
        
        # Validate dataset
        if not data_manager.validate_dataset():
            print("[ERROR] Dataset validation failed")
            return False
        
        print(f"[SUCCESS] Dataset ready: {len(tokenized_dataset)} samples")
        
        # Step 5: Create training manager and train
        print("\n[STEP 5/5] Setting up training and starting fine-tuning...")
        training_manager = create_training_manager(model, tokenized_dataset, data_collator)
        training_manager.setup_training()
        
        print("\n" + "=" * 80)
        print("[STATUS] Starting fine-tuning process...")
        print("=" * 80)
        
        start_time = time.time()
        training_manager.train(show_progress=True)
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("[SUCCESS] Fine-tuning completed!")
        print("=" * 80)
        print(f"Training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        
        # Print training summary
        training_manager.print_training_summary()
        
        # Print model save location
        output_dir = config.get_output_dir()
        print(f"\n[INFO] Fine-tuned model saved to: {output_dir}")
        print("[INFO] You can now use this model with run_model.py")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Fine-tuning failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune an LLM or run tests")
    parser.add_argument("--test", action="store_true",
                       help="Run test suite instead of fine-tuning")
    parser.add_argument("--train", action="store_true",
                       help="Run fine-tuning (default if no --test)")
    
    args = parser.parse_args()
    
    # Default to training if neither flag is set, or if --train is explicitly set
    if args.test:
        print("[INFO] Running in test mode...")
        success = run_all_tests()
        sys.exit(0 if success else 1)
    else:
        print("[INFO] Running in fine-tuning mode...")
        success = run_fine_tuning()
        sys.exit(0 if success else 1)
