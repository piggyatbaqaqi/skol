"""
Mistral Transfer Learning Module

This module provides utilities for fine-tuning Mistral models for feature extraction
from species descriptions. It includes functions for data loading, prompt generation,
tokenization, model configuration, and evaluation metrics.
"""

import os
import sys
import json
import io
import glob
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
from numpy.linalg import norm

# Deep learning imports
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from huggingface_hub import login
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig
)
import datasets


# Constants and Configuration
SEED = 12345
DEFAULT_MAX_LENGTH = 2048
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Default prompt for feature extraction
DEFAULT_PROMPT = '''Please extract features, subfeatures, optional subsubfeatures, and values from the following species description.
Format the output as JSON.
The top level of the JSON is feature names. The next level in is subfeature names . The optional next level in is subsubfeature names.
The innermost layer is lists of string-valued values.
Lists are only present at the innermost level of the JSON.
Feature values that are comma-separated strings should be broken down into separate values.
Translate Latin paragraphs to English.
'''


# Prompt Generation
def make_prompt(prompt: str, description: str, result: Optional[str] = None) -> str:
    """
    Create a formatted prompt for the Mistral model.

    Args:
        prompt: The instruction prompt
        description: The species description text
        result: Optional expected result in JSON format

    Returns:
        Formatted prompt string for the model
    """
    retval = f"""<s>[INST]{prompt}

Here is the description:
{description}[/INST]

Result:
"""
    if result is not None:
        retval += f"""
```json
{result}
```
</s>
"""
    return retval


# File Management
def listFiles(folder: str) -> List[str]:
    """
    List all txt files under a folder path, excluding Sydowia files.

    Args:
        folder: Path to the folder to search

    Returns:
        List of file paths
    """
    try:
        files = [file for file in glob.glob(f'{folder}/**/*.txt*', recursive=True)
                 if 'Sydowia' not in file]
        return files
    except FileNotFoundError:
        print(f"Folder '{folder}' not found.")
        return []
    except PermissionError:
        print(f"Permission denied to access folder '{folder}'.")
        return []


# Data Loading
def load_json_training(filename: str) -> List[Dict[str, Any]]:
    """
    Load training data from a specially formatted text file.

    The file format alternates between "Send to LLM:" sections (descriptions)
    and "Result:" sections (JSON results).

    Args:
        filename: Path to the training data file

    Returns:
        List of dictionaries with 'description' and 'result' keys
    """
    retval = []

    def val(description: str, result: Dict[str, Any]) -> Dict[str, str]:
        return {
            'description': description,
            'result': json.dumps(result, indent=4, ensure_ascii=False)
        }

    state = 'START'  # 'description', 'result'
    with open(filename, "r", encoding="utf-8") as file:
        lines = []
        description = ''
        for line in file:
            if line.startswith('Send to LLM:'):
                if state == "result":
                    result = ''.join(lines)
                    try:
                        result_dict = json.loads(result)
                    except json.JSONDecodeError as err:
                        print(f'Err: {err}\n{result}')
                    else:
                        retval.append(val(description, result_dict))
                lines = []
                state = 'description'
            elif line.startswith('Result:'):
                if state == "description":
                    description = ''.join(lines)
                    lines = []
                state = 'result'
            else:
                lines.append(line)

        if state == 'result' and len(lines) > 0:
            result = ''.join(lines)
            try:
                result_dict = json.loads(result)
            except json.JSONDecodeError as err:
                print(f'Err: {err}\n{result}')
            else:
                retval.append(val(description, result_dict))

    return retval


# Dataset Preparation
def create_train_eval_test_split(json_training: List[Dict[str, Any]]) -> Tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]:
    """
    Split training data into train, eval, and test datasets.

    Args:
        json_training: List of training examples

    Returns:
        Tuple of (train_dataset, eval_dataset, test_dataset)
    """
    dataset = datasets.Dataset.from_list(json_training)

    # First split: separate test set (1 example)
    new_dataset = datasets.Dataset.train_test_split(dataset, int(1), shuffle=False)
    temp_dataset = new_dataset["train"]
    test_dataset = new_dataset["test"]

    # Second split: separate eval set (1 example)
    new_dataset2 = datasets.Dataset.train_test_split(temp_dataset, int(1), shuffle=False)
    train_dataset = new_dataset2["train"]
    eval_dataset = new_dataset2["test"]

    return train_dataset, eval_dataset, test_dataset


# Model Configuration
def create_bnb_config(
    load_in_4bit: bool = True,
    bnb_4bit_use_double_quant: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    compute_dtype: torch.dtype = torch.bfloat16
) -> BitsAndBytesConfig:
    """
    Create a BitsAndBytes quantization configuration.

    Args:
        load_in_4bit: Whether to load model in 4-bit precision
        bnb_4bit_use_double_quant: Use double quantization
        bnb_4bit_quant_type: Quantization type (e.g., "nf4")
        compute_dtype: Computation data type

    Returns:
        BitsAndBytesConfig object
    """
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype
    )


def create_lora_config(
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    bias: str = "none",
    task_type: str = "CAUSAL_LM"
) -> LoraConfig:
    """
    Create a LoRA (Low-Rank Adaptation) configuration.

    Args:
        r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: Dropout rate
        bias: Bias configuration
        task_type: Type of task

    Returns:
        LoraConfig object
    """
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias=bias,
        lora_dropout=lora_dropout,
        task_type=task_type,
    )


def setup_accelerator() -> Accelerator:
    """
    Set up an accelerator for distributed training with FSDP.

    Returns:
        Accelerator object
    """
    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    )
    return Accelerator(fsdp_plugin=fsdp_plugin)


def print_trainable_parameters(model):
    """
    Print the number of trainable parameters in the model.

    Args:
        model: The model to inspect
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param}"
    )


# Tokenization
def create_tokenizer(
    model_id: str = BASE_MODEL_ID,
    max_length: int = DEFAULT_MAX_LENGTH
) -> AutoTokenizer:
    """
    Create and configure a tokenizer.

    Args:
        model_id: Hugging Face model ID
        max_length: Maximum sequence length

    Returns:
        Configured tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        model_max_length=max_length,
        padding_side="left",
        add_eos_token=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def create_tokenize_function(tokenizer, max_length: int = DEFAULT_MAX_LENGTH):
    """
    Create a tokenization function for a given tokenizer.

    Args:
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length

    Returns:
        Tokenization function
    """
    def tokenize(prompt):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    return tokenize


def create_prompt_tokenizer(prompt_text: str, tokenizer, max_length: int = DEFAULT_MAX_LENGTH):
    """
    Create a function that generates and tokenizes prompts from data points.

    Args:
        prompt_text: The prompt template to use
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length

    Returns:
        Function that processes data points
    """
    tokenize_fn = create_tokenize_function(tokenizer, max_length)

    def generate_and_tokenize_prompt(data_point):
        full_prompt = make_prompt(prompt_text, data_point["description"], data_point["result"])
        return tokenize_fn(full_prompt)

    return generate_and_tokenize_prompt


# Model Loading
def load_base_model(
    model_id: str = BASE_MODEL_ID,
    bnb_config: Optional[BitsAndBytesConfig] = None,
    device_map: str = "auto",
    trust_remote_code: bool = True,
    use_auth_token: bool = True
) -> AutoModelForCausalLM:
    """
    Load a base model with optional quantization.

    Args:
        model_id: Hugging Face model ID
        bnb_config: BitsAndBytes configuration
        device_map: Device mapping strategy
        trust_remote_code: Whether to trust remote code
        use_auth_token: Whether to use authentication token

    Returns:
        Loaded model
    """
    if bnb_config is None:
        bnb_config = create_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        use_auth_token=use_auth_token
    )
    return model


def prepare_model_for_training(model, lora_config: Optional[LoraConfig] = None):
    """
    Prepare a model for k-bit training with LoRA.

    Args:
        model: The base model
        lora_config: LoRA configuration (if None, uses default)

    Returns:
        Model prepared for training
    """
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    if lora_config is None:
        lora_config = create_lora_config()

    model = get_peft_model(model, lora_config)
    return model


def load_finetuned_model(
    base_model,
    checkpoint_path: str,
    tokenizer
) -> PeftModel:
    """
    Load a fine-tuned model from a checkpoint.

    Args:
        base_model: The base model
        checkpoint_path: Path to the checkpoint
        tokenizer: The tokenizer

    Returns:
        Fine-tuned model
    """
    ft_model = PeftModel.from_pretrained(base_model, checkpoint_path)
    ft_model.eval()
    return ft_model


def enable_multi_gpu(model):
    """
    Enable multi-GPU support if available.

    Args:
        model: The model to configure
    """
    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True


# JSON Extraction and Evaluation
def extract_json(md: str) -> Dict[str, Any]:
    """
    Extract the first JSON object from a Markdown string.

    Args:
        md: Markdown string containing JSON

    Returns:
        Parsed JSON object
    """
    state = "START"
    lines = []
    with io.StringIO(md) as f:
        for line in f:
            if line.startswith('```json') or line.startswith("result:"):
                state = "RECORDING"
            elif line.startswith('```'):
                state = "END"
                return json.loads("\n".join(lines))
            elif line.startswith("}"):
                lines.append(line)
                state = "END"
                return json.loads("\n".join(lines))
            elif state == "RECORDING":
                lines.append(line)
    return json.loads("\n".join(lines))


def key_value_sets(json_obj) -> Tuple[Set, Set]:
    """
    Recursively extract all keys and values from a JSON object.

    Args:
        json_obj: JSON object (dict, list, or primitive)

    Returns:
        Tuple of (keys_set, values_set)
    """
    keys = set()
    values = set()

    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            keys.add(key)
            subkeys, subvalues = key_value_sets(value)
            keys.update(subkeys)
            values.update(subvalues)
    elif isinstance(json_obj, list):
        for value in json_obj:
            subkeys, subvalues = key_value_sets(value)
            keys.update(subkeys)
            values.update(subvalues)
    elif isinstance(json_obj, str):
        values.add(json_obj)
    else:
        values.add(str(json_obj))

    return (keys, values)


def jaccard_distance(set1: Set, set2: Set) -> float:
    """
    Calculate Jaccard distance between two sets.

    Args:
        set1: First set
        set2: Second set

    Returns:
        Jaccard distance (1 - Jaccard index)
    """
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if len(union) == 0:
        return 0.0
    jaccard_index = len(intersection) / len(union)
    return 1 - jaccard_index


def jaccard_distance_json(json1: Dict[str, Any], json2: Dict[str, Any]) -> float:
    """
    Calculate Jaccard distance between two JSON objects.

    Compares both the keys/subkeys and values, returning the average
    Jaccard distance.

    Args:
        json1: First JSON object
        json2: Second JSON object

    Returns:
        Average Jaccard distance
    """
    json1_keys, json1_vals = key_value_sets(json1)
    json2_keys, json2_vals = key_value_sets(json2)

    j_key = jaccard_distance(json1_keys, json2_keys)
    j_val = jaccard_distance(json1_vals, json2_vals)

    avg_jaccard_distance = (j_key + j_val) / 2

    return avg_jaccard_distance


# Inference Helper
def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_LENGTH,
    device: str = "cuda"
) -> str:
    """
    Generate a response from the model given a prompt.

    Args:
        model: The model to use
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        device: Device to run on

    Returns:
        Generated text
    """
    model_input = tokenizer(prompt, return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        output = model.generate(
            **model_input,
            max_new_tokens=max_new_tokens,
            pad_token_id=2
        )[0]
        return tokenizer.decode(output, skip_special_tokens=True)


def compare_models(
    base_model,
    ft_model,
    tokenizer,
    test_prompt: str,
    ground_truth: Dict[str, Any],
    max_length: int = DEFAULT_MAX_LENGTH
) -> Dict[str, Any]:
    """
    Compare base model and fine-tuned model performance.

    Args:
        base_model: Base model
        ft_model: Fine-tuned model
        tokenizer: Tokenizer
        test_prompt: Test prompt
        ground_truth: Ground truth JSON
        max_length: Maximum sequence length

    Returns:
        Dictionary with comparison results
    """
    # Generate outputs
    base_output = generate_response(base_model, tokenizer, test_prompt, max_length)
    ft_output = generate_response(ft_model, tokenizer, test_prompt, max_length)

    # Extract JSON
    base_result = extract_json(base_output.lower())
    ft_result = extract_json(ft_output.lower())

    # Calculate distances
    base_distance = jaccard_distance_json(base_result, ground_truth)
    ft_distance = jaccard_distance_json(ft_result, ground_truth)

    return {
        'base_output': base_output,
        'ft_output': ft_output,
        'base_result': base_result,
        'ft_result': ft_result,
        'base_jaccard_distance': base_distance,
        'ft_jaccard_distance': ft_distance,
        'improvement': base_distance - ft_distance
    }
