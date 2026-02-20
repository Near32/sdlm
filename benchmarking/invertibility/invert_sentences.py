#!/usr/bin/env python3
"""
Sentence Inversion Script

Takes sentences as input and finds prompts that would generate those sentences
using STGS-based optimization.

Usage:
    # Single sentence
    python invert_sentences.py --sentences "Hello world"

    # Multiple sentences
    python invert_sentences.py --sentences "Hello world" "How are you?"

    # From file (one sentence per line)
    python invert_sentences.py --sentence_file sentences.txt

    # With custom model and hyperparameters
    python invert_sentences.py \
        --sentences "Hello world" \
        --model_name "HuggingFaceTB/SmolLM3-3B-Base" \
        --epochs 1000 \
        --seq_len 40
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import wandb

# Import from existing modules
from batch_optimize_main import optimize_for_target, setup_model_and_tokenizer
from metrics_registry import lcs_length

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("invert_sentences")


def create_dataset_from_sentences(
    sentences: List[str],
    tokenizer,
    model_name: str,
    prompt_length: int,
) -> Dict[str, Any]:
    """
    Creates an in-memory dataset dict in SODA-compatible format from sentences.

    Args:
        sentences: List of sentences to invert
        tokenizer: Tokenizer for the model
        model_name: Name of the model being used
        prompt_length: Length of prompt to optimize

    Returns:
        Dataset dictionary compatible with batch_optimize_main
    """
    samples = []
    for idx, sentence in enumerate(sentences):
        # Tokenize the sentence
        tokens = tokenizer(sentence, return_tensors="pt").input_ids[0].tolist()

        sample = {
            "id": idx,
            "text": sentence,
            "tokens": tokens,
            "length": len(tokens),
            "k_target": 1,  # Default for SODA compatibility
        }
        samples.append(sample)

    dataset = {
        "metadata": {
            "model_name": model_name,
            "prompt_length": prompt_length,
            "output_length": max(len(s["tokens"]) for s in samples) if samples else 0,
            "num_samples": len(samples),
            "evaluation_type": "output_match",  # Not prompt reconstruction
            "dataset_source": "user_input",
        },
        "samples": samples,
    }

    return dataset


def invert_sentences(
    sentences: List[str],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Invert sentences to find prompts that would generate them.

    Args:
        sentences: List of sentences to invert
        config: Configuration dictionary with optimization parameters

    Returns:
        List of result dictionaries for each sentence
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set random seeds
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        config["model_name"],
        device,
        model_precision=config.get("model_precision", "full"),
    )

    # Determine sequence length if not specified
    seq_len = config.get("seq_len")
    if seq_len is None:
        # Auto: use 2x the max target length
        max_target_len = max(
            len(tokenizer(s, return_tensors="pt").input_ids[0])
            for s in sentences
        )
        seq_len = max_target_len * 2
        logger.info(f"Auto-determined seq_len: {seq_len} (2x max target length {max_target_len})")
        config["seq_len"] = seq_len

    # Create dataset from sentences
    dataset = create_dataset_from_sentences(
        sentences=sentences,
        tokenizer=tokenizer,
        model_name=config["model_name"],
        prompt_length=seq_len,
    )

    # Create output directory
    output_dir = config.get("output_dir", "./invert_results")
    run_id = wandb.util.generate_id() if not config.get("no_wandb", False) else "local"
    full_output_dir = f"{output_dir}/{run_id}"
    Path(full_output_dir).mkdir(parents=True, exist_ok=True)

    results = []

    for sample in dataset["samples"]:
        logger.info(f"Inverting sentence {sample['id']}: '{sample['text']}'")

        result = optimize_for_target(
            target_info=sample,
            model=model,
            tokenizer=tokenizer,
            device=device,
            config=config,
            run_id=run_id,
            output_dir=full_output_dir,
            metric_groups=None,
            is_prompt_reconstruction=False,
        )

        results.append(result)

    return results


def format_results(results: List[Dict[str, Any]]) -> None:
    """
    Pretty-print results to console.

    Args:
        results: List of result dictionaries
    """
    for i, result in enumerate(results):
        print("\n" + "=" * 60)
        print(f"Sentence {i + 1}: {result['target_text']}")
        print("=" * 60)
        print(f"Inverted Prompt: {result['optimized_text']}")
        print(f"Generated Output: {result['generated_text']}")
        print(f"LCS Ratio: {result['evaluation'].get('lcs_ratio', 0):.4f}")
        print(f"Token Accuracy: {result['evaluation'].get('token_accuracy', 0):.4f}")
        print(f"Final Loss: {result['final_loss']:.6f}")


def save_results(results: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save results to JSON file.

    Args:
        results: List of result dictionaries
        output_file: Path to output file
    """
    # Convert tensor types to serializable formats
    serializable_results = []
    for result in results:
        r = {}
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                r[key] = value.tolist()
            elif isinstance(value, dict):
                r[key] = {
                    k: v.tolist() if isinstance(v, torch.Tensor) else v
                    for k, v in value.items()
                }
            else:
                r[key] = value
        serializable_results.append(r)

    with open(output_file, "w") as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"Results saved to {output_file}")


def str2bool(v):
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Invert sentences to find prompts that would generate them",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input arguments (one required)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--sentences",
        nargs="+",
        type=str,
        help="One or more sentences to invert",
    )
    input_group.add_argument(
        "--sentence_file",
        type=str,
        help="Path to file with sentences (one per line)",
    )

    # Model parameters
    parser.add_argument(
        "--model_name",
        type=str,
        #default="HuggingFaceTB/SmolLM2-135M",
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model name (default: HuggingFaceTB/SmolLM2-135M)",
    )
    parser.add_argument(
        "--model_precision",
        type=str,
        default="full",
        choices=["full", "half"],
        help="Model precision (default: full)",
    )

    # Optimization parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=2048,
        help="Number of training epochs (default: 2048)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Adam learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=100.0,
        help="Gumbel-Softmax temperature (default: 100.0)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="Prompt length to optimize (default: auto = 2x target length)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Optimization batch size (default: 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="stgs",
        choices=["stgs", "reinforce", "soda", "gcg", "o2p"],
        help="Optimization method (default: stgs)",
    )
    parser.add_argument(
        "--losses",
        type=str,
        default="crossentropy",
        help="Loss function (default: crossentropy)",
    )

    # Perplexity parameters for fluency control
    parser.add_argument(
        "--promptLambda",
        type=float,
        default=0.0,
        help="Weight for prompt perplexity loss (higher = more fluent prompts, default: 0.0)",
    )
    parser.add_argument(
        "--complLambda",
        type=float,
        default=0.0,
        help="Weight for completion perplexity loss (default: 0.0)",
    )
    parser.add_argument(
        "--promptTfComplLambda",
        type=float,
        default=1.0,
        help="Weight for prompt + teacher-forced completion perplexity loss (default: 0.0)",
    )

    # STGS-specific parameters
    parser.add_argument(
        "--stgs_hard",
        type=str2bool,
        default=False,
        help="Use hard Straight-Through (default: False)",
    )
    parser.add_argument(
        "--learnable_temperature",
        type=str2bool,
        default=True,
        help="Learn temperature parameter (default: True)",
    )
    parser.add_argument(
        "--decouple_learnable_temperature",
        type=str2bool,
        default=True,
        help="Decouple temperature per position (default: True)",
    )
    parser.add_argument(
        "--teacher_forcing",
        type=str2bool,
        default=True,
        help="Use teacher forcing (default: True)",
    )
    parser.add_argument(
        "--gradient_estimator",
        type=str,
        default="stgs",
        choices=["stgs", "reinforce"],
        help="Gradient estimator (default: stgs)",
    )

    # Output parameters
    parser.add_argument(
        "--output_file",
        type=str,
        default="./output_inverted_sentences.json",
        help="Save results to JSON (default: ./output_inverted_sentences.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./invert_results",
        help="Directory for intermediate results (default: ./invert_results)",
    )

    # Logging parameters
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="sentence-inversion",
        help="W&B project name (default: sentence-inversion)",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity name",
    )

    # Other parameters (matching batch_optimize_main defaults)
    parser.add_argument("--plot_every", type=int, default=100)
    parser.add_argument("--filter_vocab", type=str2bool, default=False)
    parser.add_argument("--vocab_threshold", type=float, default=0.5)
    parser.add_argument("--max_gradient_norm", type=float, default=0.0)
    parser.add_argument("--eps", type=float, default=1e-10)
    parser.add_argument("--bptt", type=str2bool, default=False)
    parser.add_argument("--bptt_temperature", type=float, default=1.0)
    parser.add_argument("--bptt_decouple_learnable_temperature", type=str2bool, default=False)
    parser.add_argument("--bptt_learnable_temperature", type=str2bool, default=False)
    parser.add_argument("--bptt_stgs_hard", type=str2bool, default=True)
    parser.add_argument("--bptt_hidden_state_conditioning", type=str2bool, default=False)
    parser.add_argument("--bptt_eps", type=float, default=1e-10)
    parser.add_argument("--reinforce_reward_scale", type=float, default=1.0)
    parser.add_argument("--reinforce_use_baseline", type=str2bool, default=True)
    parser.add_argument("--reinforce_baseline_beta", type=float, default=0.9)
    parser.add_argument("--stgs_grad_variance_samples", type=int, default=0)
    parser.add_argument("--stgs_grad_variance_period", type=int, default=1)
    parser.add_argument("--stgs_grad_bias_samples", type=int, default=0)
    parser.add_argument("--stgs_grad_bias_period", type=int, default=1)
    parser.add_argument("--stgs_grad_bias_reference_samples", type=int, default=0)
    parser.add_argument("--stgs_grad_bias_reference_batch_size", type=int, default=0)
    parser.add_argument("--stgs_grad_bias_reference_use_baseline", type=str2bool, default=True)
    parser.add_argument("--stgs_grad_bias_reference_reward_scale", type=float, default=1.0)
    parser.add_argument("--stgs_grad_bias_reference_baseline_beta", type=float, default=0.9)
    parser.add_argument("--reinforce_grad_variance_samples", type=int, default=0)
    parser.add_argument("--reinforce_grad_variance_period", type=int, default=1)

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load sentences from file if provided
    if args.sentence_file:
        with open(args.sentence_file, "r") as f:
            sentences = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(sentences)} sentences from {args.sentence_file}")
    else:
        sentences = args.sentences

    if not sentences:
        logger.error("No sentences provided")
        sys.exit(1)

    logger.info(f"Inverting {len(sentences)} sentence(s)")

    # Handle W&B mode
    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    # Create config from args
    config = vars(args)

    # Update losses with perplexity components if needed
    if config["promptLambda"] > 0.0:
        config["losses"] += "+promptPerplexity"
    if config["complLambda"] > 0.0:
        config["losses"] += "+completionPerplexity"
    if config["promptTfComplLambda"] > 0.0:
        config["losses"] += "+promptTfComplPerplexity"

    # Run inversion
    results = invert_sentences(sentences, config)

    # Display results
    format_results(results)

    # Save results to JSON
    save_results(results, args.output_file)


if __name__ == "__main__":
    main()
