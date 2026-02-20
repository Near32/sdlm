#!/usr/bin/env python3
"""
Dataset generator for SODA-style prompt reconstruction evaluation.

Generates ground-truth prompts (random tokens or Wikipedia text) and outputs
via greedy decoding from the model. Saves (prompt, output) pairs in JSON format
compatible with batch_optimize_main.py.
"""
import argparse
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("generate_gt_prompt_output_dataset")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(model_name: str, device: str = "auto"):
    """Load the model and tokenizer."""
    logger.info(f"Loading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def generate_random_prompt_tokens(
    tokenizer,
    prompt_length: int,
    include_bos: bool = False,
    vocab_size: Optional[int] = None
) -> List[int]:
    """
    Generate random token IDs for a prompt.

    Args:
        tokenizer: The tokenizer
        prompt_length: Number of tokens to generate
        include_bos: Whether to include BOS token at the start
        vocab_size: Vocabulary size (default: tokenizer.vocab_size)

    Returns:
        List of token IDs
    """
    if vocab_size is None:
        vocab_size = tokenizer.vocab_size

    tokens = []

    if include_bos and tokenizer.bos_token_id is not None:
        tokens.append(tokenizer.bos_token_id)
        prompt_length -= 1

    # Generate random tokens, avoiding special tokens
    special_tokens = set()
    if tokenizer.pad_token_id is not None:
        special_tokens.add(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        special_tokens.add(tokenizer.eos_token_id)
    if tokenizer.bos_token_id is not None:
        special_tokens.add(tokenizer.bos_token_id)
    if tokenizer.unk_token_id is not None:
        special_tokens.add(tokenizer.unk_token_id)

    valid_tokens = [i for i in range(vocab_size) if i not in special_tokens]

    for _ in range(prompt_length):
        token_id = random.choice(valid_tokens)
        tokens.append(token_id)

    return tokens


def load_wikipedia_dataset(split: str = "train", cache_dir: Optional[str] = None):
    """Load the Wikipedia dataset."""
    try:
        from datasets import load_dataset
        logger.info(f"Loading Wikipedia dataset (split: {split})")
        dataset = load_dataset(
            "lucadiliello/english_wikipedia",
            split=split,
            cache_dir=cache_dir
        )
        return dataset
    except Exception as e:
        logger.error(f"Error loading Wikipedia dataset: {e}")
        raise


def extract_wikipedia_prompt(
    text: str,
    tokenizer,
    prompt_length: int,
    random_sentence: bool = False,
    random_start: bool = False
) -> Optional[List[int]]:
    """
    Extract a prompt from Wikipedia text.

    Args:
        text: The Wikipedia article text
        tokenizer: The tokenizer
        prompt_length: Desired prompt length in tokens
        random_sentence: If True, select a random sentence from the text
        random_start: If True, start from a random position in the sentence/text

    Returns:
        List of token IDs or None if extraction fails
    """
    if not text or len(text.strip()) == 0:
        return None

    # Clean up the text
    text = text.strip()

    if random_sentence:
        # Split into sentences (simple heuristic)
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
        if not sentences:
            sentences = [text]
        text = random.choice(sentences)

    # Tokenize the text
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) < prompt_length:
        return None

    if random_start:
        # Choose a random starting position
        max_start = len(tokens) - prompt_length
        start_idx = random.randint(0, max_start) if max_start > 0 else 0
    else:
        start_idx = 0

    return tokens[start_idx:start_idx + prompt_length]


def generate_greedy_output(
    model,
    tokenizer,
    prompt_tokens: List[int],
    output_length: int,
    device: str = "cuda"
) -> List[int]:
    """
    Generate output tokens via greedy decoding.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt_tokens: Input prompt token IDs
        output_length: Number of tokens to generate
        device: Device to run on

    Returns:
        List of generated token IDs (excluding prompt)
    """
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long)

    # Move to model's device
    if hasattr(model, 'device'):
        input_ids = input_ids.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=output_length,
            do_sample=False,  # Greedy decoding
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=None,  # Don't stop at EOS for fixed-length output
        )

    # Extract only the generated tokens (excluding prompt)
    generated_tokens = outputs[0, len(prompt_tokens):].cpu().tolist()

    # Ensure we have exactly output_length tokens
    if len(generated_tokens) < output_length:
        # Pad with pad_token or eos_token if needed
        pad_token = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        generated_tokens.extend([pad_token] * (output_length - len(generated_tokens)))
    elif len(generated_tokens) > output_length:
        generated_tokens = generated_tokens[:output_length]

    return generated_tokens


def generate_dataset(
    model_name: str,
    dataset_source: str,
    output_path: str,
    prompt_length: int,
    output_length: int = 25,
    num_samples: int = 100,
    seed: int = 42,
    include_bos: bool = False,
    random_sentence: bool = False,
    random_start: bool = False,
    wikipedia_split: str = "train",
    cache_dir: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate the SODA-style dataset.

    Args:
        model_name: HuggingFace model name
        dataset_source: "random" or "wikipedia"
        output_path: Path to save the dataset JSON
        prompt_length: Length of prompt in tokens
        output_length: Length of output in tokens
        num_samples: Number of samples to generate
        seed: Random seed
        include_bos: Whether to include BOS token
        random_sentence: For Wikipedia, select random sentence
        random_start: For Wikipedia, random start position
        wikipedia_split: Wikipedia dataset split
        cache_dir: Cache directory for datasets
        wandb_project: W&B project for artifact logging
        wandb_entity: W&B entity

    Returns:
        The generated dataset dictionary
    """
    set_seed(seed)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Prepare metadata
    metadata = {
        "model_name": model_name,
        "dataset_source": dataset_source,
        "prompt_length": prompt_length,
        "output_length": output_length,
        "generation_method": "greedy",
        "num_samples": num_samples,
        "seed": seed,
        "include_bos": include_bos,
        "evaluation_type": "prompt_reconstruction",
        "created_at": datetime.now().isoformat(),
    }

    if dataset_source == "wikipedia":
        metadata["dataset_split"] = wikipedia_split
        metadata["random_sentence"] = random_sentence
        metadata["random_start"] = random_start

    samples = []

    # Load Wikipedia dataset if needed
    wikipedia_data = None
    wikipedia_indices = None
    if dataset_source == "wikipedia":
        wikipedia_data = load_wikipedia_dataset(wikipedia_split, cache_dir)
        # Pre-shuffle indices for reproducibility
        wikipedia_indices = list(range(len(wikipedia_data)))
        random.shuffle(wikipedia_indices)

    logger.info(f"Generating {num_samples} samples...")

    sample_idx = 0
    wikipedia_ptr = 0

    with tqdm(total=num_samples, desc="Generating samples") as pbar:
        while sample_idx < num_samples:
            if dataset_source == "random":
                # Generate random prompt tokens
                prompt_tokens = generate_random_prompt_tokens(
                    tokenizer,
                    prompt_length,
                    include_bos=include_bos
                )
                sample_id = f"random_{sample_idx}"

            elif dataset_source == "wikipedia":
                # Extract prompt from Wikipedia
                if wikipedia_ptr >= len(wikipedia_indices):
                    logger.warning("Exhausted Wikipedia dataset, recycling indices")
                    random.shuffle(wikipedia_indices)
                    wikipedia_ptr = 0

                wiki_idx = wikipedia_indices[wikipedia_ptr]
                wikipedia_ptr += 1

                article = wikipedia_data[wiki_idx]
                text = article.get("maintext", "") or article.get("text", "")

                prompt_tokens = extract_wikipedia_prompt(
                    text,
                    tokenizer,
                    prompt_length,
                    random_sentence=random_sentence,
                    random_start=random_start
                )

                if prompt_tokens is None:
                    continue  # Skip this article, try another

                sample_id = f"wikipedia_{sample_idx}"
            else:
                raise ValueError(f"Unknown dataset source: {dataset_source}")

            # Generate output via greedy decoding
            output_tokens = generate_greedy_output(
                model,
                tokenizer,
                prompt_tokens,
                output_length
            )

            # Decode texts
            prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=False)
            output_text = tokenizer.decode(output_tokens, skip_special_tokens=False)

            sample = {
                "id": sample_id,
                "ground_truth_prompt": prompt_text,
                "ground_truth_prompt_tokens": prompt_tokens,
                "text": output_text,  # For backward compatibility
                "target_output_tokens": output_tokens,
                "prompt_length": len(prompt_tokens),
                "output_length": len(output_tokens),
                # For backward compatibility with existing pipeline
                "k_target": 1,  # Greedy = top-1
            }

            samples.append(sample)
            sample_idx += 1
            pbar.update(1)

    dataset = {
        "metadata": metadata,
        "samples": samples
    }

    # Save dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add .json extension if not present
    if not str(output_path).endswith('.json'):
        output_path = output_path.with_suffix('.json')

    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    logger.info(f"Dataset saved to {output_path}")
    logger.info(f"Generated {len(samples)} samples")

    # Log to W&B as artifact if configured
    if wandb_project:
        try:
            import wandb

            run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                job_type="dataset_generation",
                config=metadata
            )

            artifact = wandb.Artifact(
                name=f"soda_dataset_{dataset_source}_PL{prompt_length}_OL{output_length}",
                type="dataset",
                metadata=metadata
            )
            artifact.add_file(str(output_path))
            run.log_artifact(artifact)

            run.finish()
            logger.info("Dataset logged to W&B as artifact")
        except Exception as e:
            logger.warning(f"Failed to log to W&B: {e}")

    return dataset


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate SODA-style prompt reconstruction dataset"
    )

    # Required arguments
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model name (e.g., HuggingFaceTB/SmolLM2-135M)"
    )
    parser.add_argument(
        "--dataset_source",
        type=str,
        required=True,
        choices=["random", "wikipedia"],
        help="Source of prompts: 'random' for random tokens, 'wikipedia' for Wikipedia text"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the dataset JSON file"
    )

    # Optional arguments
    parser.add_argument(
        "--prompt_length",
        type=int,
        default=10,
        help="Length of prompt in tokens (default: 10)"
    )
    parser.add_argument(
        "--output_length",
        type=int,
        default=25,
        help="Number of tokens to generate via greedy decoding (default: 25)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to generate (default: 100)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--include_bos",
        action="store_true",
        help="Include BOS token at the start of prompts"
    )

    # Wikipedia-specific arguments
    parser.add_argument(
        "--random_sentence",
        action="store_true",
        help="(Wikipedia only) Sample a random sentence from the text"
    )
    parser.add_argument(
        "--random_start",
        action="store_true",
        help="(Wikipedia only) Use a random starting position in the sentence/text"
    )
    parser.add_argument(
        "--wikipedia_split",
        type=str,
        default="train",
        help="Wikipedia dataset split (default: train)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for datasets"
    )

    # W&B arguments
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project for logging dataset as artifact"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    generate_dataset(
        model_name=args.model_name,
        dataset_source=args.dataset_source,
        output_path=args.output_path,
        prompt_length=args.prompt_length,
        output_length=args.output_length,
        num_samples=args.num_samples,
        seed=args.seed,
        include_bos=args.include_bos,
        random_sentence=args.random_sentence,
        random_start=args.random_start,
        wikipedia_split=args.wikipedia_split,
        cache_dir=args.cache_dir,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )


if __name__ == "__main__":
    main()
