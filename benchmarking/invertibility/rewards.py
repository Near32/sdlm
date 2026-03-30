"""
Custom reward functions for TRL GRPOTrainer on GSM8K.

Provides two approaches:
1. Base approach: Direct extraction of #### <number> format
2. Extractor-based approach: Append extraction prompt and check completion
"""

import math
import re
from typing import List, Union, Optional, Callable
from dataclasses import dataclass


def extract_gsm8k_answer(text: str) -> str:
    """
    Extract numerical answer from GSM8K format (#### <number>).

    Args:
        text: Model output or ground truth containing #### marker

    Returns:
        Extracted number as string, or empty string if not found
    """
    # Look for #### followed by a number (possibly with commas, decimals, negatives)
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        # Remove commas from number
        return match.group(1).replace(',', '')
    return ""


def extract_boxed_answer(text: str) -> str:
    """
    Extract answer from LaTeX \\boxed{} format.

    Args:
        text: Model output potentially containing \\boxed{answer}

    Returns:
        Content inside boxed, or empty string if not found
    """
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()
    return ""


def extract_final_number(text: str) -> str:
    """
    Extract the last number mentioned in the text as a fallback.

    Args:
        text: Model output

    Returns:
        Last number found, or empty string
    """
    # Find all numbers in the text
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    if numbers:
        return numbers[-1].replace(',', '')
    return ""


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison.

    Args:
        answer: Raw extracted answer

    Returns:
        Normalized answer string
    """
    if not answer:
        return ""

    # Remove commas and extra whitespace
    answer = answer.replace(',', '').strip()

    # Try to convert to float and back to handle formatting differences
    try:
        num = float(answer)
        if not math.isfinite(num):
            return answer
        # If it's a whole number, return as int string
        if num == int(num):
            return str(int(num))
        return str(num)
    except ValueError:
        return answer


def compare_answers(extracted: str, ground_truth: str) -> bool:
    """
    Compare extracted answer with ground truth.

    Args:
        extracted: Model's extracted answer
        ground_truth: Ground truth answer

    Returns:
        True if answers match
    """
    norm_extracted = normalize_answer(extracted)
    norm_gt = normalize_answer(ground_truth)

    if not norm_extracted or not norm_gt:
        return False

    return norm_extracted == norm_gt


def gsm8k_exact_match_reward(
    completions: List[Union[str, List[dict]]],
    ground_truth: List[str],
    use_extractor: bool = False,
    **kwargs
) -> List[float]:
    """
    TRL-compatible reward function for GSM8K.

    Args:
        completions: List of model completions (strings or conversation format)
        ground_truth: List of ground truth answers (#### format or raw numbers)
        use_extractor: If True, use extractor-based approach (not implemented in base)
        **kwargs: Additional arguments (ignored)

    Returns:
        List of binary rewards: 1.0 if correct, 0.0 otherwise
    """
    rewards = []

    for completion, gt in zip(completions, ground_truth):
        # Handle conversational format if needed
        if isinstance(completion, list):
            # Extract last assistant message
            for msg in reversed(completion):
                if msg.get("role") == "assistant":
                    completion = msg.get("content", "")
                    break
            else:
                completion = ""

        # Extract answer from completion
        extracted = extract_gsm8k_answer(completion)

        # Fallback to boxed format
        if not extracted:
            extracted = extract_boxed_answer(completion)

        # Extract ground truth answer
        gt_answer = extract_gsm8k_answer(gt) if "####" in gt else gt

        # Compare and assign reward
        if compare_answers(extracted, gt_answer):
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards


@dataclass
class ExtractorRewardConfig:
    """Configuration for extractor-based reward."""
    extraction_prompt: str = "Therefore, the final answer is "
    max_extraction_tokens: int = 20
    model: Optional[object] = None
    tokenizer: Optional[object] = None


class ExtractorBasedReward:
    """
    Extractor-based reward function.

    After the reasoning, appends an answer-extracting prompt and lets the LM
    complete it for a few tokens, then checks whether those tokens contain
    the ground truth answer.
    """

    def __init__(self, config: ExtractorRewardConfig):
        """
        Initialize extractor-based reward.

        Args:
            config: Configuration for extraction
        """
        self.config = config
        self.extraction_prompt = config.extraction_prompt
        self.max_tokens = config.max_extraction_tokens
        self.model = config.model
        self.tokenizer = config.tokenizer

    def __call__(
        self,
        completions: List[Union[str, List[dict]]],
        ground_truth: List[str],
        **kwargs
    ) -> List[float]:
        """
        Compute rewards using extractor-based approach.

        Args:
            completions: List of model completions
            ground_truth: List of ground truth answers
            **kwargs: Additional arguments

        Returns:
            List of rewards
        """
        rewards = []

        for completion, gt in zip(completions, ground_truth):
            # Handle conversational format
            if isinstance(completion, list):
                for msg in reversed(completion):
                    if msg.get("role") == "assistant":
                        completion = msg.get("content", "")
                        break
                else:
                    completion = ""

            # Try direct extraction first
            extracted = extract_gsm8k_answer(completion)
            if not extracted:
                extracted = extract_boxed_answer(completion)

            # If direct extraction fails and model is available, use extractor
            if not extracted and self.model is not None and self.tokenizer is not None:
                extracted = self._extract_with_model(completion)

            # Fallback to last number if still nothing
            if not extracted:
                extracted = extract_final_number(completion)

            # Extract ground truth answer
            gt_answer = extract_gsm8k_answer(gt) if "####" in gt else gt

            # Compare and assign reward
            if compare_answers(extracted, gt_answer):
                rewards.append(1.0)
            else:
                rewards.append(0.0)

        return rewards

    def _extract_with_model(self, completion: str) -> str:
        """
        Use model to extract answer by completing extraction prompt.

        Args:
            completion: Model's reasoning completion

        Returns:
            Extracted answer
        """
        import torch

        # Append extraction prompt
        full_text = completion + "\n\n" + self.extraction_prompt

        # Tokenize
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=4096 - self.max_tokens
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate completion
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only new tokens
        new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        extracted_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Extract number from the completion
        return extract_final_number(extracted_text)


def create_reward_function(
    use_extractor: bool = False,
    model: Optional[object] = None,
    tokenizer: Optional[object] = None,
    extraction_prompt: str = "Therefore, the final answer is ",
    max_extraction_tokens: int = 20,
) -> Callable:
    """
    Factory function to create the appropriate reward function.

    Args:
        use_extractor: Whether to use extractor-based approach
        model: Model for extraction (required if use_extractor=True)
        tokenizer: Tokenizer for extraction (required if use_extractor=True)
        extraction_prompt: Prompt to append for extraction
        max_extraction_tokens: Max tokens to generate for extraction

    Returns:
        Reward function compatible with TRL
    """
    if use_extractor:
        config = ExtractorRewardConfig(
            extraction_prompt=extraction_prompt,
            max_extraction_tokens=max_extraction_tokens,
            model=model,
            tokenizer=tokenizer,
        )
        return ExtractorBasedReward(config)
    else:
        return gsm8k_exact_match_reward


# Reward function that can be used directly with TRL's reward_funcs parameter
def make_trl_reward_fn(use_extractor: bool = False):
    """
    Create a reward function compatible with TRL GRPOTrainer.

    The returned function has the signature expected by TRL:
    reward_fn(completions, prompts=None, **kwargs) -> List[float]

    Note: Ground truth must be passed via kwargs or stored in a closure.
    """
    def reward_fn(completions, prompts=None, ground_truth=None, **kwargs):
        if ground_truth is None:
            # Try to get from kwargs
            ground_truth = kwargs.get("answer", kwargs.get("answers", []))

        return gsm8k_exact_match_reward(
            completions=completions,
            ground_truth=ground_truth,
            use_extractor=use_extractor,
            **kwargs
        )

    return reward_fn
