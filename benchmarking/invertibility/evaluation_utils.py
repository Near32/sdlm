"""
Utilities for evaluating generated outputs against targets.
"""
import logging
from typing import Dict, List, Any, Optional

import numpy as np
import torch

# Import compute_all_metrics but provide a fallback implementation in case metrics_registry isn't available
try:
    from metrics_registry import compute_all_metrics
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    logging.warning("metrics_registry not available, using fallback implementation")

logger = logging.getLogger("evaluation_utils")

# Global caches for semantic models
_sentencebert_cache = {}

def get_cached_sentencebert(model_name: str, device: str):
    """Get or create cached SentenceTransformer model."""
    cache_key = f"{model_name}_{device}"
    if cache_key not in _sentencebert_cache:
        from sentence_transformers import SentenceTransformer
        _sentencebert_cache[cache_key] = SentenceTransformer(model_name, device=device)
    return _sentencebert_cache[cache_key]


def _to_serializable(obj: Any) -> Any:
    """
    Convert objects (numpy scalars/arrays, torch tensors) to JSON-serializable representations.
    """
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()
    return obj

def compute_sentencebert_similarity(generated_text, reference_text, model_name="all-MiniLM-L6-v2", batch_size=32, device=None):
    """
    Compute semantic similarity between generated and reference texts using SentenceBERT.
    
    Args:
        generated_text: Generated text (string or list of strings)
        reference_text: Reference text (string or list of strings)
        model_name: SentenceBERT model to use
        batch_size: Batch size for encoding
        device: Device to run on
        
    Returns:
        Dictionary with similarity score
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        import torch
        
        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Ensure inputs are lists
        if isinstance(generated_text, str):
            generated_text = [generated_text]
        if isinstance(reference_text, str):
            reference_text = [reference_text]
        
        # Load model
        model = SentenceTransformer(model_name, device=device)
        
        # Encode sentences
        generated_embeddings = model.encode(generated_text, batch_size=batch_size, show_progress_bar=False)
        reference_embeddings = model.encode(reference_text, batch_size=batch_size, show_progress_bar=False)
        
        # Compute similarities
        similarities = []
        for gen_emb, ref_emb in zip(generated_embeddings, reference_embeddings):
            # Normalize vectors
            gen_norm = gen_emb / np.linalg.norm(gen_emb)
            ref_norm = ref_emb / np.linalg.norm(ref_emb)
            
            # Compute cosine similarity
            similarity = np.dot(gen_norm, ref_norm)
            similarities.append(similarity)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        return {"sentencebert_similarity": avg_similarity}
    except Exception as e:
        logger.error(f"Error computing SentenceBERT similarity: {e}")
        return {"sentencebert_similarity": 0.0}

def compute_bertscore(generated_text, reference_text, model_type="distilbert-base-uncased", batch_size=32, device=None):
    """
    Compute BERTScore for generated and reference texts.
    
    Args:
        generated_text: Generated text (string or list of strings)
        reference_text: Reference text (string or list of strings)
        model_type: Model to use for BERTScore
        batch_size: Batch size for processing
        device: Device to run on
        
    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    try:
        import bert_score
        import torch
        
        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Ensure inputs are lists
        if isinstance(generated_text, str):
            generated_text = [generated_text]
        if isinstance(reference_text, str):
            reference_text = [reference_text]
        
        # Compute BERTScore
        P, R, F1 = bert_score.score(
            generated_text, reference_text, 
            model_type=model_type,
            batch_size=batch_size,
            device=device,
            verbose=False
        )
        
        # Extract values
        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item()
        }
    except Exception as e:
        logger.error(f"Error computing BERTScore: {e}")
        return {
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0,
            "bertscore_f1": 0.0
        }

# Fallback implementation for basic metrics
def _compute_basic_metrics(generated_tokens, reference_tokens, tokenizer=None):
    """Compute basic metrics without relying on metrics_registry."""
    from collections import defaultdict
    
    metrics = {}
    
    # Calculate exact match
    metrics["exact_match"] = int(generated_tokens == reference_tokens)
    
    # Calculate token accuracy
    if len(reference_tokens) > 0:
        min_len = min(len(generated_tokens), len(reference_tokens))
        matching_tokens = sum(generated_tokens[i] == reference_tokens[i] for i in range(min_len))
        metrics["token_accuracy"] = matching_tokens / len(reference_tokens)
    else:
        metrics["token_accuracy"] = 0.0
    
    # Calculate n-gram overlaps
    if len(generated_tokens) > 0 and len(reference_tokens) > 0:
        # Unigram overlap
        gen_unigrams = [t for t in generated_tokens]
        ref_unigrams = [t for t in reference_tokens]
        
        gen_counter = defaultdict(int)
        ref_counter = defaultdict(int)
        
        for t in gen_unigrams:
            gen_counter[t] += 1
        for t in ref_unigrams:
            ref_counter[t] += 1
        
        matches = 0
        for t, count in gen_counter.items():
            matches += min(count, ref_counter[t])
        
        metrics["unigram_overlap"] = matches / len(ref_unigrams) if ref_unigrams else 0.0
        
        # Bigram overlap
        if len(generated_tokens) > 1 and len(reference_tokens) > 1:
            gen_bigrams = [(generated_tokens[i], generated_tokens[i+1]) for i in range(len(generated_tokens)-1)]
            ref_bigrams = [(reference_tokens[i], reference_tokens[i+1]) for i in range(len(reference_tokens)-1)]
            
            gen_counter = defaultdict(int)
            ref_counter = defaultdict(int)
            
            for bg in gen_bigrams:
                gen_counter[bg] += 1
            for bg in ref_bigrams:
                ref_counter[bg] += 1
            
            matches = 0
            for bg, count in gen_counter.items():
                matches += min(count, ref_counter[bg])
            
            metrics["bigram_overlap"] = matches / len(ref_bigrams) if ref_bigrams else 0.0
        else:
            metrics["bigram_overlap"] = 0.0
    else:
        metrics["unigram_overlap"] = 0.0
        metrics["bigram_overlap"] = 0.0
    
    # Calculate LCS ratio
    if len(reference_tokens) > 0:
        # Dynamic programming for LCS
        m, n = len(generated_tokens), len(reference_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if generated_tokens[i-1] == reference_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        metrics["lcs_ratio"] = lcs_length / len(reference_tokens)
    else:
        metrics["lcs_ratio"] = 0.0
    
    return metrics

def evaluate_generated_output(
    generated_tokens, 
    target_tokens, 
    tokenizer, 
    metric_groups=None, 
    skip_metric_groups=None,
    device=None,
    sentencebert_model="all-MiniLM-L6-v2",
    bertscore_model="distilbert-base-uncased"
) -> Dict[str, Any]:
    """
    Evaluate the generated output against the target using registered metrics.
    
    Args:
        generated_tokens: Tokens generated by the optimized prompt
        target_tokens: Target tokens to match
        tokenizer: Tokenizer for decoding
        metric_groups: List of metric groups to compute (None = all)
        skip_metric_groups: List of metric groups to skip
        device: Device to run on (default: None, will use GPU if available)
        sentencebert_model: SentenceBERT model to use
        bertscore_model: Model to use for BERTScore
        
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    # Get decoded texts
    generated_text = tokenizer.decode(generated_tokens)
    target_text = tokenizer.decode(target_tokens)
    
    # Use the centralized metrics computation if available
    if REGISTRY_AVAILABLE:
        # Always skip prompt_reconstruction group - those metrics are computed separately
        # via per-epoch tracking in the optimization loop
        effective_skip_groups = list(skip_metric_groups) if skip_metric_groups else []
        if "prompt_reconstruction" not in effective_skip_groups:
            effective_skip_groups.append("prompt_reconstruction")

        metrics = compute_all_metrics(
            generated_tokens=generated_tokens,
            reference_tokens=target_tokens,
            tokenizer=tokenizer,
            compute_groups=metric_groups,
            skip_groups=effective_skip_groups,
            device=device,
            bertscore_model_type=bertscore_model,
            sentencebert_model_name=sentencebert_model
        )
    else:
        # Fall back to basic metrics
        metrics = _compute_basic_metrics(generated_tokens, target_tokens, tokenizer)
        
        # Add semantic metrics if not skipped
        if not skip_metric_groups or "semantic" not in skip_metric_groups:
            if metric_groups is None or "semantic" in metric_groups:
                # Add SentenceBERT similarity
                sentencebert_metrics = compute_sentencebert_similarity(
                    generated_text=generated_text,
                    reference_text=target_text,
                    model_name=sentencebert_model,
                    device=device
                )
                metrics.update(sentencebert_metrics)
                
                # Add BERTScore
                bertscore_metrics = compute_bertscore(
                    generated_text=generated_text,
                    reference_text=target_text,
                    model_type=bertscore_model,
                    device=device
                )
                metrics.update(bertscore_metrics)
    
    # Always add the decoded texts for reference
    metrics["generated_text"] = generated_text
    metrics["target_text"] = target_text
    
    return _to_serializable(metrics)

def evaluate_prompt_reconstruction(
    optimized_tokens: List[int],
    ground_truth_tokens: List[int],
    tokenizer,
    device=None
) -> Dict[str, Any]:
    """
    Evaluate prompt reconstruction performance (SODA-style evaluation).

    Computes metrics comparing the optimized prompt tokens against
    the ground-truth prompt tokens.

    Args:
        optimized_tokens: Optimized prompt token IDs
        ground_truth_tokens: Ground-truth prompt token IDs
        tokenizer: Tokenizer for decoding texts
        device: Device to run on (unused, for API consistency)

    Returns:
        Dictionary containing prompt-level evaluation metrics:
        - prompt_exact_match: Binary (1 if optimized == ground-truth)
        - prompt_token_accuracy: Per-position token accuracy
        - prompt_lcs_ratio: LCS ratio on prompt tokens
        - per_position_match: List of per-position match indicators
        - optimized_prompt_text: Decoded optimized prompt
        - ground_truth_prompt_text: Decoded ground-truth prompt
    """
    metrics = {}

    # Compute prompt-level metrics using registry if available
    if REGISTRY_AVAILABLE:
        from metrics_registry import registry

        # Compute each prompt metric
        prompt_metrics = ["prompt_exact_match", "prompt_token_accuracy",
                         "prompt_lcs_ratio", "per_position_match"]

        for metric_name in prompt_metrics:
            result = registry.compute_metric(
                metric_name,
                optimized_prompt_tokens=optimized_tokens,
                ground_truth_prompt_tokens=ground_truth_tokens
            )
            if result is not None:
                metrics[metric_name] = result
    else:
        # Fallback implementation
        # Exact match
        metrics["prompt_exact_match"] = int(optimized_tokens == ground_truth_tokens)

        # Token accuracy
        if len(ground_truth_tokens) > 0:
            min_len = min(len(optimized_tokens), len(ground_truth_tokens))
            matching = sum(
                optimized_tokens[i] == ground_truth_tokens[i]
                for i in range(min_len)
            )
            metrics["prompt_token_accuracy"] = matching / len(ground_truth_tokens)
        else:
            metrics["prompt_token_accuracy"] = 0.0

        # LCS ratio
        if len(ground_truth_tokens) > 0:
            m, n = len(optimized_tokens), len(ground_truth_tokens)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if optimized_tokens[i-1] == ground_truth_tokens[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            lcs = dp[m][n]
            metrics["prompt_lcs_ratio"] = lcs / len(ground_truth_tokens)
        else:
            metrics["prompt_lcs_ratio"] = 0.0

        # Per-position match
        min_len = min(len(optimized_tokens), len(ground_truth_tokens))
        max_len = max(len(optimized_tokens), len(ground_truth_tokens))
        per_pos = []
        for i in range(max_len):
            if i < min_len:
                per_pos.append(int(optimized_tokens[i] == ground_truth_tokens[i]))
            else:
                per_pos.append(0)
        metrics["per_position_match"] = per_pos

    # Add decoded texts
    metrics["optimized_prompt_text"] = tokenizer.decode(optimized_tokens, skip_special_tokens=False)
    metrics["ground_truth_prompt_text"] = tokenizer.decode(ground_truth_tokens, skip_special_tokens=False)

    return _to_serializable(metrics)


def evaluate_batch_outputs(
    generated_batch, 
    target_batch, 
    tokenizer, 
    metric_groups=None, 
    skip_metric_groups=None,
    device=None,
    sentencebert_model="all-MiniLM-L6-v2",
    bertscore_model="distilbert-base-uncased"
) -> List[Dict[str, Any]]:
    """
    Evaluate a batch of generated outputs against their targets.
    
    Args:
        generated_batch: List of token lists for generated outputs
        target_batch: List of token lists for target outputs
        tokenizer: Tokenizer for decoding
        metric_groups: List of metric groups to compute (None = all)
        skip_metric_groups: List of metric groups to skip
        device: Device to run on (default: None, will use GPU if available)
        sentencebert_model: SentenceBERT model to use
        bertscore_model: Model to use for BERTScore
        
    Returns:
        List of dictionaries with evaluation metrics for each example
    """
    results = []
    
    for generated_tokens, target_tokens in zip(generated_batch, target_batch):
        metrics = evaluate_generated_output(
            generated_tokens=generated_tokens,
            target_tokens=target_tokens,
            tokenizer=tokenizer,
            metric_groups=metric_groups,
            skip_metric_groups=skip_metric_groups,
            device=device,
            sentencebert_model=sentencebert_model,
            bertscore_model=bertscore_model
        )
        results.append(metrics)
        
    return results

def aggregate_batch_metrics(batch_metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregate metrics from a batch of evaluations.
    
    Args:
        batch_metrics: List of metric dictionaries from evaluate_batch_outputs
        
    Returns:
        Dictionary with averaged metrics using the expected naming conventions
    """
    if not batch_metrics:
        return {}
        
    # Initialize aggregated metrics
    aggregated = {}
    
    # Collect all numeric values
    for metrics in batch_metrics:
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if key not in aggregated:
                    aggregated[key] = []
                aggregated[key].append(value)
    
    # Average the values with expected naming convention
    averaged = {}
    for key, values in aggregated.items():
        if key == "exact_match":
            # For backward compatibility
            averaged["success_rate"] = sum(values) / len(values)
        
        # Use avg_ prefix for compatibility
        averaged[f"avg_{key}"] = sum(values) / len(values)
    
    # Add count
    averaged["sample_count"] = len(batch_metrics)
    
    return averaged
