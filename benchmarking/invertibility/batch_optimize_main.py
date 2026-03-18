"""
Main entry point for batch optimization of prompts for target sentences.
"""
import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
from typing import Dict, List, Any, Optional
import wandb 

from transformers import AutoModelForCausalLM, AutoTokenizer

# Import custom modules
from metrics_registry import compute_all_metrics
from metrics_aggregator import MetricsAggregator
from metrics_logging import MetricsLogger
from evaluation_utils import evaluate_generated_output, evaluate_prompt_reconstruction
from main import optimize_inputs

logger = logging.getLogger("batch_optimize")
logging.basicConfig(level=logging.INFO, 
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def setup_model_and_tokenizer(model_name: str, device: str = 'cpu', model_precision: str = 'full'):
    """
    Load model and tokenizer
    
    Args:
        model_name: HuggingFace model name
        device: Device to run on ('cpu' or 'cuda')
        model_precision: Precision for model ('full' or 'half')
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_name}")
    
    # Determine torch dtype based on precision
    #model = AutoModelForCausalLM.from_pretrained(model_name)

    torch_dtype = torch.float32
    if model_precision == "half":
        torch_dtype = torch.float16
    
    # Load model with memory optimizations
    # device_map="auto" handles loading directly to GPU/offloading
    # low_cpu_mem_usage=True prevents full loading into RAM
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True
    )
    
    model.eval()  # Set to evaluation mode (frozen)

    # Freeze model weights
    for param in model.parameters():
        param.requires_grad = False

    # Note: model.to(device) is not needed/recommended when using device_map="auto"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer

def load_dataset(dataset_path: str) -> Dict[str, Any]:
    """
    Load dataset from a JSON file.

    Auto-detects dataset format (legacy vs SODA) from metadata.

    Args:
        dataset_path: Path to the dataset file

    Returns:
        Dataset dictionary
    """
    import json

    logger.info(f"Loading dataset from {dataset_path}")

    try:
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)

        num_samples = len(dataset.get('samples', []))
        logger.info(f"Loaded {num_samples} samples")

        # Detect dataset format
        metadata = dataset.get("metadata", {})
        evaluation_type = metadata.get("evaluation_type", "output_match")

        if evaluation_type == "prompt_reconstruction":
            logger.info("Detected SODA-style prompt reconstruction dataset")
            logger.info(f"  - Dataset source: {metadata.get('dataset_source', 'unknown')}")
            logger.info(f"  - Prompt length: {metadata.get('prompt_length', 'unknown')}")
            logger.info(f"  - Output length: {metadata.get('output_length', 'unknown')}")
        else:
            logger.info("Detected legacy output-match dataset")

        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def is_prompt_reconstruction_dataset(dataset: Dict[str, Any]) -> bool:
    """
    Check if the dataset is a SODA-style prompt reconstruction dataset.

    Args:
        dataset: The loaded dataset dictionary

    Returns:
        True if this is a prompt reconstruction dataset
    """
    metadata = dataset.get("metadata", {})
    return metadata.get("evaluation_type") == "prompt_reconstruction"

def prepare_targets(dataset: Dict[str, Any], target_indices: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """
    Prepare the list of targets, optionally filtering by indices.
    
    Args:
        dataset: The loaded dataset
        target_indices: Optional list of indices to select specific targets
        
    Returns:
        List of prepared target dictionaries
    """
    targets = dataset.get("samples", [])
    
    # Filter targets if indices are provided
    if target_indices is not None:
        targets = [targets[i] for i in target_indices if i < len(targets)]
        logger.info(f"Selected {len(targets)} targets based on provided indices")
    
    return targets

def optimize_for_target(target_info: Dict[str, Any], model, tokenizer, device: str,
                       config: Dict[str, Any], run_id: str, output_dir: str, metric_groups: List[str] = None,
                       is_prompt_reconstruction: bool = False) -> Dict[str, Any]:
    """
    Optimize a prompt for a specific target sentence.
    
    Args:
        target_info: Dictionary containing target sentence information
        model: The language model
        tokenizer: The tokenizer
        device: The device to run optimization on
        config: Dictionary of optimization parameters
        run_id: Unique identifier for the W&B run
        output_dir: Directory to save results locally
        metric_groups: List of metric groups to compute
        is_prompt_reconstruction: Whether this is a SODA-style prompt reconstruction evaluation

    Returns:
        result: Dictionary containing optimization results
    """

    target_id = target_info["id"]
    target_text = target_info["text"]
    k_target = target_info.get("k_target", 1)  # Default to 1 for SODA datasets

    # Get pre-computed perplexity from dataset if available
    target_perplexity = target_info.get("perplexity", None)

    # For SODA-style datasets, get ground-truth prompt information
    ground_truth_prompt = target_info.get("ground_truth_prompt", None)
    ground_truth_prompt_tokens = target_info.get("ground_truth_prompt_tokens", None)

    if is_prompt_reconstruction:
        logger.info(f"Optimizing prompt for target {target_id} (prompt reconstruction mode)")
        if ground_truth_prompt:
            logger.info(f"  Ground-truth prompt: '{ground_truth_prompt[:50]}...'")
    else:
        logger.info(f"Optimizing prompt for target {target_id}: '{target_text}'")
    
    # Create a run name that includes the target information
    run_name = f"{run_id}_target{target_id}_k{k_target}"
    
    # Initialize W&B for this target
    target_config = config.copy()
    target_config.update({
        "target_id": target_id,
        "target_text": target_text,
        "target_k": k_target,
        "target_avg_rank": target_info.get("avg_rank", 0),
        "target_length": target_info.get("length", len(target_text.split())),
        "target_perplexity": target_perplexity,
        "is_prompt_reconstruction": is_prompt_reconstruction,
    })

    # Add prompt reconstruction specific metadata
    if is_prompt_reconstruction and ground_truth_prompt_tokens:
        target_config["ground_truth_prompt_length"] = len(ground_truth_prompt_tokens)
    
    # Setup metrics logger
    metrics_logger = MetricsLogger(
        run_id=f"{run_id}_{target_id}",
        output_dir=f"{output_dir}/target_{target_id}",
        wandb_project=config.get("wandb_project"),
        wandb_entity=config.get("wandb_entity")
    )
    
    metrics_logger.init_wandb(
        config=target_config,
        name=run_name,
        group=run_id,
        job_type="single_target_optimization"
    )
    
    if config['decouple_learnable_temperature'] \
    and not config['learnable_temperature']:
        raise ValueError("decouple_learnable_temperature requires learnable_temperature")

    if config['bptt_decouple_learnable_temperature'] \
    and not config['bptt_learnable_temperature']:
        raise ValueError("bptt_decouple_learnable_temperature requires bptt_learnable_temperature")

    # Tokenize target for later comparison
    target_tokens = tokenizer(target_text, return_tensors="pt").input_ids[0].cpu().tolist()
    target_output_dir = Path(f"{output_dir}/target_{target_id}")
    target_output_dir.mkdir(parents=True, exist_ok=True)

    # Build diversity callback if requested
    _diversity_cb = None
    if config.get("diversity_n_samples", 0) > 0:
        from diversity_analysis import build_diversity_callback
        _diversity_temps_str = config.get("diversity_temperatures", "")
        _diversity_temps = (
            [float(t.strip()) for t in _diversity_temps_str.split(",") if t.strip()]
            if _diversity_temps_str
            else [float(config.get("temperature", 1.0))]
        )
        _diversity_cb = build_diversity_callback(
            tokenizer=tokenizer,
            temperatures=_diversity_temps,
            n_samples=int(config["diversity_n_samples"]),
            output_dir=str(target_output_dir),
            log_every=int(config.get("diversity_log_every", 10)),
        )

    # Optimize inputs for this target
    opt_result = optimize_inputs(
        model=model,
        tokenizer=tokenizer,
        device=device,
        target_text=target_text,
        losses=config["losses"],
        bptt=config.get("bptt", False),
        seq_len=config["seq_len"],
        epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        lookahead_k=config.get("lookahead_k", 0),
        lookahead_alpha=config.get("lookahead_alpha", 0.5),
        logits_lora_b_learning_rate=config.get("logits_lora_b_learning_rate"),
        temperature=config.get("temperature", 1.0),
        bptt_temperature=config.get("bptt_temperature", 1.0),
        learnable_temperature=config.get("learnable_temperature", False),
        decouple_learnable_temperature=config.get("decouple_learnable_temperature", False),
        bptt_learnable_temperature=config.get("bptt_learnable_temperature", False),
        bptt_decouple_learnable_temperature=config.get("bptt_decouple_learnable_temperature", False),
        stgs_hard=config.get("stgs_hard", True),
        stgs_hard_method=config.get("stgs_hard_method", "categorical"),
        stgs_hard_embsim_probs=config.get("stgs_hard_embsim_probs", "gumbel_soft"),
        stgs_hard_embsim_strategy=config.get("stgs_hard_embsim_strategy", "nearest"),
        stgs_hard_embsim_top_k=config.get("stgs_hard_embsim_top_k", 8),
        stgs_hard_embsim_rerank_alpha=config.get("stgs_hard_embsim_rerank_alpha", 0.5),
        stgs_hard_embsim_sample_tau=config.get("stgs_hard_embsim_sample_tau", 1.0),
        stgs_hard_embsim_margin=config.get("stgs_hard_embsim_margin", 0.0),
        stgs_hard_embsim_fallback=config.get("stgs_hard_embsim_fallback", "argmax"),
        bptt_stgs_hard=config.get("bptt_stgs_hard", True),
        bptt_stgs_hard_method=config.get("bptt_stgs_hard_method", "categorical"),
        bptt_stgs_hard_embsim_probs=config.get("bptt_stgs_hard_embsim_probs", "gumbel_soft"),
        bptt_stgs_hard_embsim_strategy=config.get("bptt_stgs_hard_embsim_strategy", "nearest"),
        bptt_stgs_hard_embsim_top_k=config.get("bptt_stgs_hard_embsim_top_k", 8),
        bptt_stgs_hard_embsim_rerank_alpha=config.get("bptt_stgs_hard_embsim_rerank_alpha", 0.5),
        bptt_stgs_hard_embsim_sample_tau=config.get("bptt_stgs_hard_embsim_sample_tau", 1.0),
        bptt_stgs_hard_embsim_margin=config.get("bptt_stgs_hard_embsim_margin", 0.0),
        bptt_stgs_hard_embsim_fallback=config.get("bptt_stgs_hard_embsim_fallback", "argmax"),
        bptt_hidden_state_conditioning=config.get("bptt_hidden_state_conditioning", False),
        plot_every=config.get("plot_every", 100),
        stgs_grad_variance_samples=config.get("stgs_grad_variance_samples", 0),
        stgs_grad_variance_period=config.get("stgs_grad_variance_period", 1),
        stgs_grad_bias_samples=config.get("stgs_grad_bias_samples", 0),
        stgs_grad_bias_period=config.get("stgs_grad_bias_period", 1),
        stgs_grad_bias_reference_samples=config.get("stgs_grad_bias_reference_samples", 0),
        stgs_grad_bias_reference_batch_size=config.get("stgs_grad_bias_reference_batch_size", 0),
        stgs_grad_bias_reference_use_baseline=config.get("stgs_grad_bias_reference_use_baseline", True),
        stgs_grad_bias_reference_reward_scale=config.get("stgs_grad_bias_reference_reward_scale", 1.0),
        stgs_grad_bias_reference_baseline_beta=config.get("stgs_grad_bias_reference_baseline_beta", 0.9),
        reinforce_grad_variance_samples=config.get("reinforce_grad_variance_samples", 0),
        reinforce_grad_variance_period=config.get("reinforce_grad_variance_period", 1),
        gradient_estimator=config.get("gradient_estimator", "stgs"),
        reinforce_reward_scale=config.get("reinforce_reward_scale", 1.0),
        reinforce_use_baseline=config.get("reinforce_use_baseline", True),
        reinforce_baseline_beta=config.get("reinforce_baseline_beta", 0.9),
        eps=config.get("eps", 1e-10),
        bptt_eps=config.get("bptt_eps", 1e-10),
        vocab_threshold=config.get("vocab_threshold", 0.5),
        filter_vocab=config.get("filter_vocab", True),
        logits_top_k=config.get("logits_top_k", 0),
        logits_top_p=config.get("logits_top_p", 1.0),
        logits_filter_penalty=config.get("logits_filter_penalty", 1e4),
        logits_normalize=config.get("logits_normalize", "none"),
        bptt_logits_normalize=config.get("bptt_logits_normalize", "none"),
        max_gradient_norm=config.get("max_gradient_norm", 0.0),
        batch_size=config.get("batch_size", 1),
        # Method selection (NEW)
        method=config.get("method", "stgs"),
        # Backend selection for SODA/GCG (NEW)
        baseline_backend=config.get("baseline_backend", "hf"),
        baseline_model_name=config.get("baseline_model_name") or config.get("model_name"),
        # SODA parameters (NEW)
        soda_decay_rate=config.get("soda_decay_rate", 0.9),
        soda_betas=(config.get("soda_beta1", 0.9), config.get("soda_beta2", 0.995)),
        soda_reset_epoch=config.get("soda_reset_epoch", 50),
        soda_reinit_epoch=config.get("soda_reinit_epoch", 1500),
        soda_reg_weight=config.get("soda_reg_weight"),
        soda_bias_correction=config.get("soda_bias_correction", False),
        soda_init_strategy=config.get("soda_init_strategy", "zeros"),
        soda_init_std=config.get("soda_init_std", 0.05),
        # GCG parameters (NEW)
        gcg_num_candidates=config.get("gcg_num_candidates", 704),
        gcg_top_k=config.get("gcg_top_k", 128),
        gcg_num_mutations=config.get("gcg_num_mutations", 1),
        gcg_pos_choice=config.get("gcg_pos_choice", "uniform"),
        gcg_token_choice=config.get("gcg_token_choice", "uniform"),
        gcg_init_strategy=config.get("gcg_init_strategy", "zeros"),
        gcg_candidate_batch_size=config.get("gcg_candidate_batch_size", 8),
        # Teacher forcing (for faster training)
        teacher_forcing=config.get("teacher_forcing", False),
        bptt_teacher_forcing_via_diff_model=config.get("bptt_teacher_forcing_via_diff_model", False),
        # Prompt reconstruction (SODA-style)
        ground_truth_prompt_tokens=ground_truth_prompt_tokens if is_prompt_reconstruction else None,
        ground_truth_prompt=ground_truth_prompt if is_prompt_reconstruction else None,
        # Semantic metrics (BERT/SentenceBERT) per-epoch tracking
        semantic_metrics_every_n_epochs=config.get("semantic_metrics_every_n_epochs", 0),
        bertscore_model_type=config.get("bertscore_model", "distilbert-base-uncased"),
        sentencebert_model_name=config.get("sentencebert_model", "all-MiniLM-L6-v2"),
        # Fixed logit distribution parameters
        fixed_gt_prefix_n=config.get("fixed_gt_prefix_n", 0),
        fixed_gt_suffix_n=config.get("fixed_gt_suffix_n", 0),
        fixed_gt_prefix_rank2_n=config.get("fixed_gt_prefix_rank2_n", 0),
        fixed_gt_suffix_rank2_n=config.get("fixed_gt_suffix_rank2_n", 0),
        fixed_prefix_text=config.get("fixed_prefix_text"),
        fixed_suffix_text=config.get("fixed_suffix_text"),
        assemble_strategy=config.get("assemble_strategy", "identity"),
        pos_lm_name=config.get("pos_lm_name"),
        pos_prompt=config.get("pos_prompt"),
        pos_prompt_template=config.get("pos_prompt_template"),
        pos_use_chat_template=config.get("pos_use_chat_template", False),
        pos_lm_temperature=config.get("pos_lm_temperature", 1.0),
        pos_lm_top_p=config.get("pos_lm_top_p", 1.0),
        pos_lm_top_k=config.get("pos_lm_top_k", 0),
        # STGS initialization strategy
        init_strategy=config.get("init_strategy", "randn"),
        init_std=config.get("init_std", 0.0),
        init_mlm_model=config.get("init_mlm_model", "distilbert-base-uncased"),
        init_mlm_top_k=config.get("init_mlm_top_k", 50),
        early_stop_on_exact_match=config.get("early_stop_on_exact_match", False),
        early_stop_loss_threshold=config.get("early_stop_loss_threshold", 0.01),
        early_stop_embsim_lcs_ratio_threshold=config.get("early_stop_embsim_lcs_ratio_threshold", 1.0),
        run_discrete_validation=config.get("run_discrete_validation", False),
        run_discrete_embsim_validation=config.get("run_discrete_embsim_validation", False),
        embsim_similarity=config.get("embsim_similarity", "cossim"),
        embsim_use_input_logits=config.get("embsim_use_input_logits", True),
        embsim_teacher_forcing=config.get("embsim_teacher_forcing", False),
        embsim_temperature=config.get("embsim_temperature", 1.0),
        temperature_anneal_schedule=config.get("temperature_anneal_schedule", "none"),
        temperature_anneal_min=config.get("temperature_anneal_min", 0.1),
        temperature_anneal_epochs=config.get("temperature_anneal_epochs", 0),
        temperature_anneal_reg_lambda=config.get("temperatureAnnealRegLambda", 0.0),
        temperature_anneal_reg_mode=config.get("temperatureAnnealRegMode", "mse"),
        temperature_loss_coupling_lambda=config.get("temperatureLossCouplingLambda", 0.0),
        discrete_reinit_epoch=config.get("discrete_reinit_epoch", 0),
        discrete_reinit_snap=config.get("discrete_reinit_snap", "argmax"),
        discrete_reinit_prob=config.get("discrete_reinit_prob", 1.0),
        discrete_reinit_topk=config.get("discrete_reinit_topk", 0),
        discrete_reinit_entropy_threshold=config.get("discrete_reinit_entropy_threshold", 0.0),
        discrete_reinit_embsim_probs=config.get("discrete_reinit_embsim_probs", "input_logits"),
        gumbel_noise_scale=config.get("gumbel_noise_scale", 1.0),
        adaptive_gumbel_noise=config.get("adaptive_gumbel_noise", False),
        adaptive_gumbel_noise_beta=config.get("adaptive_gumbel_noise_beta", 0.9),
        adaptive_gumbel_noise_min_scale=config.get("adaptive_gumbel_noise_min_scale", 0.0),
        stgs_input_dropout=config.get("stgs_input_dropout", 0.0),
        stgs_output_dropout=config.get("stgs_output_dropout", 0.0),
        logits_lora_rank=config.get("logits_lora_rank", 0),
        # Learnable prompt length
        prompt_length_learnable=config.get("prompt_length_learnable", False),
        prompt_length_alpha_init=config.get("prompt_length_alpha_init", 0.0),
        prompt_length_beta=config.get("prompt_length_beta", 5.0),
        prompt_length_reg_lambda=config.get("prompt_length_reg_lambda", 0.0),
        prompt_length_eos_spike=config.get("prompt_length_eos_spike", 10.0),
        prompt_length_mask_eos_attention=config.get("prompt_length_mask_eos_attention", False),
        logit_decay=config.get("logit_decay", 0.0),
        ppo_kl_lambda=config.get("ppo_kl_lambda", 0.0),
        ppo_kl_mode=config.get("ppo_kl_mode", "soft"),
        ppo_kl_epsilon=config.get("ppo_kl_epsilon", 0.0),
        ppo_ref_update_period=config.get("ppo_ref_update_period", 10),
        diversity_callback=_diversity_cb,
        superposition_metric_every=config.get("superposition_metric_every", 0),
        superposition_metric_modes=config.get("superposition_metric_modes", "dot,cos,l2"),
        superposition_vocab_top_k=config.get("superposition_vocab_top_k", 256),
        superposition_vocab_source=config.get("superposition_vocab_source", "wikipedia"),
        superposition_vocab_dataset_path=(
            config.get("superposition_vocab_dataset_path")
            or config.get("dataset_path")
        ),
        superposition_vocab_hf_name=config.get("superposition_vocab_hf_name", "lucadiliello/english_wikipedia"),
        superposition_vocab_hf_split=config.get("superposition_vocab_hf_split", "train"),
        superposition_vocab_num_texts=config.get("superposition_vocab_num_texts", 1000),
        superposition_entropy_temperature=config.get("superposition_entropy_temperature", 1.0),
        superposition_output_dir=str(target_output_dir / "superposition"),
        kwargs=config,
    )

    # Extract fields from result dict
    generated_tokens          = opt_result["generated_tokens"]
    optimized_inputs          = opt_result["optimized_inputs"]
    learnable_logits          = opt_result.get("learnable_logits", optimized_inputs)
    learnable_temperatures    = opt_result.get("learnable_temperatures", {})
    losses                    = opt_result["losses"]
    lcs_ratio_history         = opt_result["lcs_ratio_history"]
    prompt_metrics_history    = opt_result["prompt_metrics_history"]
    semantic_metrics_history  = opt_result["semantic_metrics_history"]
    discrete_generated_tokens         = opt_result.get("discrete_generated_tokens", [])
    discrete_lcs_ratio_history        = opt_result.get("discrete_lcs_ratio_history", [])
    discrete_prompt_metrics_history   = opt_result.get("discrete_prompt_metrics_history", {})
    discrete_semantic_metrics_history = opt_result.get("discrete_semantic_metrics_history", {})
    all_lcs_ratio_history             = opt_result.get("all_lcs_ratio_history", [])
    all_prompt_metrics_history        = opt_result.get("all_prompt_metrics_history", {})
    all_semantic_metrics_history      = opt_result.get("all_semantic_metrics_history", {})
    embsim_generated_tokens         = opt_result.get("embsim_generated_tokens", [])
    embsim_lcs_ratio_history        = opt_result.get("embsim_lcs_ratio_history", [])
    embsim_prompt_metrics_history   = opt_result.get("embsim_prompt_metrics_history", {})
    embsim_semantic_metrics_history = opt_result.get("embsim_semantic_metrics_history", {})

    # Extract the optimized prompt tokens
    optimized_tokens = torch.argmax(optimized_inputs[0], dim=-1).cpu().tolist()
    optimized_text = tokenizer.decode(optimized_tokens)

    # Evaluate the generated output using our centralized metrics system
    evaluation_metrics = evaluate_generated_output(
        generated_tokens=generated_tokens,
        target_tokens=target_tokens,
        tokenizer=tokenizer,
        metric_groups=metric_groups,
        device=device
    )

    # Discrete + all_ final evaluation
    if discrete_generated_tokens:
        disc_eval = evaluate_generated_output(
            generated_tokens=discrete_generated_tokens,
            target_tokens=target_tokens,
            tokenizer=tokenizer,
            metric_groups=metric_groups,
            device=device)
        for k, v in disc_eval.items():
            evaluation_metrics[f"discrete_{k}"] = v
        for k in disc_eval:
            s_val = evaluation_metrics.get(k)
            d_val = disc_eval[k]
            if s_val is not None and d_val is not None:
                if "exact_match" in k:
                    evaluation_metrics[f"all_{k}"] = int(bool(s_val) and bool(d_val))
                elif isinstance(s_val, (int, float)):
                    evaluation_metrics[f"all_{k}"] = min(s_val, d_val)

    # Embsim final evaluation
    if embsim_generated_tokens:
        embsim_eval = evaluate_generated_output(
            generated_tokens=embsim_generated_tokens,
            target_tokens=target_tokens,
            tokenizer=tokenizer,
            metric_groups=metric_groups,
            device=device)
        for k, v in embsim_eval.items():
            evaluation_metrics[f"embsim_{k}"] = v

    # For SODA-style prompt reconstruction, also compute prompt-level metrics
    prompt_metrics = {}
    if is_prompt_reconstruction and ground_truth_prompt_tokens:
        prompt_metrics = evaluate_prompt_reconstruction(
            optimized_tokens=optimized_tokens,
            ground_truth_tokens=ground_truth_prompt_tokens,
            tokenizer=tokenizer,
            device=device
        )
        # Merge prompt metrics into evaluation metrics
        evaluation_metrics.update(prompt_metrics)

        logger.info(f"Prompt reconstruction metrics for {target_id}:")
        logger.info(f"  Prompt exact match: {prompt_metrics.get('prompt_exact_match', 0)}")
        logger.info(f"  Prompt token accuracy: {prompt_metrics.get('prompt_token_accuracy', 0):.4f}")
        logger.info(f"  Prompt LCS ratio: {prompt_metrics.get('prompt_lcs_ratio', 0):.4f}")

    # Log evaluation metrics to W&B
    metrics_logger.log_metrics(evaluation_metrics)

    # Create result dictionary
    result = {
        "target_id": target_id,
        "target_text": target_text,
        "target_k": k_target,
        "target_perplexity": target_perplexity,
        "optimized_tokens": optimized_tokens,
        "optimized_text": optimized_text,
        "generated_tokens": generated_tokens,
        "generated_text": evaluation_metrics["generated_text"],
        "evaluation": evaluation_metrics,
        "final_loss": float(losses[-1]),
        "loss_history": [float(loss) for loss in losses],
        "lcs_ratio_history": lcs_ratio_history,
    }

    # Add prompt reconstruction specific fields
    if is_prompt_reconstruction:
        result["is_prompt_reconstruction"] = True
        if ground_truth_prompt_tokens:
            result["ground_truth_prompt_tokens"] = ground_truth_prompt_tokens
            result["ground_truth_prompt"] = ground_truth_prompt
        result["prompt_metrics"] = prompt_metrics
        result["prompt_metrics_history"] = prompt_metrics_history

    # Add discrete and all_ histories if populated
    if discrete_lcs_ratio_history:
        result["discrete_lcs_ratio_history"] = discrete_lcs_ratio_history
    if discrete_prompt_metrics_history:
        result["discrete_prompt_metrics_history"] = discrete_prompt_metrics_history
    if discrete_semantic_metrics_history:
        result["discrete_semantic_metrics_history"] = discrete_semantic_metrics_history
    if all_lcs_ratio_history:
        result["all_lcs_ratio_history"] = all_lcs_ratio_history
    if all_prompt_metrics_history:
        result["all_prompt_metrics_history"] = all_prompt_metrics_history
    if all_semantic_metrics_history:
        result["all_semantic_metrics_history"] = all_semantic_metrics_history
    if embsim_lcs_ratio_history:
        result["embsim_lcs_ratio_history"] = embsim_lcs_ratio_history
    if embsim_prompt_metrics_history:
        result["embsim_prompt_metrics_history"] = embsim_prompt_metrics_history
    if embsim_semantic_metrics_history:
        result["embsim_semantic_metrics_history"] = embsim_semantic_metrics_history

    # Add semantic metrics history if tracked
    if semantic_metrics_history and any(len(v) > 0 for v in semantic_metrics_history.values()):
        result["semantic_metrics_history"] = semantic_metrics_history

    # Save results to file
    metrics_logger.save_to_file(result, "result.json")
    
    # Save the tensor for future use
    optimized_inputs_path = target_output_dir / "optimized_inputs.pt"
    learnable_logits_path = target_output_dir / "learnable_logits.pt"
    learnable_temperatures_path = target_output_dir / "learnable_temperatures.pt"
    torch.save(optimized_inputs, optimized_inputs_path)
    torch.save(learnable_logits, learnable_logits_path)
    torch.save(learnable_temperatures, learnable_temperatures_path)

    if metrics_logger.wandb_run:
        logits_artifact = wandb.Artifact(
            name=f"learnable-logits-{run_id}-{target_id}",
            type="tensor",
            description="Final learnable logits at the end of sample optimization",
        )
        logits_artifact.add_file(str(learnable_logits_path))
        metrics_logger.wandb_run.log_artifact(logits_artifact)

        temps_artifact = wandb.Artifact(
            name=f"learnable-temperatures-{run_id}-{target_id}",
            type="tensor",
            description="Final learnable temperatures at the end of sample optimization",
        )
        temps_artifact.add_file(str(learnable_temperatures_path))
        metrics_logger.wandb_run.log_artifact(temps_artifact)
    
    # Finish logging
    metrics_logger.finish()
    
    logger.info(f"Optimization completed for target {target_id}")
    logger.info(f"Optimized prompt: '{optimized_text}'")
    logger.info(f"Final loss: {result['final_loss']}")
    logger.info(f"Exact match: {evaluation_metrics.get('exact_match', 0)}")
    logger.info(f"Token accuracy: {evaluation_metrics.get('token_accuracy', 0):.4f}")
    
    return result

def process_targets_sequential(targets: List[Dict[str, Any]], model, tokenizer, device: str,
                              config: Dict[str, Any], run_id: str, output_dir: str,
                              metrics_aggregator: MetricsAggregator,
                              metric_groups: List[str] = None,
                              is_prompt_reconstruction: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Process targets sequentially.

    Args:
        targets: List of target info dictionaries
        model: Language model
        tokenizer: Tokenizer
        device: Device to run on
        config: Configuration dictionary
        run_id: Unique run identifier
        output_dir: Directory to save results
        metrics_aggregator: Metrics aggregator
        metric_groups: List of metric groups to compute
        is_prompt_reconstruction: Whether this is a SODA-style prompt reconstruction evaluation

    Returns:
        results: Dictionary mapping target IDs to optimization results
    """
    results = {}

    for target in tqdm(targets, desc="Optimizing for targets"):
        result = optimize_for_target(
            target_info=target,
            model=model,
            tokenizer=tokenizer,
            device=device,
            config=config,
            run_id=run_id,
            output_dir=output_dir,
            metric_groups=metric_groups,
            is_prompt_reconstruction=is_prompt_reconstruction
        )
        results[target["id"]] = result

        # Add metrics to aggregator
        k_value = target.get("k_target", 1)
        metrics_aggregator.add_sample(result["evaluation"], k_value=k_value)
        # Add per-epoch lcs_ratio history, prompt metrics history, and semantic metrics history to aggregator
        metrics_aggregator.add_epoch_metrics(
            lcs_ratio_history=result["lcs_ratio_history"],
            k_value=k_value,
            prompt_metrics_history=result.get("prompt_metrics_history"),
            semantic_metrics_history=result.get("semantic_metrics_history"),
            discrete_lcs_ratio_history=result.get("discrete_lcs_ratio_history"),
            discrete_prompt_metrics_history=result.get("discrete_prompt_metrics_history"),
            discrete_semantic_metrics_history=result.get("discrete_semantic_metrics_history"),
            all_lcs_ratio_history=result.get("all_lcs_ratio_history"),
            all_prompt_metrics_history=result.get("all_prompt_metrics_history"),
            all_semantic_metrics_history=result.get("all_semantic_metrics_history"),
            embsim_lcs_ratio_history=result.get("embsim_lcs_ratio_history"),
            embsim_prompt_metrics_history=result.get("embsim_prompt_metrics_history"),
            embsim_semantic_metrics_history=result.get("embsim_semantic_metrics_history"),
        )

    return results

def process_targets_parallel(targets: List[Dict[str, Any]], model, tokenizer, config: Dict[str, Any],
                           run_id: str, output_dir: str, metrics_aggregator: MetricsAggregator,
                           num_workers: int, metric_groups: List[str] = None,
                           is_prompt_reconstruction: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Process targets in parallel using a ProcessPoolExecutor.

    Args:
        targets: List of target info dictionaries
        model: Language model
        tokenizer: Tokenizer
        config: Configuration dictionary
        run_id: Unique run identifier
        output_dir: Directory to save results
        metrics_aggregator: Metrics aggregator
        num_workers: Number of parallel workers
        metric_groups: List of metric groups to compute
        is_prompt_reconstruction: Whether this is a SODA-style prompt reconstruction evaluation

    Returns:
        results: Dictionary mapping target IDs to optimization results
    """
    results = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_target = {
            executor.submit(
                optimize_for_target,
                target_info=target,
                model=model,
                tokenizer=tokenizer,
                device="cuda" if torch.cuda.is_available() else "cpu",
                config=config,
                run_id=run_id,
                output_dir=output_dir,
                metric_groups=metric_groups,
                is_prompt_reconstruction=is_prompt_reconstruction
            ): target for target in targets
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_target),
                         total=len(targets),
                         desc="Optimizing for targets"):
            target = future_to_target[future]
            try:
                result = future.result()
                results[target["id"]] = result

                # Add metrics to aggregator
                k_value = target.get("k_target", 1)
                metrics_aggregator.add_sample(result["evaluation"], k_value=k_value)
                # Add per-epoch lcs_ratio history, prompt metrics history, and semantic metrics history to aggregator
                metrics_aggregator.add_epoch_metrics(
                    lcs_ratio_history=result["lcs_ratio_history"],
                    k_value=k_value,
                    prompt_metrics_history=result.get("prompt_metrics_history"),
                    semantic_metrics_history=result.get("semantic_metrics_history"),
                    discrete_lcs_ratio_history=result.get("discrete_lcs_ratio_history"),
                    discrete_prompt_metrics_history=result.get("discrete_prompt_metrics_history"),
                    discrete_semantic_metrics_history=result.get("discrete_semantic_metrics_history"),
                    all_lcs_ratio_history=result.get("all_lcs_ratio_history"),
                    all_prompt_metrics_history=result.get("all_prompt_metrics_history"),
                    all_semantic_metrics_history=result.get("all_semantic_metrics_history"),
                    embsim_lcs_ratio_history=result.get("embsim_lcs_ratio_history"),
                    embsim_prompt_metrics_history=result.get("embsim_prompt_metrics_history"),
                    embsim_semantic_metrics_history=result.get("embsim_semantic_metrics_history"),
                )

            except Exception as exc:
                logger.error(f"Target {target['id']} generated an exception: {exc}")

    return results

def batch_optimize(dataset_path: str, model_name: str, output_dir: str, 
                  config: Dict[str, Any], num_workers: int = 1, 
                  target_indices: Optional[List[int]] = None,
                  metric_groups: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Optimize prompts for multiple target sentences.
    
    Args:
        dataset_path: Path to the dataset file
        model_name: Name of the language model to use
        output_dir: Directory to save results
        config: Dictionary of optimization parameters
        num_workers: Number of parallel workers (if 1, runs sequentially)
        target_indices: Optional list of indices to select specific targets
        metric_groups: List of metric groups to compute
        
    Returns:
        summary: Dictionary with optimization summary
    """
    # Create run ID for grouping
    run_id = wandb.util.generate_id()
    logger.info(f"Starting batch optimization with run ID: {run_id}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set random seeds
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load dataset and prepare targets
    dataset = load_dataset(dataset_path)
    targets = prepare_targets(dataset, target_indices)

    # Detect if this is a SODA-style prompt reconstruction dataset
    prompt_reconstruction_mode = is_prompt_reconstruction_dataset(dataset)
    if prompt_reconstruction_mode:
        logger.info("Running in SODA-style prompt reconstruction evaluation mode")

    # Initialize model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        model_name,
        device,
        model_precision=config.get("model_precision", "full")
    )

    # Create output directory
    full_output_dir = f"{output_dir}/{run_id}"
    output_path = Path(full_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize the metrics aggregator
    metrics_aggregator = MetricsAggregator()

    # Process targets based on number of workers
    if num_workers == 1:
        results = process_targets_sequential(
            targets=targets,
            model=model,
            tokenizer=tokenizer,
            device=device,
            config=config,
            run_id=run_id,
            output_dir=full_output_dir,
            metrics_aggregator=metrics_aggregator,
            metric_groups=metric_groups,
            is_prompt_reconstruction=prompt_reconstruction_mode
        )
    else:
        results = process_targets_parallel(
            targets=targets,
            model=model,
            tokenizer=tokenizer,
            config=config,
            run_id=run_id,
            output_dir=full_output_dir,
            metrics_aggregator=metrics_aggregator,
            num_workers=num_workers,
            metric_groups=metric_groups,
            is_prompt_reconstruction=prompt_reconstruction_mode
        )

    # Initialize the metrics logger
    metrics_logger = MetricsLogger(
        run_id=run_id,
        output_dir=full_output_dir,
        wandb_project=config.get("wandb_project"),
        wandb_entity=config.get("wandb_entity")
    )

    # Initialize W&B for batch coordination
    metrics_logger.init_wandb(
        group=run_id,
        config={
            **config,
            "dataset_path": dataset_path,
            "model_name": model_name,
            "num_targets": len(targets),
            "metadata": dataset.get("metadata", {}),
            "is_prompt_reconstruction": prompt_reconstruction_mode
        },
        name=f"batch_optimization_{run_id}",
        job_type="batch_coordination"
    )
    
    # Populate summary table with individual target results
    for target in targets:
        target_id = target["id"]
        if target_id in results:
            result = results[target_id]
            metrics_logger.update_summary_table(
                sample_id=str(target_id),
                target_info=target,
                result=result,
                metrics=result["evaluation"],
                ground_truth_prompt=target.get("ground_truth_prompt")
            )

    # Get the complete summary
    summary = metrics_aggregator.get_summary()

    # Log the summary
    metrics_logger.log_summary(summary)

    # Log per-epoch avg_lcs_ratio by k-value to W&B
    avg_lcs_by_epoch_k = summary.get("avg_lcs_ratio_by_epoch_by_k", {})
    for epoch, k_dict in avg_lcs_by_epoch_k.items():
        for k_value, avg_lcs in k_dict.items():
            log_entry = {
                "epoch": epoch,
                "k": k_value,
                "avg_lcs_ratio": avg_lcs,
            }
            metrics_logger.log_metrics(log_entry)

    # Log per-epoch prompt metrics by k-value to W&B (for SODA-style evaluation)
    if summary.get("has_prompt_metrics_history"):
        avg_prompt_metrics = summary.get("avg_prompt_metrics_by_epoch_by_k", {})
        for metric_name, epoch_dict in avg_prompt_metrics.items():
            for epoch, k_dict in epoch_dict.items():
                for k_value, avg_value in k_dict.items():
                    log_entry = {
                        "epoch": epoch,
                        "k": k_value,
                        f"avg_{metric_name}": avg_value,
                    }
                    metrics_logger.log_metrics(log_entry)

    # Log per-epoch semantic metrics by k-value to W&B (BERT/SentenceBERT)
    if summary.get("has_semantic_metrics_history"):
        avg_semantic_metrics = summary.get("avg_semantic_metrics_by_epoch_by_k", {})
        for metric_name, epoch_dict in avg_semantic_metrics.items():
            for epoch, k_dict in epoch_dict.items():
                for k_value, avg_value in k_dict.items():
                    log_entry = {
                        "epoch": epoch,
                        "k": k_value,
                        f"avg_{metric_name}": avg_value,
                    }
                    metrics_logger.log_metrics(log_entry)

    # Add results to summary
    summary["results"] = results
    
    # Save complete results
    metrics_logger.save_to_file(
        data={
            "run_id": run_id,
            "summary": summary,
            "config": config,
            "dataset_metadata": dataset.get("metadata", {})
        },
        filename="batch_results.json"
    )
    
    # Finish logging
    metrics_logger.finish()
    
    return summary

def str2bool(v):
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2list(v):
    """Convert comma-separated string to list."""
    if v is None:
        return None
    return [item.strip() for item in v.split(',')]

def build_parser(description: str = "Batch optimize prompts for multiple target sentences"):
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=description)
    
    # Dataset parameters
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the dataset file containing target sentences")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-135M",
                        help="Name of the language model to use")
    parser.add_argument("--model_precision", type=str, default="full",
                        help="Precision of the language model to use", choices=["full", "half"])
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=False,
                        help="Whether to use gradient checkpointing")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="batch_optimization_results",
                        help="Directory to save optimization results")
    
    # Optimization parameters
    parser.add_argument("--losses", type=str, default="crossentropy",
                        help="Loss function(s) to use for optimization")
    parser.add_argument("--seq_len", type=int, default=40,
                        help="Length of the prompt sequence to optimize")
    parser.add_argument("--epochs", type=int, default=2000,
                        help="Number of optimization epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-2,
                        help="Learning rate for optimization")
    parser.add_argument("--lookahead_k", type=str, default="0",
                        help="Lookahead sync period(s) in optimizer steps. "
                             "Use 0 to disable, a single integer for one level, "
                             "or a comma-separated list such as '8,80,800' for nested levels.")
    parser.add_argument("--lookahead_alpha", type=float, default=0.5,
                        help="Lookahead interpolation factor between slow and fast weights.")
    parser.add_argument("--logits_lora_b_learning_rate", type=float, default=None,
                        help="Optional learning rate override for LoRA prompt-logit matrix B. "
                             "Ignored when logits_lora_rank <= 0.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for optimization")
    
    # ST-GS parameters
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for Gumbel-Softmax")
    parser.add_argument("--learnable_temperature", type=str2bool, default=False,
                        help="Whether to learn the temperature parameter")
    parser.add_argument("--decouple_learnable_temperature", type=str2bool, default=False,
                        help="Whether to learn multiple decouple temperature parameters, one for each learnable input.")
    parser.add_argument("--stgs_hard", type=str2bool, default=True,
                        help="Whether to use hard ST-GS")
    parser.add_argument("--stgs_hard_method", type=str, default="categorical",
                        choices=["categorical", "embsim-dot", "embsim-cos", "embsim-l2", "argmax"],
                        help="Hard token selection method for ST-GS: 'categorical' (Gumbel sample), "
                             "'embsim-dot' (nearest token by dot-product), "
                             "'embsim-cos' (nearest token by cosine similarity), "
                             "'embsim-l2' (nearest token by L2 distance)")
    parser.add_argument("--stgs_hard_embsim_probs", type=str, default="gumbel_soft",
                        choices=["gumbel_soft", "input_logits"],
                        help="Probability source for soft embedding in embsim hard-token methods: "
                             "'gumbel_soft' (use y_soft from Gumbel-Softmax), "
                             "'input_logits' (use softmax(x) of raw input logits)")
    parser.add_argument("--stgs_hard_embsim_strategy", type=str, default="nearest",
                        choices=["nearest", "topk_rerank", "topk_sample", "margin_fallback", "lm_topk_restrict"],
                        help="Embsim post-selection strategy: nearest neighbor, top-k rerank, top-k sample, "
                             "margin fallback, or LM-top-k-restricted nearest neighbor")
    parser.add_argument("--stgs_hard_embsim_top_k", type=int, default=8,
                        help="Top-k candidate size for topk_rerank/topk_sample/lm_topk_restrict embsim strategies")
    parser.add_argument("--stgs_hard_embsim_rerank_alpha", type=float, default=0.5,
                        help="Blend weight for topk_rerank: alpha*z(embsim) + (1-alpha)*z(lm_prob)")
    parser.add_argument("--stgs_hard_embsim_sample_tau", type=float, default=1.0,
                        help="Temperature over embsim scores for topk_sample")
    parser.add_argument("--stgs_hard_embsim_margin", type=float, default=0.0,
                        help="Nearest-vs-second-nearest margin threshold for margin_fallback")
    parser.add_argument("--stgs_hard_embsim_fallback", type=str, default="argmax",
                        choices=["argmax", "categorical"],
                        help="Fallback distribution choice used by margin_fallback")
    parser.add_argument("--eps", type=float, default=1e-10,
                        help="Epsilon value for numerical stability")
    
    # Gradient estimator parameters
    parser.add_argument("--gradient_estimator", type=str, default="stgs", choices=["stgs", "reinforce"],
                        help="Gradient estimator to use for optimizing the discrete inputs")
    parser.add_argument("--reinforce_reward_scale", type=float, default=1.0,
                        help="Scaling factor applied to the REINFORCE policy loss")
    parser.add_argument("--reinforce_use_baseline", type=str2bool, default=True,
                        help="Enable running baseline to reduce REINFORCE variance")
    parser.add_argument("--reinforce_baseline_beta", type=float, default=0.9,
                        help="Exponential moving average coefficient for the REINFORCE baseline")

    # Method selection (NEW: for SODA/GCG/O2P support)
    parser.add_argument("--method", type=str, default="stgs",
                        choices=["stgs", "reinforce", "soda", "gcg", "o2p"],
                        help="Optimization method to use")

    # Backend selection for SODA/GCG
    parser.add_argument("--baseline_backend", type=str, default="hf",
                        choices=["hf", "tl"],
                        help="Model backend for SODA/GCG: 'hf' (HuggingFace) or 'tl' (transformer_lens)")
    parser.add_argument("--baseline_model_name", type=str, default=None,
                        help="Model name for transformer_lens backend (defaults to --model_name)")

    # SODA-specific parameters
    parser.add_argument("--soda_decay_rate", type=float, default=0.9,
                        help="SODA: embedding decay rate per epoch")
    parser.add_argument("--soda_beta1", type=float, default=0.9,
                        help="SODA: Adam beta1 parameter")
    parser.add_argument("--soda_beta2", type=float, default=0.995,
                        help="SODA: Adam beta2 parameter")
    parser.add_argument("--soda_reset_epoch", type=int, default=50,
                        help="SODA: optimizer state reset frequency")
    parser.add_argument("--soda_reinit_epoch", type=int, default=1500,
                        help="SODA: embedding reinitialization frequency")
    parser.add_argument("--soda_reg_weight", type=float, default=None,
                        help="SODA: fluency regularization weight (None = disabled)")
    parser.add_argument("--soda_bias_correction", type=str2bool, default=False,
                        help="SODA: use standard Adam bias correction")
    parser.add_argument("--soda_init_strategy", type=str, default="zeros",
                        choices=["zeros", "normal"],
                        help="SODA: embedding initialization strategy")
    parser.add_argument("--soda_init_std", type=float, default=0.05,
                        help="SODA: standard deviation for normal initialization")

    # GCG-specific parameters
    parser.add_argument("--gcg_num_candidates", type=int, default=704,
                        help="GCG: number of mutation candidates per epoch")
    parser.add_argument("--gcg_top_k", type=int, default=128,
                        help="GCG: vocabulary size limit for candidates")
    parser.add_argument("--gcg_num_mutations", type=int, default=1,
                        help="GCG: number of token positions to mutate")
    parser.add_argument("--gcg_pos_choice", type=str, default="uniform",
                        choices=["uniform", "weighted", "greedy"],
                        help="GCG: position selection strategy")
    parser.add_argument("--gcg_token_choice", type=str, default="uniform",
                        choices=["uniform", "weighted"],
                        help="GCG: token selection strategy")
    parser.add_argument("--gcg_init_strategy", type=str, default="zeros",
                        choices=["zeros", "random"],
                        help="GCG: token initialization strategy")
    parser.add_argument("--gcg_candidate_batch_size", type=int, default=8,
                        help="GCG: number of candidate mutations to score per forward pass")

    # O2P-specific parameters
    parser.add_argument("--o2p_model_path", type=str, default=None,
                        help="O2P: path to trained O2P inverse model (required for method='o2p')")
    parser.add_argument("--o2p_num_beams", type=int, default=4,
                        help="O2P: beam size for T5 generation")
    parser.add_argument("--o2p_max_length", type=int, default=32,
                        help="O2P: maximum generation length for T5 decoder")

    # Teacher forcing (for faster training)
    parser.add_argument("--teacher_forcing", type=str2bool, default=False,
                        help="Use teacher forcing for faster training (single forward pass instead of autoregressive). "
                             "Changes loss semantics - predicts next token given correct prefix rather than own generations.")
    parser.add_argument("--bptt_teacher_forcing_via_diff_model", type=str2bool, default=False,
                        help="When bptt=True and teacher_forcing=True in STGS mode, use STGSDiffModel "
                             "teacher-forced hard straight-through outputs instead of a separate BPTT STGS module.")

    # BPTT parameters
    parser.add_argument("--bptt", type=str2bool, default=False,
                        help="Whether to use backpropagation through time")
    parser.add_argument("--bptt_temperature", type=float, default=1.0,
                        help="Temperature for BPTT Gumbel-Softmax")
    parser.add_argument("--bptt_learnable_temperature", type=str2bool, default=False,
                        help="Whether to learn the BPTT temperature parameter")
    parser.add_argument("--bptt_decouple_learnable_temperature", type=str2bool, default=False,
                        help="Whether to learn multiple decouple temperature parameters, one for each learnable input.")
    parser.add_argument("--bptt_stgs_hard", type=str2bool, default=False,
                        help="Whether to use hard ST-GS for BPTT")
    parser.add_argument("--bptt_stgs_hard_method", type=str, default="categorical",
                        choices=["categorical", "embsim-dot", "embsim-cos", "embsim-l2", "argmax"],
                        help="Hard token selection method for BPTT ST-GS")
    parser.add_argument("--bptt_stgs_hard_embsim_probs", type=str, default="gumbel_soft",
                        choices=["gumbel_soft", "input_logits"],
                        help="Probability source for soft embedding in BPTT embsim hard-token methods")
    parser.add_argument("--bptt_stgs_hard_embsim_strategy", type=str, default="nearest",
                        choices=["nearest", "topk_rerank", "topk_sample", "margin_fallback", "lm_topk_restrict"],
                        help="Embsim post-selection strategy for the BPTT STGS module")
    parser.add_argument("--bptt_stgs_hard_embsim_top_k", type=int, default=8,
                        help="Top-k candidate size for BPTT embsim top-k strategies")
    parser.add_argument("--bptt_stgs_hard_embsim_rerank_alpha", type=float, default=0.5,
                        help="Blend weight for BPTT topk_rerank")
    parser.add_argument("--bptt_stgs_hard_embsim_sample_tau", type=float, default=1.0,
                        help="Temperature over embsim scores for BPTT topk_sample")
    parser.add_argument("--bptt_stgs_hard_embsim_margin", type=float, default=0.0,
                        help="Margin threshold for BPTT margin_fallback")
    parser.add_argument("--bptt_stgs_hard_embsim_fallback", type=str, default="argmax",
                        choices=["argmax", "categorical"],
                        help="Fallback choice used by BPTT margin_fallback")
    parser.add_argument("--bptt_hidden_state_conditioning", type=str2bool, default=False,
                        help="Whether to condition BPTT on hidden states")
    parser.add_argument("--bptt_eps", type=float, default=1e-10,
                        help="Epsilon value for BPTT numerical stability")
    
    # Vocabulary parameters
    parser.add_argument("--filter_vocab", type=str2bool, default=False,
                        help="Whether to filter the vocabulary")
    parser.add_argument("--vocab_threshold", type=float, default=0.5,
                        help="Threshold for vocabulary filtering")
    parser.add_argument("--logits_top_k", type=int, default=0,
                        help="Top-k soft filtering of learnable logits before STGS (0 = disabled)")
    parser.add_argument("--logits_top_p", type=float, default=1.0,
                        help="Top-p nucleus soft filtering of learnable logits before STGS (1.0 = disabled)")
    parser.add_argument("--logits_filter_penalty", type=float, default=1e4,
                        help="Penalty magnitude subtracted from suppressed logits (default: 1e4)")
    parser.add_argument("--logits_normalize", type=str, default="none",
                        choices=["none", "center", "zscore"],
                        help="Per-position logit normalization before Gumbel noise: "
                             "'none' (disabled), 'center' (subtract per-position mean), "
                             "'zscore' (center + divide by std). "
                             "Vocab dim only; no mixing across token positions.")
    parser.add_argument("--bptt_logits_normalize", type=str, default="none",
                        choices=["none", "center", "zscore"],
                        help="Same as --logits_normalize but for the BPTT STGS module.")

    # Other parameters
    parser.add_argument("--max_gradient_norm", type=float, default=0.0,
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--plot_every", type=int, default=100,
                        help="Frequency of plotting loss curves")
    parser.add_argument("--stgs_grad_variance_samples", type=int, default=0,
                        help="Number of STGS gradient samples to use when estimating variance (>=2 enables metric)")
    parser.add_argument("--stgs_grad_variance_period", type=int, default=1,
                        help="Epoch interval between two gradient variance measurements")
    parser.add_argument("--stgs_grad_bias_samples", type=int, default=0,
                        help="Number of STGS gradient samples to average when estimating bias (>=1 enables metric)")
    parser.add_argument("--stgs_grad_bias_period", type=int, default=1,
                        help="Epoch interval between two STGS bias measurements")
    parser.add_argument("--stgs_grad_bias_reference_samples", type=int, default=0,
                        help="Number of REINFORCE reference samples per bias estimate (>=1 enables metric)")
    parser.add_argument("--stgs_grad_bias_reference_batch_size", type=int, default=0,
                        help="Batch size for REINFORCE reference bias runs (0 reuses training batch size)")
    parser.add_argument("--stgs_grad_bias_reference_use_baseline", type=str2bool, default=True,
                        help="Enable REINFORCE baseline when computing STGS bias")
    parser.add_argument("--stgs_grad_bias_reference_reward_scale", type=float, default=1.0,
                        help="Reward scale to apply in the reference REINFORCE bias estimator")
    parser.add_argument("--stgs_grad_bias_reference_baseline_beta", type=float, default=0.9,
                        help="EMA beta for the reference REINFORCE baseline during bias estimation")
    parser.add_argument("--reinforce_grad_variance_samples", type=int, default=0,
                        help="Number of REINFORCE gradient samples to use when estimating variance (>=2 enables metric)")
    parser.add_argument("--reinforce_grad_variance_period", type=int, default=1,
                        help="Epoch interval between two REINFORCE gradient variance measurements")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Parallelization parameters
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel workers (1 = sequential)")
    
    # Target selection parameters
    parser.add_argument("--target_indices", type=str, default=None,
                        help="Comma-separated list of target indices to optimize (e.g., '0,1,5')")
    
    # W&B parameters
    parser.add_argument("--wandb_project", type=str, default="prompt-optimization-batch",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity name")
    
    # Perplexity parameters
    parser.add_argument("--promptLambda", type=float, default=0.0,
                        help="Weight for prompt perplexity loss")
    parser.add_argument("--complLambda", type=float, default=0.0,
                        help="Weight for completion perplexity loss")
    parser.add_argument("--promptTfComplLambda", type=float, default=0.0,
                        help="Weight for prompt + teacher-forced completion perplexity loss")
    
    # Metrics parameters
    parser.add_argument("--metric_groups", type=str, default=None,
                        help="Comma-separated list of metric groups to compute (None = all)")
    parser.add_argument("--skip_metric_groups", type=str, default=None,
                        help="Comma-separated list of metric groups to skip")
    
    # SentenceBERT parameters
    parser.add_argument("--sentencebert_model", type=str, default="all-MiniLM-L6-v2",
                        help="SentenceBERT model to use for semantic similarity")
    
    # BERTScore parameters
    parser.add_argument("--bertscore_model", type=str, default="distilbert-base-uncased",
                        help="Model to use for BERTScore computation")

    # Per-epoch semantic metrics
    parser.add_argument("--semantic_metrics_every_n_epochs", type=int, default=128,
                        help="Compute BERT/SentenceBERT every N epochs (0=disabled)")

    # Early stopping parameters
    parser.add_argument("--early_stop_on_exact_match", type=str2bool, default=False,
                        help="Stop optimization when generated output exactly matches target (default: False)")
    parser.add_argument("--early_stop_loss_threshold", type=float, default=0.0,
                        help="Stop optimization when loss drops below this threshold (0 or negative to disable)")
    parser.add_argument("--early_stop_embsim_lcs_ratio_threshold", type=float, default=1.0,
                        help="Stop when embsim LCS ratio >= threshold; active when run_discrete_embsim_validation=True (>1.0 = disabled)")

    # Fixed logit distribution parameters
    parser.add_argument("--fixed_gt_prefix_n", type=int, default=0,
                        help="Fix first N prompt positions to GT token at rank 1 (prompt reconstruction only)")
    parser.add_argument("--fixed_gt_suffix_n", type=int, default=0,
                        help="Fix last N prompt positions to GT token at rank 1 (prompt reconstruction only)")
    parser.add_argument("--fixed_gt_prefix_rank2_n", type=int, default=0,
                        help="Fix next N prefix positions to GT token at rank 2 (after fixed_gt_prefix_n)")
    parser.add_argument("--fixed_gt_suffix_rank2_n", type=int, default=0,
                        help="Fix N positions before fixed_gt_suffix_n to GT token at rank 2")
    parser.add_argument("--fixed_prefix_text", type=str, default=None,
                        help="Text string whose tokens form a fixed one-hot prefix")
    parser.add_argument("--fixed_suffix_text", type=str, default=None,
                        help="Text string whose tokens form a fixed one-hot suffix")

    # EoS position regularization parameters
    parser.add_argument("--eos_reg_lambda", type=float, default=0.0,
                        help="Weight for position-weighted EoS regularization (0 = disabled)")
    parser.add_argument("--eos_reg_schedule", type=str, default="linear",
                        choices=["linear", "exponential", "power"],
                        help="Weight schedule: linear, exponential, or power")
    parser.add_argument("--eos_reg_alpha", type=float, default=1.0,
                        help="Schedule parameter: exponent p for power=(1-i/N)^p, or alpha for exp(-alpha*i/N)")

    # STGS initialization strategy
    parser.add_argument("--init_strategy", type=str, default="randn",
                        choices=["randn", "zeros", "normal", "one_hot_random",
                                 "embedding_similarity", "lm_target_prior", "mlm_mask"],
                        help="Initialization strategy for STGS learnable logits")
    parser.add_argument("--init_std", type=float, default=0.0,
                        help="Noise std for structured init strategies (normal, lm_target_prior, mlm_mask)")
    parser.add_argument("--init_mlm_model", type=str, default="distilbert-base-uncased",
                        help="MLM checkpoint for 'mlm_mask' initialization strategy")
    parser.add_argument("--init_mlm_top_k", type=int, default=1000,
                        help="Top k indices that are initialized for 'mlm_mask' initialization strategy")

    # Discrete validation pass (opt-in)
    parser.add_argument("--run_discrete_validation", type=str2bool, default=False,
                        help="Run a greedy discrete decode at each epoch and log discrete_* and all_* metrics")
    parser.add_argument("--run_discrete_embsim_validation", type=str2bool, default=False,
                        help="At each epoch, find the nearest-embedding token at each position "
                             "(cosine sim to soft embedding) and run greedy decode; logs embsim_* metrics.")
    parser.add_argument("--embsim_similarity", type=str, default="cossim",
                        choices=["cossim", "dotproduct", "l2"],
                        help="Similarity metric used by embsim validation (cossim, dotproduct, or l2)")
    parser.add_argument("--embsim_use_input_logits", type=str2bool, default=True,
                        help="Embsim validation source: True=softmax(input logits) [default], "
                             "False=STGS y_soft (Gumbel-Softmax outputs)")
    parser.add_argument("--embsim_teacher_forcing", type=str2bool, default=False,
                        help="Embsim validation decode: True=single teacher-forced forward pass "
                             "(faster, no autoregressive generation), False=model.generate() [default]")
    parser.add_argument("--embsim_temperature", type=float, default=1.0,
                        help="Temperature applied to raw logits before softmax in embsim validation "
                             "when embsim_use_input_logits=True. Match to training temperature "
                             "(e.g. 0.05). Default: 1.0 (backward-compatible).")

    # Temperature annealing (opt-in)
    parser.add_argument("--temperature_anneal_schedule", type=str, default="none",
                        choices=["none", "linear", "cosine"],
                        help="Temperature annealing schedule (none=disabled)")
    parser.add_argument("--temperature_anneal_min", type=float, default=0.1,
                        help="Minimum temperature for annealing")
    parser.add_argument("--temperature_anneal_epochs", type=int, default=0,
                        help="Number of epochs over which to anneal temperature (0=use total epochs)")
    parser.add_argument("--temperatureAnnealRegLambda", type=float, default=0.0,
                        help="Regularization weight for temperature annealing loss when learnable_temperature=True (0=disabled)")
    parser.add_argument("--temperatureAnnealRegMode", type=str, default="mse",
                        choices=["mse", "one_sided"],
                        help="Temperature reg penalty: 'mse' (symmetric) or 'one_sided' (only when tau > target)")
    parser.add_argument("--temperatureLossCouplingLambda", type=float, default=0.0,
                        help="Weight for loss-coupled temperature regularization: "
                             "reg = λ * L_main.detach() / τ pushes τ up when loss is high (0=disabled)")

    # Prompt distribution entropy regularization (opt-in)
    parser.add_argument("--promptDistEntropyLambda", type=float, default=0.0,
                        help="Weight for prompt distribution entropy regularization (0=disabled)")

    # Commitment loss (VQ-VAE style, opt-in)
    parser.add_argument("--commitmentLambda", type=float, default=0.0,
                        help="Weight for commitment loss (L2 between soft and nearest discrete embedding)")
    parser.add_argument("--commitment_similarity", type=str, default="argmax",
                        choices=["argmax", "embsim-dot", "embsim-cos", "embsim-l2"],
                        help="Nearest-token selection for commitment loss: "
                             "'argmax' (highest logit), 'embsim-dot' (dot-product), 'embsim-cos' (cosine), "
                             "'embsim-l2' (L2 distance)")

    # Commitment loss position weighting (opt-in)
    parser.add_argument("--commitment_pos_weight_schedule", type=str, default="uniform",
                        choices=["uniform", "linear_inc", "linear_dec", "exp_inc", "exp_dec"],
                        help="Position-weight schedule for commitment loss. "
                             "'uniform' = equal weighting (default); "
                             "'linear_inc' = weight increases linearly (step * position_index); "
                             "'linear_dec' = reverse of linear_inc; "
                             "'exp_inc' = base^position_index; "
                             "'exp_dec' = base^(seq_len-1-position_index). "
                             "Weights are normalized to sum to 1 before applying.")
    parser.add_argument("--commitment_pos_weight_step", type=float, default=10.0,
                        help="Linear step size for 'linear_inc'/'linear_dec' schedules "
                             "(weights: step, 2*step, ..., seq_len*step before normalization)")
    parser.add_argument("--commitment_pos_weight_base", type=float, default=2.0,
                        help="Base for 'exp_inc'/'exp_dec' schedules (base^i before normalization)")

    # Main-loss token-position weighting (opt-in)
    parser.add_argument("--loss_pos_weight_schedule", type=str, default="uniform",
                        choices=["uniform", "linear_inc", "linear_dec", "exp_inc", "exp_dec"],
                        help="Position-weight schedule for main loss terms "
                             "(crossentropy, embxentropy, hinge, embedded). "
                             "'uniform'=no weighting (default); 'linear_inc'=left-light right-heavy; "
                             "'linear_dec'=left-heavy right-light; 'exp_inc'/'exp_dec'=exponential variants.")
    parser.add_argument("--loss_pos_weight_step", type=float, default=1.0,
                        help="Multiplier for linear schedules: weight[i] = step*(i+1) or step*(T-i). "
                             "Only used when loss_pos_weight_schedule in {linear_inc, linear_dec}.")
    parser.add_argument("--loss_pos_weight_base", type=float, default=2.0,
                        help="Base for exponential schedules: weight[i] = base^i or base^(T-1-i). "
                             "Only used when loss_pos_weight_schedule in {exp_inc, exp_dec}.")

    # Gumbel noise scale parameters (opt-in)
    parser.add_argument("--gumbel_noise_scale", type=float, default=1.0,
                        help="Fixed multiplier for Gumbel noise (0=no noise, 1=standard Gumbel)")
    parser.add_argument("--adaptive_gumbel_noise", type=str2bool, default=False,
                        help="Adaptively reduce Gumbel noise scale as loss decreases")
    parser.add_argument("--adaptive_gumbel_noise_beta", type=float, default=0.9,
                        help="EMA coefficient for loss tracking in adaptive Gumbel noise (higher = slower adaptation)")
    parser.add_argument("--adaptive_gumbel_noise_min_scale", type=float, default=0.0,
                        help="Minimum noise scale floor for adaptive Gumbel noise")

    # Pre/post-STGS dropout (opt-in)
    parser.add_argument("--stgs_input_dropout", type=float, default=0.0,
                        help="Dropout applied to input logits before Gumbel-softmax sampling "
                             "(0 = disabled). F.dropout zeros ~p of vocab dims and rescales by 1/(1-p).")
    parser.add_argument("--stgs_output_dropout", type=float, default=0.0,
                        help="Dropout applied to y_soft (after softmax, before hard sampling) "
                             "(0 = disabled). F.dropout zeros ~p of prob entries and rescales by 1/(1-p).")

    # LoRA-factored prompt logits (opt-in)
    parser.add_argument("--logits_lora_rank", type=int, default=0,
                        help="LoRA rank for prompt logit factorisation (0 = disabled, "
                             "uses standard free_logits; >0 uses A@B decomposition).")

    # Diversity analysis (opt-in)
    parser.add_argument("--diversity_n_samples", type=int, default=0,
                        help="Number of Gumbel draws per temperature for diversity analysis "
                             "(0 = disabled).")
    parser.add_argument("--diversity_log_every", type=int, default=10,
                        help="Run diversity analysis every N epochs (used when diversity_n_samples > 0).")
    parser.add_argument("--diversity_temperatures", type=str, default="",
                        help="Comma-separated temperatures for diversity analysis "
                             "(e.g. '0.1,0.5,1.0,2.0'); empty falls back to the run temperature.")

    # Superposition diagnostics (opt-in)
    parser.add_argument("--superposition_metric_every", type=int, default=0,
                        help="Run superposition z_i-z_j diagnostics every N epochs (0 = disabled).")
    parser.add_argument("--superposition_metric_modes", type=str, default="dot,cos,l2",
                        help="Comma-separated modes for superposition maps: dot,cos,l2.")
    parser.add_argument("--superposition_vocab_top_k", type=int, default=256,
                        help="Top-K common tokens used to build z_i-z_j pair space "
                             "(<=0 means all allowed tokens).")
    parser.add_argument("--superposition_vocab_source", type=str, default="wikipedia",
                        choices=["wikipedia", "dataset", "none"],
                        help="Source used to rank common tokens for top-K subset.")
    parser.add_argument("--superposition_vocab_dataset_path", type=str, default=None,
                        help="Path to local dataset used when superposition_vocab_source=dataset. "
                             "If unset in batch mode, falls back to --dataset_path.")
    parser.add_argument("--superposition_vocab_hf_name", type=str, default="lucadiliello/english_wikipedia",
                        help="HF dataset name used when superposition_vocab_source=wikipedia.")
    parser.add_argument("--superposition_vocab_hf_split", type=str, default="train",
                        help="HF split used when superposition_vocab_source=wikipedia.")
    parser.add_argument("--superposition_vocab_num_texts", type=int, default=1000,
                        help="Number of texts sampled to estimate token frequency for top-K ranking.")
    parser.add_argument("--superposition_entropy_temperature", type=float, default=1.0,
                        help="Temperature used in softmax(score/T) for entropy over (i,j) pairs.")

    # Periodic discrete reinitialization (opt-in)
    parser.add_argument("--discrete_reinit_epoch", type=int, default=0,
                        help="Snap free_logits to discrete projection every N epochs (0 = disabled)")
    parser.add_argument("--discrete_reinit_snap", type=str, default="argmax",
                        choices=["argmax", "embsim-dot", "embsim-cos", "embsim-l2"],
                        help="Projection method for periodic discrete reinitialization")
    parser.add_argument("--discrete_reinit_prob", type=float, default=1.0,
                        help="In top-k discrete reinit mode, probability mass assigned to the snapped token "
                             "(1.0 keeps legacy spike behavior when discrete_reinit_topk=0).")
    parser.add_argument("--discrete_reinit_topk", type=int, default=0,
                        help="Preserve this many top-k tokens from the pre-reinit distribution with the "
                             "residual probability mass; 0 disables the softer top-k reinit.")
    parser.add_argument("--discrete_reinit_entropy_threshold", type=float, default=0.0,
                        help="Per-position entropy threshold for discrete reinit. 0 disables it. "
                             "If set with discrete_reinit_epoch > 0, only positions below the threshold "
                             "are reinitialized at each reinit event; if discrete_reinit_epoch=0, the "
                             "threshold is checked every epoch.")
    parser.add_argument("--discrete_reinit_embsim_probs", type=str, default="input_logits",
                        choices=["input_logits", "gumbel_soft"],
                        help="Distribution source used for embsim-based discrete reinit and entropy gating.")

    # Exponential logit weight decay (opt-in)
    parser.add_argument("--logit_decay", type=float, default=0.0,
                        help="Multiplicative decay applied to free_logits after each optimizer step "
                             "(0 = disabled; e.g. 0.999 for mild decay). "
                             "Shrinks logit magnitudes exponentially toward zero.")

    # PPO-KL distributional trust-region parameters
    parser.add_argument("--ppo_kl_lambda", type=float, default=0.0,
                        help="Weight for PPO-KL trust-region regularizer (0 = disabled)")
    parser.add_argument("--ppo_kl_mode", type=str, default="soft",
                        choices=["soft", "hinge"],
                        help="'soft': penalize all KL divergence from reference; "
                             "'hinge': penalize only when per-position KL > ppo_kl_epsilon (PPO-clip analogue)")
    parser.add_argument("--ppo_kl_epsilon", type=float, default=0.0,
                        help="KL divergence threshold for hinge mode (ignored in soft mode)")
    parser.add_argument("--ppo_ref_update_period", type=int, default=10,
                        help="Update the PPO reference snapshot every N epochs (0 = never update after init)")

    # Learnable prompt length (differentiable soft masking toward EoS)
    parser.add_argument("--prompt_length_learnable", type=str2bool, default=False,
                        help="Learn prompt length jointly with token logits via soft EoS masking")
    parser.add_argument("--prompt_length_alpha_init", type=float, default=0.0,
                        help="Initial alpha: lambda=n_free*sigmoid(alpha) "
                             "(0.0->n_free/2 active; -3.0->~5%% active)")
    parser.add_argument("--prompt_length_beta", type=float, default=5.0,
                        help="Gate sharpness (higher=sharper; inf->hard threshold)")
    parser.add_argument("--prompt_length_reg_lambda", type=float, default=0.0,
                        help="Weight gamma for R(lambda)=gamma*lambda length regularizer (0=disabled)")
    parser.add_argument("--prompt_length_eos_spike", type=float, default=10.0,
                        help="Logit spike for EoS at inactive positions (sets EoS sampling probability)")
    parser.add_argument("--prompt_length_mask_eos_attention", type=str2bool, default=False,
                        help="Mask LM attention to EoS-sampled positions. "
                             "When ON: no LM-attention gradient flows back through EoS positions. "
                             "Gradient still flows through the gate (length_alpha) and STGS.")

    parser.add_argument("--assemble_strategy", type=str, default="identity",
                        choices=["identity", "product_of_speaker"],
                        help="Transform applied to assembled prompt logits before optimization.")
    parser.add_argument("--pos_lm_name", type=str, default=None,
                        help="Auxiliary LM used by product_of_speaker. Defaults to --model_name.")
    parser.add_argument("--pos_prompt", type=str, default=None,
                        help="Fixed task prompt used to seed the product_of_speaker LM.")
    parser.add_argument("--pos_prompt_template", type=str, default=None,
                        help="PoS prompt template for batch single-target runs. Supports {target}.")
    parser.add_argument("--pos_use_chat_template", type=str2bool, default=False,
                        help="Wrap the product_of_speaker prompt with tokenizer.apply_chat_template "
                             "when the auxiliary tokenizer provides one.")
    parser.add_argument("--pos_lm_temperature", type=float, default=1.0,
                        help="Temperature applied to the auxiliary LM logits in product_of_speaker.")
    parser.add_argument("--pos_lm_top_p", type=float, default=1.0,
                        help="Top-p filter applied to the auxiliary LM logits in product_of_speaker.")
    parser.add_argument("--pos_lm_top_k", type=int, default=0,
                        help="Top-k filter applied to the auxiliary LM logits in product_of_speaker.")

    return parser


def parse_args(argv=None):
    """Parse command-line arguments."""
    return build_parser().parse_args(argv)


def apply_loss_arg_expansions(args):
    """Expand opt-in loss flags into the composite losses string."""
    if args.promptLambda > 0.0:
        args.losses += '+promptPerplexity'
    if args.complLambda > 0.0:
        args.losses += '+completionPerplexity'
    if args.promptTfComplLambda > 0.0:
        args.losses += '+promptTfComplPerplexity'
    if args.eos_reg_lambda > 0.0:
        args.losses += '+eosPositionReg'
    if args.promptDistEntropyLambda > 0.0:
        args.losses += '+promptDistEntropy'
    if args.commitmentLambda > 0.0:
        args.losses += '+commitmentLoss'
    return args

def main():
    """Main entry point."""
    args = parse_args()
    #if args.decouple_learnable_temperature \
    #and not args.learnable_temperature:
    #    raise ValueError("decouple_learnable_temperature requires learnable_temperature")

    # Parse target indices if provided
    target_indices = None
    if args.target_indices:
        target_indices = [int(idx) for idx in args.target_indices.split(",")]
    
    # Parse metric groups if provided
    metric_groups = str2list(args.metric_groups)
    skip_metric_groups = str2list(args.skip_metric_groups)
    
    apply_loss_arg_expansions(args)

    # Prepare configuration dictionary
    config = vars(args)
    
    # Run batch optimization
    batch_optimize(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        config=config,
        num_workers=args.num_workers,
        target_indices=target_indices,
        metric_groups=metric_groups
    )

if __name__ == "__main__":
    main()
