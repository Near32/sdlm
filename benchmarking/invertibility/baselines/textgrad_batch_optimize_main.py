"""
Batch optimisation driver that mirrors benchmarking/batch_optimize_main.py
but routes optimisation through TextGrad (benchmarking/textgrad_main.py).
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import wandb
from tqdm import tqdm

from metrics_aggregator import MetricsAggregator
from metrics_logging import MetricsLogger
from evaluation_utils import evaluate_generated_output

from textgrad_main import optimize_inputs as textgrad_optimize_inputs
from textgrad_main import setup_model_and_tokenizer as textgrad_setup_model_and_tokenizer

logger = logging.getLogger("textgrad_batch_optimize")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def setup_model_and_tokenizer(model_name: str, device: str = "cpu", model_precision: str = "full"):
    """
    Thin wrapper that delegates to textgrad_main.setup_model_and_tokenizer so both
    scripts share identical model/tokeniser setup behaviour.
    """
    return textgrad_setup_model_and_tokenizer(model_name=model_name, device=device, model_precision=model_precision)


def load_dataset(dataset_path: str) -> Dict[str, Any]:
    """
    Load dataset from a JSON file.
    """
    logger.info(f"Loading dataset from {dataset_path}")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    logger.info(f"Loaded {len(dataset.get('samples', []))} samples")
    return dataset


def prepare_targets(dataset: Dict[str, Any], target_indices: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """
    Prepare the list of targets, optionally filtering by indices.
    """
    targets = dataset.get("samples", [])
    if target_indices is not None:
        targets = [targets[i] for i in target_indices if i < len(targets)]
        logger.info(f"Selected {len(targets)} targets based on provided indices")
    return targets


def optimise_single_target(
    target_info: Dict[str, Any],
    model,
    tokenizer,
    device: torch.device,
    config: Dict[str, Any],
    run_id: str,
    output_dir: str,
    metric_groups: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Optimise a prompt for a single target sentence using TextGrad.
    """
    target_id = target_info["id"]
    target_text = target_info["text"]
    k_target = target_info["k_target"]
    target_perplexity = target_info.get("perplexity")
    pre_prompt = target_info.get("pre_prompt", config.get("pre_prompt"))

    logger.info(f"Optimising prompt for target {target_id}: '{target_text}'")

    run_name = f"{run_id}_target{target_id}_k{k_target}"

    target_config = config.copy()
    target_config.update(
        {
            "target_id": target_id,
            "target_text": target_text,
            "target_k": k_target,
            "target_avg_rank": target_info.get("avg_rank", 0),
            "target_length": target_info.get("length", len(target_text.split())),
            "target_perplexity": target_perplexity,
            "pre_prompt": pre_prompt,
        }
    )

    metrics_logger = MetricsLogger(
        run_id=f"{run_id}_{target_id}",
        output_dir=f"{output_dir}/target_{target_id}",
        wandb_project=config.get("wandb_project"),
        wandb_entity=config.get("wandb_entity"),
    )
    metrics_logger.init_wandb(
        config=target_config,
        name=run_name,
        group=run_id,
        job_type="single_target_optimization_textgrad",
    )

    target_tokens = tokenizer(target_text, return_tensors="pt").input_ids[0].cpu().tolist()

    generated_tokens_tensor, optimised_prompt_body, feedback_history = textgrad_optimize_inputs(
        model=model,
        tokenizer=tokenizer,
        device=device,
        target_text=target_text,
        pre_prompt=pre_prompt,
        seq_len=config["seq_len"],
        epochs=config["epochs"],
        temperature=config.get("temperature", 1.0),
        vocab_threshold=config.get("vocab_threshold", 0.5),
        filter_vocab=config.get("filter_vocab", True),
        kwargs=config,
    )

    generated_tokens = generated_tokens_tensor[0].cpu().tolist()
    optimised_tokens = tokenizer(
        optimised_prompt_body,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[0].cpu().tolist() if optimised_prompt_body else []

    evaluation_metrics = evaluate_generated_output(
        generated_tokens=generated_tokens,
        target_tokens=target_tokens,
        tokenizer=tokenizer,
        metric_groups=metric_groups,
        skip_metric_groups=config.get("skip_metric_groups"),
        device=str(device),
        sentencebert_model=config.get("sentencebert_model", "all-MiniLM-L6-v2"),
        bertscore_model=config.get("bertscore_model", "distilbert-base-uncased"),
    )

    metrics_logger.log_metrics(evaluation_metrics)

    result = {
        "target_id": target_id,
        "target_text": target_text,
        "target_k": k_target,
        "target_perplexity": target_perplexity,
        "optimised_prompt_body": optimised_prompt_body,
        "optimised_tokens": optimised_tokens,
        "generated_tokens": generated_tokens,
        "generated_text": evaluation_metrics["generated_text"],
        "evaluation": evaluation_metrics,
        "feedback_history": feedback_history,
        "num_feedback_steps": len(feedback_history),
    }

    metrics_logger.save_to_file(result, "result.json")

    target_output_dir = Path(f"{output_dir}/target_{target_id}")
    target_output_dir.mkdir(parents=True, exist_ok=True)
    with open(target_output_dir / "optimised_prompt.txt", "w") as f:
        f.write(optimised_prompt_body)
    with open(target_output_dir / "feedback_history.json", "w") as f:
        json.dump(feedback_history, f, indent=2)
    with open(target_output_dir / "generated_tokens.json", "w") as f:
        json.dump(generated_tokens, f)

    metrics_logger.finish()

    logger.info(f"Optimisation completed for target {target_id}")
    logger.info(f"Optimised prompt body: '{optimised_prompt_body}'")
    logger.info(f"Exact match: {evaluation_metrics.get('exact_match', 0)}")
    logger.info(f"Token accuracy: {evaluation_metrics.get('token_accuracy', 0):.4f}")

    return result


def process_targets_sequential(
    targets: List[Dict[str, Any]],
    model,
    tokenizer,
    device: torch.device,
    config: Dict[str, Any],
    run_id: str,
    output_dir: str,
    metrics_aggregator: MetricsAggregator,
    metric_groups: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Process targets sequentially.
    """
    results = {}
    for target in tqdm(targets, desc="Optimising targets (TextGrad)"):
        result = optimise_single_target(
            target_info=target,
            model=model,
            tokenizer=tokenizer,
            device=device,
            config=config,
            run_id=run_id,
            output_dir=output_dir,
            metric_groups=metric_groups,
        )
        results[target["id"]] = result
        metrics_aggregator.add_sample(result["evaluation"], k_value=target["k_target"])
    return results


def batch_optimize(
    dataset_path: str,
    model_name: str,
    output_dir: str,
    config: Dict[str, Any],
    target_indices: Optional[List[int]] = None,
    metric_groups: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Batch optimise prompts for multiple targets using TextGrad.
    """
    run_id = wandb.util.generate_id()
    logger.info(f"Starting TextGrad batch optimisation with run ID: {run_id}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = load_dataset(dataset_path)
    targets = prepare_targets(dataset, target_indices)

    model, tokenizer = setup_model_and_tokenizer(
        model_name=model_name,
        device=device,
        model_precision=config.get("model_precision", "full"),
    )

    if config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    if config.get("num_workers", 1) != 1:
        logger.warning(
            "TextGrad batch optimisation currently runs sequentially; ignoring num_workers=%s",
            config.get("num_workers"),
        )

    full_output_dir = Path(output_dir) / run_id
    full_output_dir.mkdir(parents=True, exist_ok=True)

    metrics_aggregator = MetricsAggregator()

    results = process_targets_sequential(
        targets=targets,
        model=model,
        tokenizer=tokenizer,
        device=device,
        config=config,
        run_id=run_id,
        output_dir=str(full_output_dir),
        metrics_aggregator=metrics_aggregator,
        metric_groups=metric_groups,
    )

    metrics_logger = MetricsLogger(
        run_id=run_id,
        output_dir=str(full_output_dir),
        wandb_project=config.get("wandb_project"),
        wandb_entity=config.get("wandb_entity"),
    )
    metrics_logger.init_wandb(
        group=run_id,
        config={
            **config,
            "dataset_path": dataset_path,
            "model_name": model_name,
            "num_targets": len(targets),
            "metadata": dataset.get("metadata", {}),
        },
        name=f"textgrad_batch_optimization_{run_id}",
        job_type="batch_coordination_textgrad",
    )

    summary = metrics_aggregator.get_summary()
    metrics_logger.log_summary(summary)
    summary["results"] = results

    metrics_logger.save_to_file(
        data={
            "run_id": run_id,
            "summary": summary,
            "config": config,
            "dataset_metadata": dataset.get("metadata", {}),
        },
        filename="batch_results.json",
    )
    metrics_logger.finish()

    return summary


def str2bool(v):
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def str2list(v):
    """Convert comma-separated string to list."""
    if v is None:
        return None
    return [item.strip() for item in v.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch optimise prompts with TextGrad for multiple targets.",
    )

    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset file.")
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-135M", help="Model name.")
    parser.add_argument("--model_precision", type=str, default="full", choices=["full", "half"])
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=False)
    parser.add_argument("--output_dir", type=str, default="textgrad_batch_optimization_results")

    parser.add_argument("--seq_len", type=int, default=20, help="Prompt body length to optimise.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of TextGrad iterations.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for model completion.")
    parser.add_argument("--filter_vocab", type=str2bool, default=True, help="Enable cosine filtering of vocab.")
    parser.add_argument("--vocab_threshold", type=float, default=0.5, help="Cosine similarity threshold for vocab filtering.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Max tokens to sample during completion.")
    parser.add_argument("--pre_prompt", type=str, default=None, help="Optional fixed prefix before the learnable body.")

    parser.add_argument("--backward_engine", type=str, default="gpt-4o", help="TextGrad backward engine identifier.")
    parser.add_argument("--optimizer_engine", type=str, default=None, help="Engine for TextGrad updates (defaults to backward engine).")
    parser.add_argument("--textgrad_constraints", type=str, nargs="*", default=None, help="Natural-language constraints for TextGrad.")
    parser.add_argument("--textgrad_system_prompt", type=str, default=None, help="System prompt for TextGrad evaluator.")
    parser.add_argument("--textgrad_verbose", type=int, default=0, help="Verbosity level for TextGrad updates.")
    parser.add_argument("--textgrad_do_sample", type=str2bool, default=True, help="Whether to sample during generation.")
    parser.add_argument("--stop_on_done", type=str2bool, default=True, help="Stop optimisation when evaluator replies DONE.")

    parser.add_argument("--num_workers", type=int, default=1, help="Parallel workers (TextGrad version runs sequentially).")
    parser.add_argument("--target_indices", type=str, default=None, help="Comma-separated list of dataset indices to optimise.")

    parser.add_argument("--wandb_project", type=str, default="prompt-optimization-textgrad-batch", help="Weights & Biases project.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity.")

    parser.add_argument("--metric_groups", type=str, default=None, help="Comma-separated metric groups to compute.")
    parser.add_argument("--skip_metric_groups", type=str, default=None, help="Comma-separated metric groups to skip.")
    parser.add_argument("--sentencebert_model", type=str, default="all-MiniLM-L6-v2", help="SentenceBERT model name.")
    parser.add_argument("--bertscore_model", type=str, default="distilbert-base-uncased", help="BERTScore model name.")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.target_indices:
        target_indices = [int(idx) for idx in args.target_indices.split(",")]
    else:
        target_indices = None

    metric_groups = str2list(args.metric_groups)
    skip_metric_groups = str2list(args.skip_metric_groups)

    config = vars(args)
    config["skip_metric_groups"] = skip_metric_groups

    batch_optimize(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        config=config,
        target_indices=target_indices,
        metric_groups=metric_groups,
    )


if __name__ == "__main__":
    main()
