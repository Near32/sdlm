"""
CLI entry point for partially-conditioned multi-target prompt inversion.

Mirrors batch_optimize_main.py structure.
Optimizes a single shared prompt p across all (x, R, y) pairs using the PC pipeline.
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import wandb

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(1, str(Path(__file__).parent.parent.parent))
sys.path.insert(2, str(Path(__file__).parent.parent / "tinylora"))

from batch_optimize_main import setup_model_and_tokenizer
from sdlm import STGS
from sdlm.stgs_diff_model import STGSDiffModel
from pc_main import pc_optimize_inputs, evaluate_shared_prompt
from pc_weave_logging import init_weave

from rewards import (
    extract_gsm8k_answer,
    extract_boxed_answer,
    extract_final_number,
    compare_answers,
)

logger = logging.getLogger("batch_optimize_pc")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v!r}")


def subsample_split(split: List[Dict], size: int, seed: int) -> List[Dict]:
    if size <= 0 or size >= len(split):
        return split
    rng = random.Random(seed)
    return rng.sample(split, size)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_pc_dataset(dataset_path: str) -> Dict:
    """
    Load a PC-format JSON dataset.

    Expected format:
    {
      "metadata": {"evaluation_type": "partially_conditioned", ...},
      "train": [{"id": ..., "input": ..., "reasoning": ..., "answer": ...}],
      "validation": [{"id": ..., "input": ..., "answer": ...}],
      "test":       [{"id": ..., "input": ..., "answer": ...}]
    }
    """
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    logger.info(f"Loaded PC dataset from {dataset_path}: "
                f"{len(dataset.get('train', []))} train, "
                f"{len(dataset.get('validation', []))} val, "
                f"{len(dataset.get('test', []))} test samples")
    return dataset


def load_hf_gsm8k(
    train_size: int = 50,
    val_size: int = 20,
    test_size: int = 100,
    train_seed: int = 42,
    val_seed: int = 43,
    test_seed: int = 44,
    subset: str = "main",
) -> Dict:
    """
    Load openai/gsm8k from HuggingFace datasets.

    Val is carved from train BEFORE train subsample to prevent contamination.
    train[*].reasoning = CoT before "####"
    train[*].answer    = number after "####"
    val/test[*].answer = raw number only
    """
    from datasets import load_dataset as hf_load_dataset

    logger.info(f"Loading openai/gsm8k subset={subset}")
    ds = hf_load_dataset("openai/gsm8k", subset)

    def _extract_cot_and_answer(solution: str) -> Tuple[str, str]:
        """Split '#### <number>' from CoT reasoning."""
        if "####" in solution:
            parts = solution.split("####", 1)
            reasoning = parts[0].strip()
            answer = parts[1].strip().replace(",", "")
        else:
            reasoning = solution.strip()
            answer = ""
        return reasoning, answer

    # ----- train split -----
    raw_train = list(ds["train"])
    # Carve val from the full train set before subsampling
    val_pool = subsample_split(raw_train, val_size, val_seed)
    val_pool_ids = {id(item) for item in val_pool}

    # Remaining candidates for train subsampling (exclude val)
    train_candidates = [item for item in raw_train if id(item) not in val_pool_ids]
    train_subset = subsample_split(train_candidates, train_size, train_seed)

    train_records = []
    for i, item in enumerate(train_subset):
        reasoning, answer = _extract_cot_and_answer(item["answer"])
        train_records.append({
            "id": f"train_{i}",
            "input": item["question"],
            "reasoning": reasoning,
            "answer": answer,
        })

    val_records = []
    for i, item in enumerate(val_pool):
        _, answer = _extract_cot_and_answer(item["answer"])
        val_records.append({
            "id": f"val_{i}",
            "input": item["question"],
            "answer": answer,
        })

    # ----- test split -----
    raw_test = list(ds["test"])
    test_subset = subsample_split(raw_test, test_size, test_seed)
    test_records = []
    for i, item in enumerate(test_subset):
        _, answer = _extract_cot_and_answer(item["answer"])
        test_records.append({
            "id": f"test_{i}",
            "input": item["question"],
            "answer": answer,
        })

    dataset = {
        "metadata": {
            "evaluation_type": "partially_conditioned",
            "dataset_source": f"openai/gsm8k/{subset}",
        },
        "train": train_records,
        "validation": val_records,
        "test": test_records,
    }
    logger.info(f"GSM8K loaded: {len(train_records)} train, "
                f"{len(val_records)} val, {len(test_records)} test")
    return dataset


# ---------------------------------------------------------------------------
# Main optimization function
# ---------------------------------------------------------------------------

def batch_pc_optimize(config: Dict) -> Dict:
    """
    Full pipeline: load dataset, build model, run pc_optimize_inputs, evaluate.
    """
    # ---- model ----
    model, tokenizer = setup_model_and_tokenizer(
        config["model_name"],
        config.get("device", "cpu"),
        config.get("model_precision", "full"),
    )
    device = str(next(model.parameters()).device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ---- STGSDiffModel ----
    diff_model = STGSDiffModel(
        model=model,
        tokenizer=tokenizer,
        stgs_kwargs={
            "hard": config.get("diff_model_hard", True),
            "temperature": config.get("diff_model_temperature", 1.0),
            "learnable_temperature": False,
            "hidden_state_conditioning": False,
            "dropout": 0.0,
        },
        stgs_logits_generation=True,
    )

    # ---- dataset ----
    if config.get("dataset_path"):
        dataset = load_pc_dataset(config["dataset_path"])
    else:
        dataset = load_hf_gsm8k(
            train_size=config.get("train_size", 50),
            val_size=config.get("val_size", 20),
            test_size=config.get("test_size", 100),
            train_seed=config.get("train_seed", 42),
            val_seed=config.get("val_seed", 43),
            test_seed=config.get("test_seed", 44),
            subset=config.get("hf_dataset_subset", "main"),
        )

    train_records = dataset.get("train", [])
    val_records = dataset.get("validation", [])
    test_records = dataset.get("test", [])

    # Build (x_str, R_gt_str, y_raw_str) triples
    train_triples: List[Tuple[str, str, str]] = [
        (r["input"], r.get("reasoning", ""), r["answer"])
        for r in train_records
    ]
    val_pairs: List[Tuple[str, str]] = [
        (r["input"], r["answer"]) for r in val_records
    ]
    test_pairs: List[Tuple[str, str]] = [
        (r["input"], r["answer"]) for r in test_records
    ]

    # ---- extraction functions ----
    extraction_prompt = config.get("extraction_prompt", "Therefore, the final answer is $")
    extraction_fns: Dict = {
        "gsm8k_hash": extract_gsm8k_answer,
        "boxed": extract_boxed_answer,
        "final_number": extract_final_number,
        "extractor": extract_final_number,   # fallback: use final_number on Y_text
    }

    # ---- W&B ----
    run_name = config.get("run_name", "pc_opt")
    wandb_tags = config.get("wandb_tags", [])
    if isinstance(wandb_tags, str):
        wandb_tags = [t.strip() for t in wandb_tags.split(",") if t.strip()]

    wandb.init(
        project=config.get("wandb_project", "sdlm-pc-inversion"),
        entity=config.get("wandb_entity", None),
        name=run_name,
        config=config,
        tags=wandb_tags,
    )
    init_weave(
        project=config.get("weave_project") or config.get("wandb_project", "sdlm-pc-inversion"),
        entity=config.get("wandb_entity", None),
    )

    # ---- test eval callback (periodic during training) ----
    test_eval_every = config.get("test_eval_every", 0)
    test_eval_size = config.get("test_eval_size", len(test_pairs))
    test_pairs_eval = test_pairs[:test_eval_size] if test_eval_size < len(test_pairs) else test_pairs

    embedding_layer = model.get_input_embeddings()
    embedding_weights = embedding_layer.weight.detach()
    E_input_ids = tokenizer(
        extraction_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    with torch.no_grad():
        E_embeds = embedding_layer(E_input_ids)

    # Shared STGS module for evaluation callbacks (greedy, no noise)
    stgs_for_eval = STGS(
        vocab_size=model.config.vocab_size,
        stgs_hard=config.get("stgs_hard", True),
        stgs_hard_method=config.get("stgs_hard_method", "categorical"),
        init_temperature=config.get("temperature", 1.0),
        logits_normalize=config.get("logits_normalize", "none"),
        eps=config.get("eps", 1e-10),
        device=device,
    )

    _accumulate_embeds = config.get("efficient_generate", False)

    def _test_eval_callback(epoch: int, fl: torch.Tensor) -> Dict:
        if test_eval_every <= 0 or (epoch % test_eval_every != 0):
            return {}
        metrics = evaluate_shared_prompt(
            free_logits=fl,
            eval_pairs=test_pairs_eval,
            model=model,
            diff_model=diff_model,
            stgs_module=stgs_for_eval,
            embedding_weights=embedding_weights,
            tokenizer=tokenizer,
            device=device,
            E_input_ids=E_input_ids,
            E_embeds=E_embeds,
            extraction_fns=extraction_fns,
            max_new_tokens_reasoning=config.get("max_new_tokens_reasoning", 200),
            max_new_tokens_answer=config.get("max_new_tokens_answer", 20),
            eval_mode=config.get("val_eval_mode", "discrete"),
            use_chat_template=config.get("use_chat_template", False),
            epoch=epoch,
            accumulate_embeds=_accumulate_embeds,
        )
        p_tokens = fl.argmax(dim=-1).squeeze(0).tolist()
        p_text = tokenizer.decode(p_tokens, skip_special_tokens=False)
        log_dict = {f"test/{k}": v for k, v in metrics.items()}
        log_dict["test/p_text"] = p_text
        log_dict["epoch"] = epoch
        return log_dict

    # ---- build LossClass kwargs ----
    lc_kwargs = {
        "seq_len": config.get("seq_len", 20),
        "promptLambda": config.get("promptLambda", 0.0),
        "complLambda": config.get("complLambda", 0.0),
        "promptTfComplLambda": config.get("promptTfComplLambda", 0.0),
        "promptDistEntropyLambda": config.get("promptDistEntropyLambda", 0.0),
        "commitmentLambda": config.get("commitmentLambda", 0.0),
        "commitment_similarity": config.get("commitment_similarity", "argmax"),
        "eos_reg_lambda": config.get("eos_reg_lambda", 0.0),
        "eos_reg_schedule": config.get("eos_reg_schedule", "linear"),
        "eos_reg_alpha": config.get("eos_reg_alpha", 1.0),
        "loss_pos_weight_schedule": config.get("loss_pos_weight_schedule", "uniform"),
        "loss_pos_weight_step": config.get("loss_pos_weight_step", 1.0),
        "loss_pos_weight_base": config.get("loss_pos_weight_base", 2.0),
        "commitment_pos_weight_schedule": config.get("commitment_pos_weight_schedule", "uniform"),
        "commitment_pos_weight_step": config.get("commitment_pos_weight_step", 10.0),
        "commitment_pos_weight_base": config.get("commitment_pos_weight_base", 2.0),
    }

    # ---- set opt seed ----
    opt_seed = config.get("opt_seed", 0)
    torch.manual_seed(opt_seed)
    random.seed(opt_seed)

    # ---- run optimization ----
    result = pc_optimize_inputs(
        model=model,
        diff_model=diff_model,
        tokenizer=tokenizer,
        device=device,
        model_precision=config.get("model_precision", "full"),
        train_triples=train_triples,
        val_pairs=val_pairs if val_records else None,
        extraction_prompt=extraction_prompt,
        extraction_fns=extraction_fns,
        max_new_tokens_reasoning=config.get("max_new_tokens_reasoning", 200),
        max_new_tokens_answer=config.get("max_new_tokens_answer", 20),
        teacher_forcing_r=config.get("teacher_forcing_r", True),
        bptt=config.get("bptt", False),
        losses=config.get("losses", "crossentropy"),
        seq_len=config.get("seq_len", 20),
        epochs=config.get("epochs", 2000),
        learning_rate=config.get("learning_rate", 0.01),
        inner_batch_size=config.get("inner_batch_size", 4),
        temperature=config.get("temperature", 1.0),
        stgs_hard=config.get("stgs_hard", True),
        stgs_hard_method=config.get("stgs_hard_method", "categorical"),
        logits_normalize=config.get("logits_normalize", "none"),
        stgs_input_dropout=config.get("stgs_input_dropout", 0.0),
        stgs_output_dropout=config.get("stgs_output_dropout", 0.0),
        eps=config.get("eps", 1e-10),
        gumbel_noise_scale=config.get("gumbel_noise_scale", 1.0),
        adaptive_gumbel_noise=config.get("adaptive_gumbel_noise", False),
        adaptive_gumbel_noise_beta=config.get("adaptive_gumbel_noise_beta", 0.9),
        adaptive_gumbel_noise_min_scale=config.get("adaptive_gumbel_noise_min_scale", 0.0),
        logits_top_k=config.get("logits_top_k", 0),
        logits_top_p=config.get("logits_top_p", 1.0),
        logits_filter_penalty=config.get("logits_filter_penalty", 1e4),
        init_strategy=config.get("init_strategy", "randn"),
        init_std=config.get("init_std", 0.0),
        logits_lora_rank=config.get("logits_lora_rank", 0),
        max_gradient_norm=config.get("max_gradient_norm", 0.0),
        temperature_anneal_schedule=config.get("temperature_anneal_schedule", "none"),
        temperature_anneal_min=config.get("temperature_anneal_min", 0.1),
        temperature_anneal_epochs=config.get("temperature_anneal_epochs", 0),
        discrete_reinit_epoch=config.get("discrete_reinit_epoch", 0),
        discrete_reinit_snap=config.get("discrete_reinit_snap", "argmax"),
        logit_decay=config.get("logit_decay", 0.0),
        val_eval_every=config.get("val_eval_every", 100),
        val_eval_mode=config.get("val_eval_mode", "discrete"),
        per_epoch_callback=_test_eval_callback if test_eval_every > 0 else None,
        use_chat_template=config.get("use_chat_template", False),
        accumulate_embeds=config.get("efficient_generate", False),
        kwargs=lc_kwargs,
    )

    # ---- final test evaluation ----
    final_fl = result["final_p_logits"]

    logger.info("Running final discrete test evaluation...")
    test_metrics_discrete = evaluate_shared_prompt(
        free_logits=final_fl,
        eval_pairs=test_pairs_eval,
        model=model,
        diff_model=diff_model,
        stgs_module=stgs_for_eval,
        embedding_weights=embedding_weights,
        tokenizer=tokenizer,
        device=device,
        E_input_ids=E_input_ids,
        E_embeds=E_embeds,
        extraction_fns=extraction_fns,
        max_new_tokens_reasoning=config.get("max_new_tokens_reasoning", 200),
        max_new_tokens_answer=config.get("max_new_tokens_answer", 20),
        eval_mode="discrete",
        use_chat_template=config.get("use_chat_template", False),
        accumulate_embeds=_accumulate_embeds,
    )

    test_metrics_soft = {}
    if config.get("val_eval_mode", "discrete") == "soft" or config.get("run_soft_test_eval", False):
        logger.info("Running final soft test evaluation...")
        test_metrics_soft = evaluate_shared_prompt(
            free_logits=final_fl,
            eval_pairs=test_pairs_eval,
            model=model,
            diff_model=diff_model,
            stgs_module=stgs_for_eval,
            embedding_weights=embedding_weights,
            tokenizer=tokenizer,
            device=device,
            E_input_ids=E_input_ids,
            E_embeds=E_embeds,
            extraction_fns=extraction_fns,
            max_new_tokens_reasoning=config.get("max_new_tokens_reasoning", 200),
            max_new_tokens_answer=config.get("max_new_tokens_answer", 20),
            eval_mode="soft",
            use_chat_template=config.get("use_chat_template", False),
            accumulate_embeds=_accumulate_embeds,
        )

    # Summarize to W&B
    final_p_text = result["final_p_text"]
    final_p_tokens = result["final_p_tokens"]
    val_acc_history = result["val_accuracy_history"]
    best_val_any = 0.0
    best_val_epoch = 0
    for entry in val_acc_history:
        if entry.get("any_correct", 0.0) > best_val_any:
            best_val_any = entry["any_correct"]
            best_val_epoch = entry.get("epoch", 0)

    final_test_any = test_metrics_discrete.get("any_correct", 0.0)
    learnable_logits = result.get("learnable_logits", final_fl)
    output_dir = Path(config.get("output_dir", "pc_optimization_results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    learnable_logits_path = output_dir / f"{run_name}_learnable_logits.pt"
    torch.save(learnable_logits, learnable_logits_path)

    wandb.log({
        "test/p_text": final_p_text,
        "test/p_token_ids": final_p_tokens,
        **{f"test/discrete/{k}": v for k, v in test_metrics_discrete.items()},
        **{f"test/soft/{k}": v for k, v in test_metrics_soft.items()},
    })
    artifact = wandb.Artifact(
        name=f"learnable-logits-{run_name}",
        type="tensor",
        description="Final learnable logits at the end of PC optimization",
    )
    artifact.add_file(str(learnable_logits_path))
    wandb.run.log_artifact(artifact)
    wandb.run.summary.update({
        "final_val_accuracy": val_acc_history[-1].get("any_correct", 0.0) if val_acc_history else 0.0,
        "final_test_accuracy": final_test_any,
        "best_val_accuracy": best_val_any,
        "best_val_epoch": best_val_epoch,
        "final_p_text": final_p_text,
    })
    wandb.finish()

    # ---- save JSON result ----
    out_path = output_dir / f"{run_name}.json"

    summary = {
        "run_name": run_name,
        "config": config,
        "final_p_text": final_p_text,
        "final_p_token_ids": final_p_tokens,
        "loss_history": result["loss_history"],
        "val_accuracy_history": val_acc_history,
        "test_accuracy_discrete": test_metrics_discrete,
        "test_accuracy_soft": test_metrics_soft,
        "learnable_logits_path": str(learnable_logits_path),
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Results saved to {out_path}")

    return summary


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Partially-conditioned multi-target prompt inversion."
    )

    # Dataset (mutually exclusive)
    ds_group = p.add_mutually_exclusive_group()
    ds_group.add_argument("--dataset_path", type=str, default=None,
                          help="Path to PC-format JSON dataset.")
    ds_group.add_argument("--hf_dataset", type=str, default=None,
                          help="HuggingFace dataset name (e.g. openai/gsm8k).")
    p.add_argument("--hf_dataset_subset", type=str, default="main")

    # Split sizes and seeds
    p.add_argument("--train_size", type=int, default=50)
    p.add_argument("--val_size", type=int, default=20)
    p.add_argument("--test_size", type=int, default=100)
    p.add_argument("--train_seed", type=int, default=42)
    p.add_argument("--val_seed", type=int, default=43)
    p.add_argument("--test_seed", type=int, default=44)
    p.add_argument("--opt_seed", type=int, default=0)

    # Extraction / evaluation
    p.add_argument("--extraction_prompt", type=str,
                   default="Therefore, the final answer is $")
    p.add_argument("--max_new_tokens_reasoning", type=int, default=200)
    p.add_argument("--max_new_tokens_answer", type=int, default=20)
    p.add_argument("--few_shot_n", type=int, default=0)
    p.add_argument("--few_shot_seed", type=int, default=0)
    p.add_argument("--val_eval_mode", type=str, default="discrete",
                   choices=["discrete", "soft"])

    # Optimization
    p.add_argument("--losses", type=str, default="crossentropy")
    p.add_argument("--seq_len", type=int, default=20)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--learning_rate", type=float, default=0.01)
    p.add_argument("--inner_batch_size", type=int, default=4)
    p.add_argument("--teacher_forcing_r", type=str2bool, default=True)
    p.add_argument("--bptt", type=str2bool, default=False)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--stgs_hard", type=str2bool, default=True)
    p.add_argument("--stgs_hard_method", type=str, default="categorical")
    p.add_argument("--logits_normalize", type=str, default="none",
                   choices=["none", "center", "zscore"])
    p.add_argument("--stgs_input_dropout", type=float, default=0.0)
    p.add_argument("--stgs_output_dropout", type=float, default=0.0)
    p.add_argument("--eps", type=float, default=1e-10)
    p.add_argument("--logits_top_k", type=int, default=0)
    p.add_argument("--logits_top_p", type=float, default=1.0)
    p.add_argument("--logits_filter_penalty", type=float, default=1e4)
    p.add_argument("--init_strategy", type=str, default="randn")
    p.add_argument("--init_std", type=float, default=0.0)
    p.add_argument("--logits_lora_rank", type=int, default=0)
    p.add_argument("--max_gradient_norm", type=float, default=0.0)
    p.add_argument("--temperature_anneal_schedule", type=str, default="none",
                   choices=["none", "linear", "cosine"])
    p.add_argument("--temperature_anneal_min", type=float, default=0.1)
    p.add_argument("--temperature_anneal_epochs", type=int, default=0)
    p.add_argument("--discrete_reinit_epoch", type=int, default=0)
    p.add_argument("--discrete_reinit_snap", type=str, default="argmax")
    p.add_argument("--gumbel_noise_scale", type=float, default=1.0)
    p.add_argument("--adaptive_gumbel_noise", type=str2bool, default=False)
    p.add_argument("--adaptive_gumbel_noise_beta", type=float, default=0.9)
    p.add_argument("--adaptive_gumbel_noise_min_scale", type=float, default=0.0)
    p.add_argument("--logit_decay", type=float, default=0.0)

    # STGSDiffModel config (for R/Y generation)
    p.add_argument("--diff_model_temperature", type=float, default=1.0)
    p.add_argument("--diff_model_hard", type=str2bool, default=True)
    p.add_argument("--diff_model_use_soft_embeds", type=str2bool, default=True)

    # Validation / test eval
    p.add_argument("--val_eval_every", type=int, default=100)
    p.add_argument("--test_eval_every", type=int, default=0)
    p.add_argument("--test_eval_size", type=int, default=0,
                   help="0 = same as test_size")

    # LossClass passthrough
    p.add_argument("--promptLambda", type=float, default=0.0)
    p.add_argument("--complLambda", type=float, default=0.0)
    p.add_argument("--promptTfComplLambda", type=float, default=0.0)
    p.add_argument("--promptDistEntropyLambda", type=float, default=0.0)
    p.add_argument("--commitmentLambda", type=float, default=0.0)
    p.add_argument("--commitment_similarity", type=str, default="argmax")
    p.add_argument("--eos_reg_lambda", type=float, default=0.0)
    p.add_argument("--eos_reg_schedule", type=str, default="linear")
    p.add_argument("--eos_reg_alpha", type=float, default=1.0)
    p.add_argument("--loss_pos_weight_schedule", type=str, default="uniform")
    p.add_argument("--loss_pos_weight_step", type=float, default=1.0)
    p.add_argument("--loss_pos_weight_base", type=float, default=2.0)
    p.add_argument("--commitment_pos_weight_schedule", type=str, default="uniform")
    p.add_argument("--commitment_pos_weight_step", type=float, default=10.0)
    p.add_argument("--commitment_pos_weight_base", type=float, default=2.0)

    # Input formatting
    p.add_argument("--use_chat_template", type=str2bool, default=False,
                   help="Apply tokenizer chat template to x before concatenation: [chat(x), p]")
    p.add_argument("--efficient_generate", type=str2bool, default=False,
                   help="Accumulate soft embeddings instead of vocab-size one-hots in generate() "
                        "(~40x less peak GPU memory). Set False only to debug or inspect one-hots.")

    # Model
    p.add_argument("--model_name", type=str,
                   default="HuggingFaceTB/SmolLM-135M")
    p.add_argument("--model_precision", type=str, default="full",
                   choices=["full", "half"])
    p.add_argument("--device", type=str, default="auto",
                   help="Device override; 'auto' uses device_map=auto")

    # Output
    p.add_argument("--output_dir", type=str, default="pc_optimization_results")
    p.add_argument("--run_name", type=str, default="pc_opt")

    # W&B
    p.add_argument("--wandb_project", type=str, default="sdlm-pc-inversion")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_tags", type=str, default="",
                   help="Comma-separated tags")
    p.add_argument("--weave_project", type=str, default=None,
                   help="W&B Weave project name (defaults to --wandb_project if not set).")

    return p.parse_args()


def main():
    args = parse_args()
    config = vars(args)

    # Resolve dataset source
    if config["dataset_path"] is None and config["hf_dataset"] is None:
        # Default to HF GSM8K
        config["hf_dataset"] = "openai/gsm8k"

    # test_eval_size defaults to test_size
    if config.get("test_eval_size", 0) <= 0:
        config["test_eval_size"] = config.get("test_size", 100)

    logger.info(f"Config: {config}")
    batch_pc_optimize(config)


if __name__ == "__main__":
    main()
