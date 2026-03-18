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
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import wandb

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(1, str(Path(__file__).parent.parent.parent))
sys.path.insert(2, str(Path(__file__).parent.parent / "tinylora"))

from batch_optimize_main import setup_model_and_tokenizer
from sdlm import STGS
from sdlm.stgs_diff_model import STGSDiffModel
from sdlm.utils.tqdm_utils import tqdm
from pc_main import (
    ensure_pc_main_compatibility,
    pc_optimize_inputs,
    evaluate_shared_prompt,
    evaluate_shared_prompt_batched,
)
from pc_weave_logging import init_weave
from product_of_speaker import build_pc_product_of_speaker_transform

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


def collect_explicit_arg_names(argv: List[str]) -> Set[str]:
    explicit: Set[str] = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        name = token[2:].split("=", 1)[0].replace("-", "_")
        explicit.add(name)
    return explicit


def add_loss_component(losses: str, enabled: bool, component: str) -> str:
    if enabled and component not in losses.split("+"):
        return f"{losses}+{component}" if losses else component
    return losses


REASONING_GENERATE_ARG_MAP = {
    "reasoning_generate_do_sample": "do_sample",
    "reasoning_generate_temperature": "temperature",
    "reasoning_generate_top_p": "top_p",
    "reasoning_generate_top_k": "top_k",
    "reasoning_generate_min_p": "min_p",
    "reasoning_generate_typical_p": "typical_p",
    "reasoning_generate_num_beams": "num_beams",
    "reasoning_generate_num_return_sequences": "num_return_sequences",
    "reasoning_generate_repetition_penalty": "repetition_penalty",
    "reasoning_generate_length_penalty": "length_penalty",
    "reasoning_generate_no_repeat_ngram_size": "no_repeat_ngram_size",
    "reasoning_generate_early_stopping": "early_stopping",
    "reasoning_generate_pad_token_id": "pad_token_id",
    "reasoning_generate_eos_token_id": "eos_token_id",
}


def collect_reasoning_generate_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    for config_key, generate_key in REASONING_GENERATE_ARG_MAP.items():
        value = config.get(config_key)
        if value is not None:
            kwargs[generate_key] = value
    return kwargs


def _split_metric_columns(extraction_fns: Dict[str, Any]) -> List[str]:
    return [
        "split",
        "eval_mode",
        "n_examples",
        "any_correct",
        "all_correct",
        *[f"accuracy_{method_name}" for method_name in extraction_fns],
    ]


def _build_split_metrics_table(
    split_metrics: Dict[str, Tuple[str, int, Dict[str, Any]]],
    extraction_fns: Dict[str, Any],
):
    table = wandb.Table(columns=_split_metric_columns(extraction_fns))
    for split_name, (eval_mode, n_examples, metrics) in split_metrics.items():
        row = [
            split_name,
            eval_mode,
            n_examples,
            metrics.get("any_correct"),
            metrics.get("all_correct"),
        ]
        for method_name in extraction_fns:
            row.append(metrics.get(f"accuracy_{method_name}"))
        table.add_data(*row)
    return table


def _build_prediction_table(
    split_name: str,
    examples: List[Dict[str, Any]],
    extraction_fns: Dict[str, Any],
):
    columns = [
        "split",
        "sample_index",
        "epoch",
        "eval_mode",
        "reasoning_generation_backend",
        "input_text",
        "target_answer",
        "prompt_text",
        "generated_reasoning",
        "generated_answer",
        "any_correct",
        "all_correct",
    ]
    columns.extend(f"correct_{method_name}" for method_name in extraction_fns)
    table = wandb.Table(columns=columns)
    for example in examples:
        row = [
            split_name,
            example.get("sample_index"),
            example.get("epoch"),
            example.get("eval_mode"),
            example.get("reasoning_generation_backend"),
            example.get("input_text"),
            example.get("target_answer"),
            example.get("prompt_text"),
            example.get("generated_reasoning"),
            example.get("generated_answer"),
            example.get("any_correct"),
            example.get("all_correct"),
        ]
        for method_name in extraction_fns:
            row.append(example.get(f"correct_{method_name}"))
        table.add_data(*row)
    return table


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
    pipeline_pbar = tqdm(
        total=8,
        desc="PC pipeline: model",
        leave=True,
    )

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
            "temperature": config.get("bptt_temperature", config.get("diff_model_temperature", 1.0)),
            "learnable_temperature": config.get("bptt_learnable_temperature", False),
            "hidden_state_conditioning": config.get("bptt_hidden_state_conditioning", False),
            "dropout": 0.0,
        },
        stgs_logits_generation=True,
    )
    pipeline_pbar.update(1)

    # ---- dataset ----
    pipeline_pbar.set_description("PC pipeline: dataset")
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
    if not val_pairs:
        pipeline_pbar.total = max(pipeline_pbar.total - 1, 1)
        pipeline_pbar.refresh()
    pipeline_pbar.set_postfix(
        train=len(train_triples),
        val=len(val_pairs),
        test=len(test_pairs),
        refresh=False,
    )
    pipeline_pbar.update(1)

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
    wandb.config.update(
        {
            "dataset_metadata": dataset.get("metadata", {}),
            "train_examples": len(train_triples),
            "val_examples": len(val_pairs),
            "test_examples": len(test_pairs),
            "extraction_prompt": extraction_prompt,
            "reasoning_generate_kwargs": config.get("reasoning_generate_kwargs", {}),
        },
        allow_val_change=True,
    )
    wandb.log(
        {
            "dataset/train_examples": len(train_triples),
            "dataset/val_examples": len(val_pairs),
            "dataset/test_examples": len(test_pairs),
        }
    )
    init_weave(
        project=config.get("weave_project") or config.get("wandb_project", "sdlm-pc-inversion"),
        entity=config.get("wandb_entity", None),
        call_link_log_path=Path(config["output_dir"]) / "weave_call_links.log",
    )

    # ---- test eval callback (full test epoch during training) ----
    test_eval_every = config.get("test_eval_every", 0)
    test_eval_size = config.get("test_eval_size", len(test_pairs))
    test_pairs_eval = test_pairs[:test_eval_size] if test_eval_size < len(test_pairs) else test_pairs

    embedding_layer = model.get_input_embeddings()
    embedding_weights = embedding_layer.weight.detach()
    allowed_tokens = torch.arange(model.config.vocab_size, device=device)
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
        stgs_hard_embsim_probs=config.get("stgs_hard_embsim_probs", "gumbel_soft"),
        stgs_hard_embsim_strategy=config.get("stgs_hard_embsim_strategy", "nearest"),
        stgs_hard_embsim_top_k=config.get("stgs_hard_embsim_top_k", 8),
        stgs_hard_embsim_rerank_alpha=config.get("stgs_hard_embsim_rerank_alpha", 0.5),
        stgs_hard_embsim_sample_tau=config.get("stgs_hard_embsim_sample_tau", 1.0),
        stgs_hard_embsim_margin=config.get("stgs_hard_embsim_margin", 0.0),
        stgs_hard_embsim_fallback=config.get("stgs_hard_embsim_fallback", "argmax"),
        logits_normalize=config.get("logits_normalize", "none"),
        eps=config.get("eps", 1e-10),
        device=device,
    )

    _accumulate_embeds = config.get("efficient_generate", False)
    prompt_logits_transform = None
    if config.get("assemble_strategy", "identity") == "product_of_speaker":
        prompt_logits_transform = build_pc_product_of_speaker_transform(
            model=model,
            tokenizer=tokenizer,
            allowed_tokens=allowed_tokens,
            seq_len=config.get("seq_len", 20),
            base_model_name=config.get("model_name"),
            pos_lm_name=config.get("pos_lm_name"),
            pos_prompt=config.get("pos_prompt"),
            pos_prompt_template=config.get("pos_prompt_template"),
            pos_pair_template=config.get("pos_pair_template"),
            pos_pair_separator=config.get("pos_pair_separator", "\n\n"),
            pos_use_chat_template=config.get("pos_use_chat_template", False),
            pos_lm_temperature=config.get("pos_lm_temperature", 1.0),
            pos_lm_top_p=config.get("pos_lm_top_p", 1.0),
            pos_lm_top_k=config.get("pos_lm_top_k", 0),
            logits_filter_penalty=config.get("logits_filter_penalty", 1e4),
        )

    def _test_eval_callback(epoch: int, fl: torch.Tensor) -> Dict:
        if test_eval_every <= 0 or ((epoch + 1) % test_eval_every != 0):
            return {}
        _eval_fn = evaluate_shared_prompt_batched if config.get("use_batched_eval", False) else evaluate_shared_prompt
        metrics = _eval_fn(
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
            eval_mode=config.get("test_prompt_eval_mode", "discrete"),
            reasoning_generation_backend=config.get("reasoning_generation_backend", "diff"),
            reasoning_generate_kwargs=config.get("reasoning_generate_kwargs", {}),
            use_chat_template=config.get("use_chat_template", False),
            epoch=epoch,
            eval_split="test",
            accumulate_embeds=_accumulate_embeds,
            prompt_logits_transform=prompt_logits_transform,
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
    pipeline_pbar.set_description("PC pipeline: setup")
    pipeline_pbar.update(1)

    # ---- set opt seed ----
    opt_seed = config.get("opt_seed", 0)
    torch.manual_seed(opt_seed)
    random.seed(opt_seed)

    # ---- run optimization ----
    pipeline_pbar.set_description("PC pipeline: optimize")
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
        reasoning_generation_backend=config.get("reasoning_generation_backend", "diff"),
        reasoning_generate_kwargs=config.get("reasoning_generate_kwargs", {}),
        losses=config.get("losses", "crossentropy"),
        seq_len=config.get("seq_len", 20),
        epochs=config.get("epochs", 2000),
        learning_rate=config.get("learning_rate", 0.01),
        logits_lora_b_learning_rate=config.get("logits_lora_b_learning_rate"),
        inner_batch_size=config.get("inner_batch_size", 4),
        batch_size=config.get("batch_size"),
        temperature=config.get("temperature", 1.0),
        learnable_temperature=config.get("learnable_temperature", False),
        decouple_learnable_temperature=config.get("decouple_learnable_temperature", False),
        stgs_hard=config.get("stgs_hard", True),
        stgs_hard_method=config.get("stgs_hard_method", "categorical"),
        stgs_hard_embsim_probs=config.get("stgs_hard_embsim_probs", "gumbel_soft"),
        stgs_hard_embsim_strategy=config.get("stgs_hard_embsim_strategy", "nearest"),
        stgs_hard_embsim_top_k=config.get("stgs_hard_embsim_top_k", 8),
        stgs_hard_embsim_rerank_alpha=config.get("stgs_hard_embsim_rerank_alpha", 0.5),
        stgs_hard_embsim_sample_tau=config.get("stgs_hard_embsim_sample_tau", 1.0),
        stgs_hard_embsim_margin=config.get("stgs_hard_embsim_margin", 0.0),
        stgs_hard_embsim_fallback=config.get("stgs_hard_embsim_fallback", "argmax"),
        bptt_temperature=config.get("bptt_temperature", config.get("diff_model_temperature", 1.0)),
        bptt_learnable_temperature=config.get("bptt_learnable_temperature", False),
        bptt_decouple_learnable_temperature=config.get("bptt_decouple_learnable_temperature", False),
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
        logits_normalize=config.get("logits_normalize", "none"),
        bptt_logits_normalize=config.get("bptt_logits_normalize", "none"),
        stgs_input_dropout=config.get("stgs_input_dropout", 0.0),
        stgs_output_dropout=config.get("stgs_output_dropout", 0.0),
        eps=config.get("eps", 1e-10),
        bptt_eps=config.get("bptt_eps", 1e-10),
        gumbel_noise_scale=config.get("gumbel_noise_scale", 1.0),
        adaptive_gumbel_noise=config.get("adaptive_gumbel_noise", False),
        adaptive_gumbel_noise_beta=config.get("adaptive_gumbel_noise_beta", 0.9),
        adaptive_gumbel_noise_min_scale=config.get("adaptive_gumbel_noise_min_scale", 0.0),
        logits_top_k=config.get("logits_top_k", 0),
        logits_top_p=config.get("logits_top_p", 1.0),
        logits_filter_penalty=config.get("logits_filter_penalty", 1e4),
        init_strategy=config.get("init_strategy", "randn"),
        init_std=config.get("init_std", 0.0),
        init_mlm_model=config.get("init_mlm_model", "distilbert-base-uncased"),
        init_mlm_top_k=config.get("init_mlm_top_k", 50),
        logits_lora_rank=config.get("logits_lora_rank", 0),
        max_gradient_norm=config.get("max_gradient_norm", 0.0),
        temperature_anneal_schedule=config.get("temperature_anneal_schedule", "none"),
        temperature_anneal_min=config.get("temperature_anneal_min", 0.1),
        temperature_anneal_epochs=config.get("temperature_anneal_epochs", 0),
        temperature_anneal_reg_lambda=config.get(
            "temperature_anneal_reg_lambda",
            config.get("temperatureAnnealRegLambda", 0.0),
        ),
        temperature_anneal_reg_mode=config.get(
            "temperature_anneal_reg_mode",
            config.get("temperatureAnnealRegMode", "mse"),
        ),
        temperature_loss_coupling_lambda=config.get(
            "temperature_loss_coupling_lambda",
            config.get("temperatureLossCouplingLambda", 0.0),
        ),
        discrete_reinit_epoch=config.get("discrete_reinit_epoch", 0),
        discrete_reinit_snap=config.get("discrete_reinit_snap", "argmax"),
        discrete_reinit_prob=config.get("discrete_reinit_prob", 1.0),
        discrete_reinit_topk=config.get("discrete_reinit_topk", 0),
        discrete_reinit_entropy_threshold=config.get("discrete_reinit_entropy_threshold", 0.0),
        discrete_reinit_embsim_probs=config.get("discrete_reinit_embsim_probs", "input_logits"),
        logit_decay=config.get("logit_decay", 0.0),
        prompt_length_learnable=config.get("prompt_length_learnable", False),
        prompt_length_alpha_init=config.get("prompt_length_alpha_init", 0.0),
        prompt_length_beta=config.get("prompt_length_beta", 5.0),
        prompt_length_reg_lambda=config.get("prompt_length_reg_lambda", 0.0),
        prompt_length_eos_spike=config.get("prompt_length_eos_spike", 10.0),
        prompt_length_mask_eos_attention=config.get("prompt_length_mask_eos_attention", False),
        ppo_kl_lambda=config.get("ppo_kl_lambda", 0.0),
        ppo_kl_mode=config.get("ppo_kl_mode", "soft"),
        ppo_kl_epsilon=config.get("ppo_kl_epsilon", 0.0),
        ppo_ref_update_period=config.get("ppo_ref_update_period", 10),
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
        superposition_output_dir=config.get(
            "superposition_output_dir",
            str(Path(config.get("output_dir", "pc_optimization_results")) / "superposition"),
        ),
        fixed_gt_prefix_n=config.get("fixed_gt_prefix_n", 0),
        fixed_gt_suffix_n=config.get("fixed_gt_suffix_n", 0),
        fixed_gt_prefix_rank2_n=config.get("fixed_gt_prefix_rank2_n", 0),
        fixed_gt_suffix_rank2_n=config.get("fixed_gt_suffix_rank2_n", 0),
        fixed_prefix_text=config.get("fixed_prefix_text"),
        fixed_suffix_text=config.get("fixed_suffix_text"),
        early_stop_loss_threshold=config.get("early_stop_loss_threshold", 0.0),
        val_eval_every=config.get("val_eval_every", 100),
        val_prompt_eval_mode=config.get("val_prompt_eval_mode", "discrete"),
        per_epoch_callback=_test_eval_callback if test_eval_every > 0 else None,
        use_batched_forward_pass=config.get("use_batched_forward_pass", False),
        batched_stgs_noise_mode=config.get("batched_stgs_noise_mode", "shared"),
        use_batched_eval=config.get("use_batched_eval", False),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        use_chat_template=config.get("use_chat_template", False),
        accumulate_embeds=config.get("efficient_generate", False),
        prompt_logits_transform=prompt_logits_transform,
        kwargs=lc_kwargs,
    )
    pipeline_pbar.update(1)

    # ---- final validation evaluation ----
    final_fl = result["final_p_logits"]
    final_val_metrics = None
    final_val_examples: List[Dict[str, Any]] = []
    if val_pairs:
        val_prompt_eval_mode = config.get("val_prompt_eval_mode", "discrete")
        pipeline_pbar.set_description(f"PC pipeline: val eval ({val_prompt_eval_mode})")
        logger.info("Running final validation evaluation with prompt eval mode=%s...", val_prompt_eval_mode)
        final_val_metrics, final_val_examples = evaluate_shared_prompt(
            free_logits=final_fl,
            eval_pairs=val_pairs,
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
            eval_mode=val_prompt_eval_mode,
            reasoning_generation_backend=config.get("reasoning_generation_backend", "diff"),
            reasoning_generate_kwargs=config.get("reasoning_generate_kwargs", {}),
            use_chat_template=config.get("use_chat_template", False),
            eval_split="val",
            accumulate_embeds=_accumulate_embeds,
            prompt_logits_transform=prompt_logits_transform,
            return_examples=True,
        )
        pipeline_pbar.update(1)

    # ---- final test evaluation ----
    test_prompt_eval_mode = config.get("test_prompt_eval_mode", "discrete")

    pipeline_pbar.set_description(f"PC pipeline: test eval ({test_prompt_eval_mode})")
    logger.info("Running final test evaluation with prompt eval mode=%s...", test_prompt_eval_mode)
    test_metrics, test_examples = evaluate_shared_prompt(
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
        eval_mode=test_prompt_eval_mode,
        reasoning_generation_backend=config.get("reasoning_generation_backend", "diff"),
        reasoning_generate_kwargs=config.get("reasoning_generate_kwargs", {}),
        use_chat_template=config.get("use_chat_template", False),
        eval_split="test",
        accumulate_embeds=_accumulate_embeds,
        prompt_logits_transform=prompt_logits_transform,
        return_examples=True,
    )
    pipeline_pbar.update(1)

    # Summarize to W&B
    pipeline_pbar.set_description("PC pipeline: save")
    final_p_text = result["final_p_text"]
    final_p_tokens = result["final_p_tokens"]
    val_acc_history = result["val_accuracy_history"]
    train_acc_history = result.get("train_accuracy_history", [])
    best_val_any = 0.0
    best_val_epoch = 0
    for entry in val_acc_history:
        if entry.get("any_correct", 0.0) > best_val_any:
            best_val_any = entry["any_correct"]
            best_val_epoch = entry.get("epoch", 0)

    final_loss = result["loss_history"][-1] if result.get("loss_history") else None
    last_train_metrics = train_acc_history[-1] if train_acc_history else {}
    final_train_any = last_train_metrics.get("any_correct", 0.0)
    final_train_all = last_train_metrics.get("all_correct", 0.0)
    final_val_any = final_val_metrics.get("any_correct", 0.0) if final_val_metrics else 0.0
    final_test_any = test_metrics.get("any_correct", 0.0)
    learnable_logits = result.get("learnable_logits", final_fl)
    learnable_temperatures = result.get("learnable_temperatures", {})
    output_dir = Path(config.get("output_dir", "pc_optimization_results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    learnable_logits_path = output_dir / f"{run_name}_learnable_logits.pt"
    learnable_temperatures_path = output_dir / f"{run_name}_learnable_temperatures.pt"
    torch.save(learnable_logits, learnable_logits_path)
    torch.save(learnable_temperatures, learnable_temperatures_path)

    final_log_dict = {
        "final/p_text": final_p_text,
        "final/p_token_ids": final_p_tokens,
        "final/loss": final_loss,
        "train_final/prompt_eval_mode": "teacher_forcing_r" if config.get("teacher_forcing_r", True) else "generated_reasoning",
        "train_final/p_text": final_p_text,
        "train_final/p_token_ids": final_p_tokens,
        "train_final/any_correct": final_train_any,
        "train_final/all_correct": final_train_all,
        "test/p_text": final_p_text,
        "test/p_token_ids": final_p_tokens,
        "test/prompt_eval_mode": test_prompt_eval_mode,
        **{f"test/{k}": v for k, v in test_metrics.items()},
    }
    if final_val_metrics is not None:
        final_log_dict.update({
            "val/p_text": final_p_text,
            "val/p_token_ids": final_p_tokens,
            "val/prompt_eval_mode": config.get("val_prompt_eval_mode", "discrete"),
            **{f"val/{k}": v for k, v in final_val_metrics.items()},
        })
    wandb.log(final_log_dict)

    split_metrics = {
        "train_final": ("teacher_forcing_r" if config.get("teacher_forcing_r", True) else "generated_reasoning", len(train_triples), {
            "any_correct": final_train_any,
            "all_correct": final_train_all,
            **{
                f"accuracy_{method_name}": last_train_metrics.get(f"accuracy_{method_name}")
                for method_name in extraction_fns
            },
        }),
        "test": (test_prompt_eval_mode, len(test_pairs_eval), test_metrics),
    }
    if final_val_metrics is not None:
        split_metrics["val"] = (
            config.get("val_prompt_eval_mode", "discrete"),
            len(val_pairs),
            final_val_metrics,
        )
    wandb.log({"split_metrics": _build_split_metrics_table(split_metrics, extraction_fns)})
    if final_val_examples:
        wandb.log({"val_predictions": _build_prediction_table("val", final_val_examples, extraction_fns)})
    if test_examples:
        wandb.log({"test_predictions": _build_prediction_table("test", test_examples, extraction_fns)})
    artifact = wandb.Artifact(
        name=f"learnable-logits-{run_name}",
        type="tensor",
        description="Final learnable logits at the end of PC optimization",
    )
    artifact.add_file(str(learnable_logits_path))
    wandb.run.log_artifact(artifact)
    if learnable_temperatures:
        temps_artifact = wandb.Artifact(
            name=f"learnable-temperatures-{run_name}",
            type="tensor",
            description="Final learnable temperatures at the end of PC optimization",
        )
        temps_artifact.add_file(str(learnable_temperatures_path))
        wandb.run.log_artifact(temps_artifact)
    wandb.run.summary.update({
        "epochs_completed": len(result.get("loss_history", [])),
        "final_loss": final_loss,
        "final_train_accuracy": final_train_any,
        "final_train_all_correct": final_train_all,
        "final_val_accuracy": final_val_any,
        "final_test_accuracy": final_test_any,
        "best_val_accuracy": best_val_any,
        "best_val_epoch": best_val_epoch,
        "final_p_text": final_p_text,
        "final_p_token_ids": final_p_tokens,
    })

    # ---- save JSON result ----
    out_path = output_dir / f"{run_name}.json"

    summary = {
        "run_name": run_name,
        "config": config,
        "final_p_text": final_p_text,
        "final_p_token_ids": final_p_tokens,
        "loss_history": result["loss_history"],
        "train_accuracy_history": train_acc_history,
        "val_accuracy_history": val_acc_history,
        "final_val_metrics": final_val_metrics,
        "final_val_examples": final_val_examples,
        "test_prompt_eval_mode": test_prompt_eval_mode,
        "test_accuracy": test_metrics,
        "test_examples": test_examples,
        "learnable_logits_path": str(learnable_logits_path),
        "learnable_temperatures_path": str(learnable_temperatures_path),
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Results saved to {out_path}")
    results_artifact = wandb.Artifact(
        name=f"{run_name}-summary",
        type="results",
        description="PC batch optimization summary JSON",
    )
    results_artifact.add_file(str(out_path))
    wandb.run.log_artifact(results_artifact)
    wandb.finish()
    pipeline_pbar.update(1)
    pipeline_pbar.close()

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
    p.add_argument("--seed", type=int, default=None,
                   help="Alias for --opt_seed for single-target main.py compatibility.")

    # Extraction / evaluation
    p.add_argument("--extraction_prompt", type=str,
                   default="Therefore, the final answer is $")
    p.add_argument("--max_new_tokens_reasoning", type=int, default=200)
    p.add_argument("--max_new_tokens_answer", type=int, default=20)
    p.add_argument("--few_shot_n", type=int, default=0)
    p.add_argument("--few_shot_seed", type=int, default=0)
    p.add_argument("--val_prompt_eval_mode", type=str, default="discrete",
                   choices=["discrete", "soft"])
    p.add_argument("--test_prompt_eval_mode", type=str, default="discrete",
                   choices=["discrete", "soft"])

    # Optimization
    p.add_argument("--losses", type=str, default="crossentropy")
    p.add_argument("--seq_len", type=int, default=20)
    p.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="Number of full passes over the training split.",
    )
    p.add_argument("--learning_rate", type=float, default=0.01)
    p.add_argument(
        "--logits_lora_b_learning_rate",
        type=float,
        default=None,
        help="Optional learning rate override for LoRA prompt-logit matrix B. "
             "Ignored when logits_lora_rank <= 0.",
    )
    p.add_argument("--inner_batch_size", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=None,
                   help="Alias for the shared-prompt inner batch size.")
    p.add_argument("--teacher_forcing_r", type=str2bool, default=True)
    p.add_argument("--bptt", type=str2bool, default=False)
    p.add_argument(
        "--reasoning_generation_backend",
        type=str,
        default="diff",
        choices=["diff", "hf_generate"],
        help="Reasoning generation backend when --teacher_forcing_r=False. "
             "'diff' uses the existing wrapper rollout; 'hf_generate' delegates to model.generate(...).",
    )
    p.add_argument("--reasoning_generate_do_sample", type=str2bool, default=None)
    p.add_argument("--reasoning_generate_temperature", type=float, default=None)
    p.add_argument("--reasoning_generate_top_p", type=float, default=None)
    p.add_argument("--reasoning_generate_top_k", type=int, default=None)
    p.add_argument("--reasoning_generate_min_p", type=float, default=None)
    p.add_argument("--reasoning_generate_typical_p", type=float, default=None)
    p.add_argument("--reasoning_generate_num_beams", type=int, default=None)
    p.add_argument("--reasoning_generate_num_return_sequences", type=int, default=None)
    p.add_argument("--reasoning_generate_repetition_penalty", type=float, default=None)
    p.add_argument("--reasoning_generate_length_penalty", type=float, default=None)
    p.add_argument("--reasoning_generate_no_repeat_ngram_size", type=int, default=None)
    p.add_argument("--reasoning_generate_early_stopping", type=str2bool, default=None)
    p.add_argument("--reasoning_generate_pad_token_id", type=int, default=None)
    p.add_argument("--reasoning_generate_eos_token_id", type=int, default=None)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--learnable_temperature", type=str2bool, default=False)
    p.add_argument("--decouple_learnable_temperature", type=str2bool, default=False)
    p.add_argument("--stgs_hard", type=str2bool, default=True)
    p.add_argument("--stgs_hard_method", type=str, default="categorical",
                   choices=["categorical", "embsim-dot", "embsim-cos", "embsim-l2", "argmax"])
    p.add_argument("--stgs_hard_embsim_probs", type=str, default="gumbel_soft",
                   choices=["gumbel_soft", "input_logits"])
    p.add_argument("--stgs_hard_embsim_strategy", type=str, default="nearest",
                   choices=["nearest", "topk_rerank", "topk_sample", "margin_fallback", "lm_topk_restrict"])
    p.add_argument("--stgs_hard_embsim_top_k", type=int, default=8)
    p.add_argument("--stgs_hard_embsim_rerank_alpha", type=float, default=0.5)
    p.add_argument("--stgs_hard_embsim_sample_tau", type=float, default=1.0)
    p.add_argument("--stgs_hard_embsim_margin", type=float, default=0.0)
    p.add_argument("--stgs_hard_embsim_fallback", type=str, default="argmax",
                   choices=["argmax", "categorical"])
    p.add_argument("--logits_normalize", type=str, default="none",
                   choices=["none", "center", "zscore"])
    p.add_argument("--bptt_logits_normalize", type=str, default="none",
                   choices=["none", "center", "zscore"])
    p.add_argument("--stgs_input_dropout", type=float, default=0.0)
    p.add_argument("--stgs_output_dropout", type=float, default=0.0)
    p.add_argument("--eps", type=float, default=1e-10)
    p.add_argument("--bptt_eps", type=float, default=1e-10)
    p.add_argument("--logits_top_k", type=int, default=0)
    p.add_argument("--logits_top_p", type=float, default=1.0)
    p.add_argument("--logits_filter_penalty", type=float, default=1e4)
    p.add_argument("--assemble_strategy", type=str, default="identity",
                   choices=["identity", "product_of_speaker"])
    p.add_argument("--pos_lm_name", type=str, default=None,
                   help="Auxiliary LM used for product-of-speaker prompt assembly.")
    p.add_argument("--pos_prompt", type=str, default=None,
                   help="Literal PoS prompt text. Ignored when --pos_prompt_template is set.")
    p.add_argument("--pos_prompt_template", type=str, default=None,
                   help="PoS prompt template. PC mode supports the outer placeholder {pairs}.")
    p.add_argument("--pos_pair_template", type=str, default=None,
                   help="Per-pair PoS template. Supports {partial_conditioning_text} and {target}.")
    p.add_argument("--pos_pair_separator", type=str, default="\n\n",
                   help="Separator used when joining rendered PoS pairs for {pairs}.")
    p.add_argument("--pos_use_chat_template", type=str2bool, default=False,
                   help="Apply the auxiliary tokenizer chat template to the rendered PoS prompt.")
    p.add_argument("--pos_lm_temperature", type=float, default=1.0)
    p.add_argument("--pos_lm_top_p", type=float, default=1.0)
    p.add_argument("--pos_lm_top_k", type=int, default=0)
    p.add_argument("--init_strategy", type=str, default="randn")
    p.add_argument("--init_std", type=float, default=0.0)
    p.add_argument("--init_mlm_model", type=str, default="distilbert-base-uncased")
    p.add_argument("--init_mlm_top_k", type=int, default=50)
    p.add_argument("--logits_lora_rank", type=int, default=0)
    p.add_argument("--max_gradient_norm", type=float, default=0.0)
    p.add_argument("--temperature_anneal_schedule", type=str, default="none",
                   choices=["none", "linear", "cosine"])
    p.add_argument("--temperature_anneal_min", type=float, default=0.1)
    p.add_argument("--temperature_anneal_epochs", type=int, default=0)
    p.add_argument("--temperature_anneal_reg_lambda", "--temperatureAnnealRegLambda",
                   dest="temperature_anneal_reg_lambda", type=float, default=0.0)
    p.add_argument("--temperature_anneal_reg_mode", "--temperatureAnnealRegMode",
                   dest="temperature_anneal_reg_mode", type=str, default="mse",
                   choices=["mse", "one_sided"])
    p.add_argument("--temperature_loss_coupling_lambda", "--temperatureLossCouplingLambda",
                   dest="temperature_loss_coupling_lambda", type=float, default=0.0)
    p.add_argument("--discrete_reinit_epoch", type=int, default=0)
    p.add_argument("--discrete_reinit_snap", type=str, default="argmax")
    p.add_argument("--discrete_reinit_prob", type=float, default=1.0)
    p.add_argument("--discrete_reinit_topk", type=int, default=0)
    p.add_argument("--discrete_reinit_entropy_threshold", type=float, default=0.0)
    p.add_argument("--discrete_reinit_embsim_probs", type=str, default="input_logits",
                   choices=["input_logits", "gumbel_soft"])
    p.add_argument("--gumbel_noise_scale", type=float, default=1.0)
    p.add_argument("--adaptive_gumbel_noise", type=str2bool, default=False)
    p.add_argument("--adaptive_gumbel_noise_beta", type=float, default=0.9)
    p.add_argument("--adaptive_gumbel_noise_min_scale", type=float, default=0.0)
    p.add_argument("--logit_decay", type=float, default=0.0)
    p.add_argument("--prompt_length_learnable", type=str2bool, default=False)
    p.add_argument("--prompt_length_alpha_init", type=float, default=0.0)
    p.add_argument("--prompt_length_beta", type=float, default=5.0)
    p.add_argument("--prompt_length_reg_lambda", type=float, default=0.0)
    p.add_argument("--prompt_length_eos_spike", type=float, default=10.0)
    p.add_argument("--prompt_length_mask_eos_attention", type=str2bool, default=False)
    p.add_argument("--ppo_kl_lambda", type=float, default=0.0)
    p.add_argument("--ppo_kl_mode", type=str, default="soft", choices=["soft", "hinge"])
    p.add_argument("--ppo_kl_epsilon", type=float, default=0.0)
    p.add_argument("--ppo_ref_update_period", type=int, default=10)
    p.add_argument("--fixed_gt_prefix_n", type=int, default=0)
    p.add_argument("--fixed_gt_suffix_n", type=int, default=0)
    p.add_argument("--fixed_gt_prefix_rank2_n", type=int, default=0)
    p.add_argument("--fixed_gt_suffix_rank2_n", type=int, default=0)
    p.add_argument("--fixed_prefix_text", type=str, default=None)
    p.add_argument("--fixed_suffix_text", type=str, default=None)
    p.add_argument("--early_stop_loss_threshold", type=float, default=0.0)

    # STGSDiffModel config (for R/Y generation)
    p.add_argument("--diff_model_temperature", type=float, default=1.0)
    p.add_argument("--diff_model_hard", type=str2bool, default=True)
    p.add_argument("--diff_model_use_soft_embeds", type=str2bool, default=True)
    p.add_argument("--bptt_temperature", type=float, default=1.0)
    p.add_argument("--bptt_learnable_temperature", type=str2bool, default=False)
    p.add_argument("--bptt_decouple_learnable_temperature", type=str2bool, default=False)
    p.add_argument("--bptt_stgs_hard", type=str2bool, default=True)
    p.add_argument("--bptt_stgs_hard_method", type=str, default="categorical",
                   choices=["categorical", "embsim-dot", "embsim-cos", "embsim-l2", "argmax"])
    p.add_argument("--bptt_stgs_hard_embsim_probs", type=str, default="gumbel_soft",
                   choices=["gumbel_soft", "input_logits"])
    p.add_argument("--bptt_stgs_hard_embsim_strategy", type=str, default="nearest",
                   choices=["nearest", "topk_rerank", "topk_sample", "margin_fallback", "lm_topk_restrict"])
    p.add_argument("--bptt_stgs_hard_embsim_top_k", type=int, default=8)
    p.add_argument("--bptt_stgs_hard_embsim_rerank_alpha", type=float, default=0.5)
    p.add_argument("--bptt_stgs_hard_embsim_sample_tau", type=float, default=1.0)
    p.add_argument("--bptt_stgs_hard_embsim_margin", type=float, default=0.0)
    p.add_argument("--bptt_stgs_hard_embsim_fallback", type=str, default="argmax",
                   choices=["argmax", "categorical"])
    p.add_argument("--bptt_hidden_state_conditioning", type=str2bool, default=False)

    # Batched forward pass + gradient accumulation
    p.add_argument("--use_batched_forward_pass", type=str2bool, default=False,
                   help="Use batched GPU forward pass for training mini-batches.")
    p.add_argument("--batched_stgs_noise_mode", type=str, default="shared",
                   choices=["shared", "independent"],
                   help="Gumbel noise sharing within a batch: 'shared' (one draw) "
                        "or 'independent' (one draw per sample).")
    p.add_argument("--use_batched_eval", type=str2bool, default=False,
                   help="Use batched evaluate_shared_prompt for validation/test.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1,
                   help="Accumulate gradients over N mini-batches before optimizer.step(). "
                        "Effective batch size = inner_batch_size × gradient_accumulation_steps.")

    # Validation / test eval
    p.add_argument(
        "--val_eval_every",
        type=int,
        default=100,
        help="Run one full validation epoch every N completed train epochs (0 = disabled).",
    )
    p.add_argument(
        "--test_eval_every",
        type=int,
        default=0,
        help="Run one full scheduled test epoch every N completed train epochs (0 = disabled).",
    )
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

    # Parsed for main.py compatibility; rejected later when explicitly enabled.
    p.add_argument("--gradient_estimator", type=str, default="stgs", choices=["stgs", "reinforce"])
    p.add_argument("--reinforce_reward_scale", type=float, default=1.0)
    p.add_argument("--reinforce_use_baseline", type=str2bool, default=True)
    p.add_argument("--reinforce_baseline_beta", type=float, default=0.9)
    p.add_argument("--teacher_forcing", type=str2bool, default=False)
    p.add_argument("--bptt_teacher_forcing_via_diff_model", type=str2bool, default=False)
    p.add_argument("--plot_every", type=int, default=100000)
    p.add_argument("--stgs_grad_variance_samples", type=int, default=0)
    p.add_argument("--stgs_grad_variance_period", type=int, default=1)
    p.add_argument("--stgs_grad_bias_samples", type=int, default=0)
    p.add_argument("--stgs_grad_bias_period", type=int, default=1)
    p.add_argument("--stgs_grad_bias_reference_samples", type=int, default=0)
    p.add_argument("--stgs_grad_bias_reference_batch_size", type=int, default=0)
    p.add_argument("--stgs_grad_bias_reference_use_baseline", type=str2bool, default=True)
    p.add_argument("--stgs_grad_bias_reference_reward_scale", type=float, default=1.0)
    p.add_argument("--stgs_grad_bias_reference_baseline_beta", type=float, default=0.9)
    p.add_argument("--reinforce_grad_variance_samples", type=int, default=0)
    p.add_argument("--reinforce_grad_variance_period", type=int, default=1)
    p.add_argument("--filter_vocab", type=str2bool, default=False)
    p.add_argument("--vocab_threshold", type=float, default=-1.0)
    p.add_argument("--method", type=str, default="stgs",
                   choices=["stgs", "reinforce", "soda", "gcg", "o2p"])
    p.add_argument("--baseline_backend", type=str, default="hf", choices=["hf", "tl"])
    p.add_argument("--baseline_model_name", type=str, default=None)
    p.add_argument("--soda_decay_rate", type=float, default=0.9)
    p.add_argument("--soda_beta1", type=float, default=0.9)
    p.add_argument("--soda_beta2", type=float, default=0.995)
    p.add_argument("--soda_reset_epoch", type=int, default=50)
    p.add_argument("--soda_reinit_epoch", type=int, default=1500)
    p.add_argument("--soda_reg_weight", type=float, default=None)
    p.add_argument("--soda_bias_correction", type=str2bool, default=False)
    p.add_argument("--soda_init_strategy", type=str, default="zeros")
    p.add_argument("--soda_init_std", type=float, default=0.05)
    p.add_argument("--gcg_num_candidates", type=int, default=704)
    p.add_argument("--gcg_top_k", type=int, default=128)
    p.add_argument("--gcg_num_mutations", type=int, default=1)
    p.add_argument("--gcg_pos_choice", type=str, default="uniform")
    p.add_argument("--gcg_token_choice", type=str, default="uniform")
    p.add_argument("--gcg_init_strategy", type=str, default="zeros")
    p.add_argument("--gcg_candidate_batch_size", type=int, default=8)
    p.add_argument("--o2p_model_path", type=str, default=None)
    p.add_argument("--o2p_num_beams", type=int, default=4)
    p.add_argument("--o2p_max_length", type=int, default=32)
    p.add_argument("--early_stop_on_exact_match", type=str2bool, default=False)
    p.add_argument("--early_stop_embsim_lcs_ratio_threshold", type=float, default=1.0)
    p.add_argument("--run_discrete_validation", type=str2bool, default=False)
    p.add_argument("--run_discrete_embsim_validation", type=str2bool, default=False)
    p.add_argument("--embsim_similarity", type=str, default="cossim",
                   choices=["cossim", "dotproduct", "l2"])
    p.add_argument("--embsim_use_input_logits", type=str2bool, default=True)
    p.add_argument("--embsim_teacher_forcing", type=str2bool, default=False)
    p.add_argument("--embsim_temperature", type=float, default=1.0)
    p.add_argument("--superposition_metric_every", type=int, default=0)
    p.add_argument("--superposition_metric_modes", type=str, default="dot,cos,l2")
    p.add_argument("--superposition_vocab_top_k", type=int, default=256)
    p.add_argument("--superposition_vocab_source", type=str, default="wikipedia")
    p.add_argument("--superposition_vocab_dataset_path", type=str, default=None)
    p.add_argument("--superposition_vocab_hf_name", type=str, default="lucadiliello/english_wikipedia")
    p.add_argument("--superposition_vocab_hf_split", type=str, default="train")
    p.add_argument("--superposition_vocab_num_texts", type=int, default=1000)
    p.add_argument("--superposition_entropy_temperature", type=float, default=1.0)
    p.add_argument("--superposition_output_dir", type=str, default=".")

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
    config["reasoning_generate_kwargs"] = collect_reasoning_generate_kwargs(config)
    explicit_args = collect_explicit_arg_names(sys.argv[1:])

    if config.get("seed") is not None and "opt_seed" not in explicit_args:
        config["opt_seed"] = config["seed"]

    config["losses"] = add_loss_component(
        config.get("losses", ""),
        config.get("promptLambda", 0.0) > 0.0,
        "promptPerplexity",
    )
    config["losses"] = add_loss_component(
        config["losses"],
        config.get("complLambda", 0.0) > 0.0,
        "completionPerplexity",
    )
    config["losses"] = add_loss_component(
        config["losses"],
        config.get("promptTfComplLambda", 0.0) > 0.0,
        "promptTfComplPerplexity",
    )
    config["losses"] = add_loss_component(
        config["losses"],
        config.get("promptDistEntropyLambda", 0.0) > 0.0,
        "promptDistEntropy",
    )
    config["losses"] = add_loss_component(
        config["losses"],
        config.get("commitmentLambda", 0.0) > 0.0,
        "commitmentLoss",
    )
    config["losses"] = add_loss_component(
        config["losses"],
        config.get("eos_reg_lambda", 0.0) > 0.0,
        "eosPositionReg",
    )

    # Resolve dataset source
    if config["dataset_path"] is None and config["hf_dataset"] is None:
        # Default to HF GSM8K
        config["hf_dataset"] = "openai/gsm8k"
    elif config.get("hf_dataset") not in (None, "openai/gsm8k"):
        raise ValueError("PC batch optimization currently only supports --hf_dataset openai/gsm8k.")

    # test_eval_size defaults to test_size
    if config.get("test_eval_size", 0) <= 0:
        config["test_eval_size"] = config.get("test_size", 100)

    ensure_pc_main_compatibility(config, explicit_args=explicit_args)

    logger.info(f"Config: {config}")
    batch_pc_optimize(config)


if __name__ == "__main__":
    main()
