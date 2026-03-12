"""
Superposition diagnostics for learnable logits.

This module builds a callback that can be plugged into optimize_inputs().
At configurable epochs, it:
1) computes weighted embeddings from softmax(logits),
2) compares each position embedding against pairwise differences z_i - z_j,
3) logs dot/cos/l2 heatmaps and entropy over the induced (i, j) distribution.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

logger = logging.getLogger("superposition_analysis")


def _parse_modes(modes: Sequence[str] | str) -> List[str]:
    if isinstance(modes, str):
        parsed = [m.strip().lower() for m in modes.split(",") if m.strip()]
    else:
        parsed = [str(m).strip().lower() for m in modes if str(m).strip()]
    if not parsed:
        parsed = ["dot", "cos", "l2"]
    valid = {"dot", "cos", "l2"}
    invalid = [m for m in parsed if m not in valid]
    if invalid:
        raise ValueError(f"Invalid superposition mode(s): {invalid}. Valid: {sorted(valid)}")
    # Preserve user order, deduplicate.
    uniq = []
    seen = set()
    for m in parsed:
        if m not in seen:
            uniq.append(m)
            seen.add(m)
    return uniq


def _collect_strings(obj, out: List[str], max_items: int) -> None:
    if len(out) >= max_items:
        return
    if isinstance(obj, str):
        txt = obj.strip()
        if txt:
            out.append(txt)
        return
    if isinstance(obj, list):
        for item in obj:
            if len(out) >= max_items:
                break
            _collect_strings(item, out, max_items)
        return
    if isinstance(obj, dict):
        preferred = (
            "maintext",
            "text",
            "content",
            "input",
            "target_text",
            "ground_truth_prompt",
            "reasoning",
            "answer",
        )
        for k in preferred:
            if len(out) >= max_items:
                break
            if k in obj:
                _collect_strings(obj[k], out, max_items)
        for k, v in obj.items():
            if len(out) >= max_items:
                break
            if k in preferred:
                continue
            _collect_strings(v, out, max_items)


def _load_texts_from_local_dataset(dataset_path: str, max_texts: int) -> List[str]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    texts: List[str] = []
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".jl"}:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if len(texts) >= max_texts:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                _collect_strings(obj, texts, max_texts)
        return texts

    if suffix == ".txt":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if len(texts) >= max_texts:
                    break
                line = line.strip()
                if line:
                    texts.append(line)
        return texts

    # Default JSON.
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    _collect_strings(obj, texts, max_texts)
    return texts


def _load_texts_from_wikipedia(
    hf_name: str,
    hf_split: str,
    max_texts: int,
) -> List[str]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("datasets library is required for wikipedia token subset loading") from exc

    texts: List[str] = []
    try:
        ds = load_dataset(hf_name, split=hf_split, streaming=True)
        rows = islice(ds, max_texts)
    except Exception:
        ds = load_dataset(hf_name, split=hf_split)
        rows = (ds[i] for i in range(min(max_texts, len(ds))))

    for row in rows:
        if len(texts) >= max_texts:
            break
        txt = (
            row.get("maintext", "")
            or row.get("text", "")
            or row.get("content", "")
            or row.get("document", "")
        )
        if isinstance(txt, str):
            txt = txt.strip()
        if txt:
            texts.append(txt)
    return texts


def _top_common_tokens_from_texts(
    texts: Iterable[str],
    tokenizer,
    top_k: int,
) -> List[int]:
    counter: Counter = Counter()
    for text in texts:
        if not text:
            continue
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            continue
        counter.update(ids)
    return [int(tok) for tok, _ in counter.most_common(max(top_k, 1))]


def _build_ranked_token_ids(
    tokenizer,
    source: str,
    top_k: int,
    dataset_path: Optional[str],
    hf_name: str,
    hf_split: str,
    num_texts: int,
) -> List[int]:
    source = source.lower()
    if source == "dataset":
        if not dataset_path:
            raise ValueError("superposition_vocab_source=dataset requires superposition_vocab_dataset_path")
        texts = _load_texts_from_local_dataset(dataset_path, num_texts)
    elif source == "wikipedia":
        texts = _load_texts_from_wikipedia(hf_name=hf_name, hf_split=hf_split, max_texts=num_texts)
    elif source == "none":
        return []
    else:
        raise ValueError("superposition_vocab_source must be one of: wikipedia, dataset, none")

    if not texts:
        return []
    # Oversample ranking candidates so we can filter by allowed-vocab membership afterwards.
    return _top_common_tokens_from_texts(texts, tokenizer, top_k=max(top_k * 4, top_k))


def _select_subset_token_ids(
    tokenizer,
    allowed_tokens: torch.Tensor,
    vocab_top_k: int,
    vocab_source: str,
    vocab_dataset_path: Optional[str],
    vocab_hf_name: str,
    vocab_hf_split: str,
    vocab_num_texts: int,
) -> List[int]:
    allowed_full_ids = [int(t) for t in allowed_tokens.detach().cpu().tolist()]
    allowed_set = set(allowed_full_ids)

    if vocab_top_k <= 0:
        return allowed_full_ids

    selected: List[int] = []
    seen = set()
    try:
        ranked = _build_ranked_token_ids(
            tokenizer=tokenizer,
            source=vocab_source,
            top_k=vocab_top_k,
            dataset_path=vocab_dataset_path,
            hf_name=vocab_hf_name,
            hf_split=vocab_hf_split,
            num_texts=vocab_num_texts,
        )
        for tid in ranked:
            if tid in allowed_set and tid not in seen:
                selected.append(tid)
                seen.add(tid)
                if len(selected) >= vocab_top_k:
                    break
    except Exception as exc:
        logger.warning(
            "Failed to build ranked token subset from source=%s: %s. Falling back to allowed-vocab order.",
            vocab_source,
            exc,
        )

    if len(selected) < vocab_top_k:
        for tid in allowed_full_ids:
            if tid not in seen:
                selected.append(tid)
                seen.add(tid)
                if len(selected) >= vocab_top_k:
                    break

    return selected[:vocab_top_k]


def _entropy_from_scores(scores: torch.Tensor, temperature: float) -> tuple[float, float]:
    temp = max(float(temperature), 1e-6)
    probs = torch.softmax((scores.reshape(-1) / temp), dim=0)
    entropy = -(probs * (probs + 1e-12).log()).sum()
    max_ent = math.log(max(int(probs.numel()), 1))
    norm_ent = float(entropy.item() / max(max_ent, 1e-12))
    return float(entropy.item()), norm_ent


def _plot_mode_heatmaps(
    mode: str,
    matrices_cpu: torch.Tensor,  # (seq_len, K, K)
    output_path: Path,
) -> None:
    seq_len = int(matrices_cpu.shape[0])
    ncols = min(4, seq_len)
    nrows = (seq_len + ncols - 1) // ncols
    fig = plt.figure(figsize=(3.2 * ncols + 0.8, 3.0 * nrows))
    grid = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols + 1,
        width_ratios=[1.0] * ncols + [0.06],
        wspace=0.10,
        hspace=0.25,
    )
    axes = [[fig.add_subplot(grid[r, c]) for c in range(ncols)] for r in range(nrows)]
    cax = fig.add_subplot(grid[:, -1])
    im = None
    for pos in range(seq_len):
        r = pos // ncols
        c = pos % ncols
        ax = axes[r][c]
        mat = matrices_cpu[pos].numpy()
        im = ax.imshow(mat, cmap="viridis", aspect="auto")
        ax.set_title(f"pos {pos}")
        ax.set_xticks([])
        ax.set_yticks([])
    for pos in range(seq_len, nrows * ncols):
        r = pos // ncols
        c = pos % ncols
        axes[r][c].axis("off")
    if im is not None:
        fig.colorbar(im, cax=cax)
    else:
        cax.axis("off")
    fig.suptitle(f"Superposition heatmap ({mode})", fontsize=12, y=0.98)
    fig.subplots_adjust(top=0.86, right=0.97)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_superposition_callback(
    tokenizer,
    embedding_weights_subset: torch.Tensor,
    allowed_tokens: torch.Tensor,
    output_dir: str,
    log_every: int = 10,
    modes: Sequence[str] | str = ("dot", "cos", "l2"),
    vocab_top_k: int = 0,
    vocab_source: str = "wikipedia",
    vocab_dataset_path: Optional[str] = None,
    vocab_hf_name: str = "lucadiliello/english_wikipedia",
    vocab_hf_split: str = "train",
    vocab_num_texts: int = 1000,
    entropy_temperature: float = 1.0,
) -> Callable[[int, torch.Tensor], Dict]:
    """
    Build per-epoch callback for superposition diagnostics.
    """
    mode_list = _parse_modes(modes)
    selected_full_ids = _select_subset_token_ids(
        tokenizer=tokenizer,
        allowed_tokens=allowed_tokens,
        vocab_top_k=vocab_top_k,
        vocab_source=vocab_source,
        vocab_dataset_path=vocab_dataset_path,
        vocab_hf_name=vocab_hf_name,
        vocab_hf_split=vocab_hf_split,
        vocab_num_texts=vocab_num_texts,
    )
    if len(selected_full_ids) < 2:
        raise ValueError("Superposition metric requires at least 2 selected tokens.")

    allowed_full_ids = [int(t) for t in allowed_tokens.detach().cpu().tolist()]
    allowed_token_to_idx = {tid: idx for idx, tid in enumerate(allowed_full_ids)}
    selected_allowed_idx = torch.tensor(
        [allowed_token_to_idx[tid] for tid in selected_full_ids],
        device=embedding_weights_subset.device,
        dtype=torch.long,
    )

    z = embedding_weights_subset[selected_allowed_idx].detach()  # (K, D)
    z_norm_sq = (z * z).sum(dim=-1)  # (K,)
    # ||z_i - z_j||^2 = ||z_i||^2 + ||z_j||^2 - 2 z_i.z_j
    gram = z @ z.T
    pair_norm_sq = (z_norm_sq[:, None] + z_norm_sq[None, :] - 2.0 * gram).clamp(min=1e-12)
    pair_norm = pair_norm_sq.sqrt()

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Superposition callback enabled: log_every=%d, modes=%s, subset_size=%d, source=%s",
        log_every,
        ",".join(mode_list),
        len(selected_full_ids),
        vocab_source,
    )

    def callback(epoch: int, logits: torch.Tensor, force: bool = False) -> Dict:
        if (not force) and (log_every <= 0 or (epoch % log_every != 0)):
            return {}
        if logits.dim() != 3 or logits.shape[0] != 1:
            logger.warning("Superposition callback expected logits shape (1, seq_len, vocab), got %s", tuple(logits.shape))
            return {}

        with torch.no_grad():
            probs = F.softmax(logits[0], dim=-1)  # (seq_len, allowed_vocab)
            weighted_embeds = probs @ embedding_weights_subset.detach()  # (seq_len, D)
            seq_len = int(weighted_embeds.shape[0])

            mode_mats: Dict[str, List[torch.Tensor]] = {m: [] for m in mode_list}
            mode_entropy: Dict[str, List[float]] = {m: [] for m in mode_list}
            mode_entropy_norm: Dict[str, List[float]] = {m: [] for m in mode_list}

            for pos in range(seq_len):
                w = weighted_embeds[pos]  # (D,)
                a = z @ w  # (K,)
                dot_mat = a[:, None] - a[None, :]  # (K, K)
                w_norm_sq = (w * w).sum().clamp(min=1e-12)
                w_norm = w_norm_sq.sqrt()

                if "dot" in mode_list:
                    dot_scores = dot_mat
                    ent, ent_norm = _entropy_from_scores(dot_scores, entropy_temperature)
                    mode_mats["dot"].append(dot_scores.detach().float().cpu())
                    mode_entropy["dot"].append(ent)
                    mode_entropy_norm["dot"].append(ent_norm)

                if "cos" in mode_list:
                    cos_scores = dot_mat / (w_norm * pair_norm + 1e-12)
                    ent, ent_norm = _entropy_from_scores(cos_scores, entropy_temperature)
                    mode_mats["cos"].append(cos_scores.detach().float().cpu())
                    mode_entropy["cos"].append(ent)
                    mode_entropy_norm["cos"].append(ent_norm)

                if "l2" in mode_list:
                    l2_dist = (pair_norm_sq + w_norm_sq - 2.0 * dot_mat).clamp(min=1e-12).sqrt()
                    ent, ent_norm = _entropy_from_scores(-l2_dist, entropy_temperature)
                    mode_mats["l2"].append(l2_dist.detach().float().cpu())
                    mode_entropy["l2"].append(ent)
                    mode_entropy_norm["l2"].append(ent_norm)

        metrics: Dict = {}
        for mode in mode_list:
            mode_tensor = torch.stack(mode_mats[mode], dim=0)  # (seq_len, K, K)
            heatmap_path = output_dir_path / f"superposition_{mode}_epoch_{epoch:04d}.png"
            _plot_mode_heatmaps(mode=mode, matrices_cpu=mode_tensor, output_path=heatmap_path)

            ent_values = mode_entropy[mode]
            ent_norm_values = mode_entropy_norm[mode]
            metrics[f"superposition/{mode}/entropy_mean"] = float(sum(ent_values) / len(ent_values))
            metrics[f"superposition/{mode}/entropy_norm_mean"] = float(sum(ent_norm_values) / len(ent_norm_values))
            for pos, (ent, ent_norm) in enumerate(zip(ent_values, ent_norm_values)):
                metrics[f"superposition/{mode}/entropy_pos_{pos}"] = float(ent)
                metrics[f"superposition/{mode}/entropy_norm_pos_{pos}"] = float(ent_norm)

            try:
                import wandb

                if wandb.run is not None:
                    metrics[f"superposition/{mode}/heatmap"] = wandb.Image(str(heatmap_path))
            except Exception:
                pass

        return metrics

    return callback
