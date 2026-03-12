#!/usr/bin/env python3
"""
Smoke test for invertibility superposition diagnostics.

This script verifies:
1) callback construction succeeds,
2) callback returns entropy metrics,
3) local heatmap files are produced.
"""

import json
import shutil
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarking.invertibility.superposition_analysis import build_superposition_callback


class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        # Deterministic tokenization into a bounded pseudo-vocab.
        return [ord(ch) % 64 for ch in text if not ch.isspace()]


def _run_case(output_dir: Path, vocab_source: str, dataset_path: str | None = None) -> None:
    tokenizer = DummyTokenizer()
    embedding_weights_subset = torch.randn(64, 24)
    allowed_tokens = torch.arange(64)
    callback = build_superposition_callback(
        tokenizer=tokenizer,
        embedding_weights_subset=embedding_weights_subset,
        allowed_tokens=allowed_tokens,
        output_dir=str(output_dir),
        log_every=1,
        modes="dot,cos,l2",
        vocab_top_k=12,
        vocab_source=vocab_source,
        vocab_dataset_path=dataset_path,
        vocab_num_texts=8,
    )

    logits = torch.randn(1, 4, 64)
    metrics = callback(0, logits)
    expected = [
        "superposition/dot/entropy_mean",
        "superposition/cos/entropy_mean",
        "superposition/l2/entropy_mean",
    ]
    for key in expected:
        assert key in metrics, f"Missing metric key: {key}"

    for mode in ("dot", "cos", "l2"):
        heatmap_path = output_dir / f"superposition_{mode}_epoch_0000.png"
        assert heatmap_path.exists(), f"Missing heatmap: {heatmap_path}"


def main() -> None:
    root = Path("tests/.smoke_superposition_artifacts")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    # Case 1: no external dataset ranking
    _run_case(output_dir=root / "none_source", vocab_source="none")

    # Case 2: dataset-driven top-K ranking
    dataset_path = root / "toy_dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "samples": [
                    {"text": "alpha beta gamma delta"},
                    {"text": "beta beta gamma epsilon"},
                    {"text": "zeta eta theta iota"},
                ]
            }
        ),
        encoding="utf-8",
    )
    _run_case(
        output_dir=root / "dataset_source",
        vocab_source="dataset",
        dataset_path=str(dataset_path),
    )

    print(f"Smoke test passed. Artifacts written to: {root}")


if __name__ == "__main__":
    main()
