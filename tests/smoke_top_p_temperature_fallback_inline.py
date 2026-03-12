#!/usr/bin/env python3
"""
Smoke test for inline fallback temperatures in top_p_lcs_sweep.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
from pathlib import Path

import torch


def _install_mock_main() -> None:
    mock_main = types.ModuleType("main")

    def setup_model_and_tokenizer(model_name, device, model_precision):
        class _Tok:
            def __call__(self, text, return_tensors="pt"):
                # fixed target length for deterministic checks
                return types.SimpleNamespace(input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long))

        return object(), _Tok()

    def optimize_inputs(**kwargs):
        assert kwargs.get("learnable_temperature") is False
        assert kwargs.get("bptt_learnable_temperature") is False
        init_temps = kwargs.get("initial_learnable_temperatures")
        assert isinstance(init_temps, dict), "Expected dict initial_learnable_temperatures from fallback values."
        assert "stgs_effective_temperature" in init_temps, "Missing stgs_effective_temperature"
        got = init_temps["stgs_effective_temperature"].detach().cpu().flatten().tolist()
        exp = [0.25]
        assert len(got) == len(exp) and all(abs(a - b) < 1e-6 for a, b in zip(got, exp)), (
            f"Unexpected fallback vector: {got} vs {exp}"
        )

        return {
            "lcs_ratio_history": [0.5],
            "discrete_lcs_ratio_history": [0.4],
            "embsim_lcs_ratio_history": [0.3],
            "learnable_temperatures": {
                "stgs_effective_temperature": init_temps["stgs_effective_temperature"].detach().clone(),
            },
        }

    mock_main.setup_model_and_tokenizer = setup_model_and_tokenizer
    mock_main.optimize_inputs = optimize_inputs
    sys.modules["main"] = mock_main


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    inv_dir = repo_root / "benchmarking" / "invertibility"
    sys.path.insert(0, str(inv_dir))

    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_CACHE_DIR", "/tmp/wandb_cache")
    os.environ.setdefault("WANDB_CONFIG_DIR", "/tmp/wandb_config")
    os.environ.setdefault("WANDB_DATA_DIR", "/tmp/wandb_data")
    os.environ.setdefault("WANDB_DIR", "/tmp/wandb_runs")
    os.environ.setdefault("MPLBACKEND", "Agg")

    _install_mock_main()
    import top_p_lcs_sweep as sweep

    with tempfile.TemporaryDirectory(prefix="smoke_top_p_temp_fb_", dir="/tmp") as td:
        root = Path(td)
        dataset_path = root / "dataset.json"
        logits_root = root / "artifacts"
        target_dir = logits_root / "target_k1_sample0"
        target_dir.mkdir(parents=True, exist_ok=True)

        # No learnable_temperatures.pt on purpose; fallback must be used.
        torch.save(torch.randn(1, 4, 16), target_dir / "learnable_logits.pt")

        with dataset_path.open("w", encoding="utf-8") as f:
            json.dump({"samples": [{"id": "k1_sample0", "text": "hello world"}]}, f)

        parser = sweep.build_parser()
        args = parser.parse_args(
            [
                "--dataset_path",
                str(dataset_path),
                "--learnable_logits_root",
                str(logits_root),
                "--target_indices",
                "0",
                "--top_p_values",
                "1.0",
                "--output_dir",
                str(root / "out"),
                "--wandb_project",
                "smoke-test",
                "--wandb_run_name",
                "smoke_top_p_temperature_fallback_inline",
                "--learnable_temperature",
                "True",
                "--decouple_learnable_temperature",
                "True",
                "--seq_len",
                "4",
                "--fallback_learnable_temperatures",
                "0.1,0.2,0.3,0.4",
            ]
        )

        summary = sweep.run_sweep(args)
        assert summary["mode"] == "dataset"
        assert int(summary["used_fallback_temperature_targets"]) == 1
        print("PASS: inline fallback temperatures smoke test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
