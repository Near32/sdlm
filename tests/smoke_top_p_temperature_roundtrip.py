#!/usr/bin/env python3
"""
Smoke test for top_p_lcs_sweep temperature roundtrip.

This test avoids loading real models by mocking the `main` module expected by
`benchmarking/invertibility/top_p_lcs_sweep.py`.
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
        return object(), object()

    def optimize_inputs(**kwargs):
        top_p = float(kwargs["logits_top_p"])
        init_temps = kwargs.get("initial_learnable_temperatures")
        if init_temps is not None and not isinstance(init_temps, dict):
            raise AssertionError("initial_learnable_temperatures must be a dict when provided.")

        return {
            "lcs_ratio_history": [top_p],
            "discrete_lcs_ratio_history": [top_p * 0.5],
            "embsim_lcs_ratio_history": [top_p * 0.25],
            "learnable_temperatures": {
                "stgs_temperature_param": torch.tensor([top_p], dtype=torch.float32),
                "stgs_effective_temperature": torch.tensor([top_p + 0.1], dtype=torch.float32),
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

    with tempfile.TemporaryDirectory(prefix="smoke_top_p_temp_", dir="/tmp") as td:
        root = Path(td)
        dataset_path = root / "dataset.json"
        logits_root = root / "artifacts"
        target_dir = logits_root / "target_k1_sample0"
        target_dir.mkdir(parents=True, exist_ok=True)

        torch.save(torch.randn(1, 4, 16), target_dir / "learnable_logits.pt")
        torch.save(
            {"stgs_temperature_param": torch.tensor([0.7], dtype=torch.float32)},
            target_dir / "learnable_temperatures.pt",
        )

        with dataset_path.open("w", encoding="utf-8") as f:
            json.dump({"samples": [{"id": "k1_sample0", "text": "hello world"}]}, f)

        parser = sweep.build_parser()
        args = parser.parse_args(
            [
                "--dataset_path",
                str(dataset_path),
                "--learnable_logits_root",
                str(logits_root),
                "--learnable_temperatures_root",
                str(logits_root),
                "--target_indices",
                "0",
                "--top_p_values",
                "1.0,0.8",
                "--output_dir",
                str(root / "out"),
                "--wandb_project",
                "smoke-test",
                "--wandb_run_name",
                "smoke_top_p_temperature_roundtrip",
            ]
        )

        summary = sweep.run_sweep(args)
        assert summary["mode"] == "dataset"
        result = summary["results"]["k1_sample0"]
        assert len(result["records"]) == 2, "Expected two top_p records."

        for row in result["learnable_temperature_paths"]:
            path = Path(row["path"])
            assert path.exists(), f"Missing saved temperature file: {path}"
            payload = torch.load(path, map_location="cpu")
            assert "stgs_temperature_param" in payload

        print("PASS: top_p temperature roundtrip smoke test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
