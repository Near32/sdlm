"""
Sweep logits_top_p and measure final LCS ratios after 1-epoch optimize_inputs calls.

Modes:
- Single target: provide --target_text (+ optional --learnable_logits_path / --learnable_temperatures_path)
- Dataset mode: provide --dataset_path + --learnable_logits_root and optional --target_indices

In dataset mode, artifacts are discovered recursively and ordered by (k, sample) from
directories named target_kX_sampleY.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import wandb


TARGET_DIR_RE = re.compile(r"target_k(\d+)_sample(\d+)$")
SAMPLE_ID_RE = re.compile(r"k(\d+)_sample(\d+)$")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.lower().strip()
        if s in ("1", "true", "t", "yes", "y"):
            return True
        if s in ("0", "false", "f", "no", "n"):
            return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {v}")


def _parse_top_p_values(raw: str) -> List[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("top_p_values is empty")
    for v in vals:
        if not (0.0 < v <= 1.0):
            raise ValueError(f"Invalid top_p value {v}; expected 0 < top_p <= 1")
    return vals


def _parse_indices(raw: Optional[str], n: int) -> List[int]:
    if raw is None or raw.strip() == "":
        return list(range(n))
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0 or idx >= n:
            raise ValueError(f"target index out of range: {idx} (n={n})")
        out.append(idx)
    return out


def _last_or_nan(xs: List[float]) -> float:
    return float(xs[-1]) if xs else float("nan")


def _plot_records(records: List[Dict], out_png: Path, title: str) -> None:
    if not records:
        return
    recs = sorted(records, key=lambda r: r["top_p"])
    xs = [r["top_p"] for r in recs]
    y_orig = [r["original_lcs_ratio"] for r in recs]
    y_disc = [r["discrete_lcs_ratio"] for r in recs]
    y_embs = [r["embsim_lcs_ratio"] for r in recs]

    plt.figure(figsize=(8, 5))
    plt.plot(xs, y_orig, marker="o", label="original_lcs_ratio")
    if any(not math.isnan(v) for v in y_disc):
        plt.plot(xs, y_disc, marker="o", label="discrete_lcs_ratio")
    if any(not math.isnan(v) for v in y_embs):
        plt.plot(xs, y_embs, marker="o", label="embsim_lcs_ratio")
    plt.xlabel("logits_top_p")
    plt.ylabel("LCS ratio")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def _extract_ks_from_sample_id(sample_id: str) -> Tuple[int, int]:
    m = SAMPLE_ID_RE.search(sample_id)
    if not m:
        raise ValueError(f"Could not parse sample id (expected kX_sampleY): {sample_id}")
    return int(m.group(1)), int(m.group(2))


def _extract_ks_from_path(path: Path) -> Optional[Tuple[int, int]]:
    for part in reversed(path.parts):
        m = TARGET_DIR_RE.search(part)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def _discover_named_artifacts(
    root: Path,
    filename: str,
) -> Tuple[List[Tuple[int, int, Path]], Dict[Tuple[int, int], Path]]:
    found: List[Tuple[int, int, Path]] = []
    for p in root.rglob(filename):
        ks = _extract_ks_from_path(p)
        if ks is None:
            continue
        found.append((ks[0], ks[1], p))
    found.sort(key=lambda x: (x[0], x[1], str(x[2])))

    key_to_path: Dict[Tuple[int, int], Path] = {}
    for k, s, p in found:
        key = (k, s)
        if key not in key_to_path:
            key_to_path[key] = p
    return found, key_to_path


def _save_records(records: List[Dict], out_dir: Path, prefix: str) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{prefix}.json"
    csv_path = out_dir / f"{prefix}.csv"
    png_path = out_dir / f"{prefix}.png"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"records": records}, f, indent=2)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["top_p", "original_lcs_ratio", "discrete_lcs_ratio", "embsim_lcs_ratio"]
        )
        writer.writeheader()
        writer.writerows(records)
    return {"json_path": str(json_path), "csv_path": str(csv_path), "png_path": str(png_path)}


def _clone_temperature_payload(payload):
    if payload is None:
        return None
    if torch.is_tensor(payload):
        return payload.detach().clone()
    if isinstance(payload, dict):
        out = {}
        for k, v in payload.items():
            out[k] = v.detach().clone() if torch.is_tensor(v) else v
        return out
    return payload


def _parse_fallback_temperature_arg(raw: Optional[str], device) -> Optional[object]:
    """Parse fallback temperatures from either:
    - existing file path (loaded via torch.load), or
    - inline float / comma-separated floats.
    """
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None

    p = Path(s)
    if p.is_file():
        return torch.load(p, map_location=device)

    try:
        vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError(
            "Invalid --fallback_learnable_temperatures. Provide either an existing .pt path "
            "or a float / comma-separated floats (e.g. '0.7' or '0.5,0.6,0.7')."
        ) from exc
    if not vals:
        raise ValueError(
            "Invalid --fallback_learnable_temperatures: empty value. "
            "Provide an existing .pt path or numeric values."
        )
    return torch.tensor(vals, dtype=torch.float32, device=device)


def _fit_temperature_values(values_1d: torch.Tensor, expected_len: int, label: str) -> torch.Tensor:
    if expected_len <= 0:
        raise ValueError(f"{label}: expected_len must be > 0, got {expected_len}")
    flat = values_1d.detach().flatten().float()
    n = int(flat.numel())
    if n == expected_len:
        return flat.clone()
    if n == 1:
        return flat.repeat(expected_len)
    raise ValueError(
        f"{label}: got {n} fallback values, expected {expected_len} "
        "(or a single scalar that can be broadcast)."
    )


def _materialize_temperature_payload(
    base_payload,
    args: argparse.Namespace,
    tokenizer,
    target_text: str,
    device,
    force_non_learnable: bool = False,
):
    """Prepare per-target initial_learnable_temperatures payload for optimize_inputs."""
    if base_payload is None:
        return None

    def _as_scalar(values, label: str) -> torch.Tensor:
        v = values.detach().flatten().float() if torch.is_tensor(values) else torch.tensor(values, dtype=torch.float32).flatten()
        if v.numel() == 0:
            return torch.tensor([float(args.temperature)], dtype=torch.float32, device=device)
        if v.numel() == 1:
            return v.to(device=device)
        logging.warning(
            "%s received %d values while fallback is forcing non-learnable temperatures; "
            "using their mean as scalar.",
            label,
            int(v.numel()),
        )
        return v.mean().reshape(1).to(device=device)

    if isinstance(base_payload, dict):
        if not force_non_learnable:
            return _clone_temperature_payload(base_payload)
        payload: Dict[str, torch.Tensor] = {}
        stgs_eff = None
        if "stgs_effective_temperature" in base_payload:
            stgs_eff = _as_scalar(base_payload["stgs_effective_temperature"], "stgs_effective_temperature")
        elif "stgs_temperature_param" in base_payload or "temperature_param" in base_payload:
            _k = "stgs_temperature_param" if "stgs_temperature_param" in base_payload else "temperature_param"
            _p = base_payload[_k].detach().float() if torch.is_tensor(base_payload[_k]) else torch.tensor(base_payload[_k], dtype=torch.float32)
            stgs_eff = _as_scalar(1e-12 + float(args.temperature) * (1.0 + torch.tanh(_p)), _k)
        if stgs_eff is not None:
            payload["stgs_effective_temperature"] = stgs_eff

        if args.bptt:
            bptt_eff = None
            if "bptt_effective_temperature" in base_payload:
                bptt_eff = _as_scalar(base_payload["bptt_effective_temperature"], "bptt_effective_temperature")
            elif "bptt_temperature_param" in base_payload:
                _p = (
                    base_payload["bptt_temperature_param"].detach().float()
                    if torch.is_tensor(base_payload["bptt_temperature_param"])
                    else torch.tensor(base_payload["bptt_temperature_param"], dtype=torch.float32)
                )
                bptt_eff = _as_scalar(1e-12 + float(args.bptt_temperature) * (1.0 + torch.tanh(_p)), "bptt_temperature_param")
            if bptt_eff is not None:
                payload["bptt_effective_temperature"] = bptt_eff
        return payload if payload else None

    if torch.is_tensor(base_payload):
        flat = base_payload.detach().to(device=device).flatten().float()
        if flat.numel() == 0:
            return None

        payload: Dict[str, torch.Tensor] = {}
        if force_non_learnable:
            payload["stgs_effective_temperature"] = _as_scalar(flat, "stgs_effective_temperature")
        else:
            n_stgs = args.seq_len if (args.learnable_temperature and args.decouple_learnable_temperature) else 1
            payload["stgs_effective_temperature"] = _fit_temperature_values(flat, n_stgs, "stgs_effective_temperature")

        if args.bptt:
            if force_non_learnable:
                payload["bptt_effective_temperature"] = _as_scalar(flat, "bptt_effective_temperature")
            else:
                target_len = int(tokenizer(target_text, return_tensors="pt").input_ids.shape[1])
                n_bptt = target_len if (args.bptt_learnable_temperature and args.bptt_decouple_learnable_temperature) else 1
                payload["bptt_effective_temperature"] = _fit_temperature_values(flat, n_bptt, "bptt_effective_temperature")
        return payload if payload else None

    return _clone_temperature_payload(base_payload)


def _run_single_target_sweep(
    args: argparse.Namespace,
    model,
    tokenizer,
    device,
    top_p_values: List[float],
    target_text: str,
    initial_logits: Optional[torch.Tensor],
    initial_temperatures,
    force_non_learnable_temperatures: bool,
    target_key: str,
    output_dir: Path,
    run,
) -> Dict:
    from main import optimize_inputs

    records: List[Dict] = []
    temperature_paths: List[Dict] = []

    for idx, top_p in enumerate(top_p_values):
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        use_learnable_temperature = args.learnable_temperature and (not force_non_learnable_temperatures)
        use_bptt_learnable_temperature = args.bptt_learnable_temperature and (not force_non_learnable_temperatures)

        opt_result = optimize_inputs(
            model=model,
            tokenizer=tokenizer,
            device=device,
            target_text=target_text,
            pre_prompt=args.pre_prompt,
            losses=args.losses,
            bptt=args.bptt,
            seq_len=args.seq_len,
            epochs=1,
            eval_only=True,
            learning_rate=args.learning_rate,
            temperature=args.temperature,
            bptt_temperature=args.bptt_temperature,
            learnable_temperature=use_learnable_temperature,
            decouple_learnable_temperature=args.decouple_learnable_temperature and use_learnable_temperature,
            bptt_learnable_temperature=use_bptt_learnable_temperature,
            bptt_decouple_learnable_temperature=args.bptt_decouple_learnable_temperature and use_bptt_learnable_temperature,
            teacher_forcing=args.teacher_forcing,
            bptt_teacher_forcing_via_diff_model=args.bptt_teacher_forcing_via_diff_model,
            stgs_hard=args.stgs_hard,
            stgs_hard_method=args.stgs_hard_method,
            stgs_hard_embsim_probs=args.stgs_hard_embsim_probs,
            bptt_stgs_hard=args.bptt_stgs_hard,
            bptt_stgs_hard_method=args.bptt_stgs_hard_method,
            bptt_stgs_hard_embsim_probs=args.bptt_stgs_hard_embsim_probs,
            gradient_estimator=args.gradient_estimator,
            batch_size=args.batch_size,
            filter_vocab=args.filter_vocab,
            vocab_threshold=args.vocab_threshold,
            logits_top_k=args.logits_top_k,
            logits_top_p=top_p,
            logits_filter_penalty=args.logits_filter_penalty,
            gumbel_noise_scale=args.gumbel_noise_scale,
            init_strategy=args.init_strategy,
            init_std=args.init_std,
            initial_learnable_logits=initial_logits.clone().detach() if initial_logits is not None else None,
            initial_learnable_temperatures=_clone_temperature_payload(initial_temperatures),
            run_discrete_validation=args.run_discrete_validation,
            run_discrete_embsim_validation=args.run_discrete_embsim_validation,
            embsim_similarity=args.embsim_similarity,
            embsim_use_input_logits=args.embsim_use_input_logits,
            embsim_teacher_forcing=args.embsim_teacher_forcing,
            embsim_temperature=args.embsim_temperature,
            plot_every=10_000_000,
            kwargs=vars(args),
        )

        rec = {
            "top_p": float(top_p),
            "original_lcs_ratio": _last_or_nan(opt_result.get("lcs_ratio_history", [])),
            "discrete_lcs_ratio": _last_or_nan(opt_result.get("discrete_lcs_ratio_history", [])),
            "embsim_lcs_ratio": _last_or_nan(opt_result.get("embsim_lcs_ratio_history", [])),
        }
        records.append(rec)

        final_temps = opt_result.get("learnable_temperatures", {})
        top_p_tag = str(top_p).replace(".", "p")
        temps_dir = output_dir / "learnable_temperatures"
        temps_dir.mkdir(parents=True, exist_ok=True)
        temps_path = temps_dir / f"learnable_temperatures_{target_key}_top_p_{top_p_tag}.pt"
        torch.save(final_temps, temps_path)
        temperature_paths.append({"top_p": float(top_p), "path": str(temps_path)})

        if final_temps:
            temps_artifact = wandb.Artifact(
                name=f"learnable-temperatures-{target_key}-tp-{top_p_tag}",
                type="tensor",
                description="Final learnable temperatures from top-p sweep call",
            )
            temps_artifact.add_file(str(temps_path))
            run.log_artifact(temps_artifact)

        run.log(
            {
                f"top_p_sweep/{target_key}/top_p": rec["top_p"],
                f"top_p_sweep/{target_key}/original_lcs_ratio": rec["original_lcs_ratio"],
                f"top_p_sweep/{target_key}/discrete_lcs_ratio": rec["discrete_lcs_ratio"],
                f"top_p_sweep/{target_key}/embsim_lcs_ratio": rec["embsim_lcs_ratio"],
                f"top_p_sweep/{target_key}/learnable_temperatures_path": str(temps_path),
                f"top_p_sweep/{target_key}/fallback_forced_non_learnable_temperature": int(force_non_learnable_temperatures),
                "top_p_sweep/target_key": target_key,
            },
            step=idx,
        )

    saved = _save_records(records, output_dir, prefix=f"top_p_lcs_sweep_{target_key}")
    _plot_records(
        records=records,
        out_png=Path(saved["png_path"]),
        title=f"LCS ratio vs logits_top_p ({target_key}, epochs=1)",
    )

    table = wandb.Table(columns=["top_p", "original_lcs_ratio", "discrete_lcs_ratio", "embsim_lcs_ratio"])
    for r in sorted(records, key=lambda x: x["top_p"]):
        table.add_data(r["top_p"], r["original_lcs_ratio"], r["discrete_lcs_ratio"], r["embsim_lcs_ratio"])
    run.log(
        {
            f"top_p_sweep/{target_key}/table": table,
            f"top_p_sweep/{target_key}/plot": wandb.Image(saved["png_path"]),
        }
    )

    return {"records": records, "learnable_temperature_paths": temperature_paths, **saved}


def run_sweep(args: argparse.Namespace) -> Dict:
    from main import setup_model_and_tokenizer

    top_p_values = _parse_top_p_values(args.top_p_values)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = setup_model_and_tokenizer(
        model_name=args.model_name,
        device=device,
        model_precision=args.model_precision,
    )

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=vars(args),
    )

    fallback_initial_temperatures = _parse_fallback_temperature_arg(
        args.fallback_learnable_temperatures,
        device=device,
    )
    if args.fallback_learnable_temperatures:
        run.summary["fallback_learnable_temperatures"] = str(args.fallback_learnable_temperatures)

    # Dataset mode: select targets and match recursively discovered artifacts.
    if args.dataset_path:
        dataset = json.load(open(args.dataset_path, "r", encoding="utf-8"))
        samples = dataset.get("samples", [])
        selected_idx = _parse_indices(args.target_indices, len(samples))
        if not args.learnable_logits_root:
            raise ValueError("--learnable_logits_root is required when --dataset_path is provided.")

        discovered_logits, key_to_logits_path = _discover_named_artifacts(
            Path(args.learnable_logits_root), "learnable_logits.pt"
        )
        if not discovered_logits:
            raise ValueError(f"No learnable_logits.pt files found under: {args.learnable_logits_root}")
        run.summary["discovered_learnable_logits_count"] = len(discovered_logits)

        temps_root = Path(args.learnable_temperatures_root) if args.learnable_temperatures_root else Path(args.learnable_logits_root)
        discovered_temps, key_to_temps_path = _discover_named_artifacts(
            temps_root, "learnable_temperatures.pt"
        )
        run.summary["discovered_learnable_temperatures_count"] = len(discovered_temps)

        all_results: Dict[str, Dict] = {}
        unresolved: List[str] = []
        used_fallback_temperature_targets = 0
        for idx in selected_idx:
            sample = samples[idx]
            sample_id = str(sample.get("id", f"idx{idx}"))
            target_text = sample.get("text")
            if not target_text:
                unresolved.append(f"{sample_id}: missing sample.text")
                continue
            try:
                key = _extract_ks_from_sample_id(sample_id)
            except Exception as exc:
                unresolved.append(f"{sample_id}: {exc}")
                continue
            logits_path = key_to_logits_path.get(key)
            if logits_path is None:
                unresolved.append(f"{sample_id}: no matching learnable_logits.pt for key={key}")
                continue

            initial_logits = torch.load(logits_path, map_location=device)
            if not torch.is_tensor(initial_logits):
                unresolved.append(f"{sample_id}: loaded non-tensor from {logits_path}")
                continue

            initial_temperatures = None
            fallback_triggered = False
            temp_path = key_to_temps_path.get(key)
            if temp_path is not None:
                initial_temperatures = torch.load(temp_path, map_location=device)
            elif fallback_initial_temperatures is not None:
                initial_temperatures = _materialize_temperature_payload(
                    base_payload=fallback_initial_temperatures,
                    args=args,
                    tokenizer=tokenizer,
                    target_text=target_text,
                    device=device,
                    force_non_learnable=True,
                )
                used_fallback_temperature_targets += 1
                fallback_triggered = True

            target_key = f"k{key[0]}_sample{key[1]}"
            target_out_dir = output_dir / target_key
            res = _run_single_target_sweep(
                args=args,
                model=model,
                tokenizer=tokenizer,
                device=device,
                top_p_values=top_p_values,
                target_text=target_text,
                initial_logits=initial_logits,
                initial_temperatures=initial_temperatures,
                force_non_learnable_temperatures=fallback_triggered,
                target_key=target_key,
                output_dir=target_out_dir,
                run=run,
            )
            res["learnable_logits_path"] = str(logits_path)
            if temp_path is not None:
                res["learnable_temperatures_path"] = str(temp_path)
            elif fallback_initial_temperatures is not None:
                res["fallback_learnable_temperatures_path"] = str(args.fallback_learnable_temperatures)
            all_results[target_key] = res

        summary = {
            "mode": "dataset",
            "dataset_path": args.dataset_path,
            "learnable_logits_root": args.learnable_logits_root,
            "learnable_temperatures_root": str(temps_root),
            "fallback_learnable_temperatures": args.fallback_learnable_temperatures,
            "selected_indices": selected_idx,
            "results": all_results,
            "unresolved": unresolved,
            "used_fallback_temperature_targets": used_fallback_temperature_targets,
            "discovered_ordered_logits": [
                {"k": int(k), "sample": int(s), "path": str(p)}
                for k, s, p in discovered_logits
            ],
            "discovered_ordered_temperatures": [
                {"k": int(k), "sample": int(s), "path": str(p)}
                for k, s, p in discovered_temps
            ],
        }
        with (output_dir / "top_p_lcs_sweep_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        run.summary["top_p_sweep_output_dir"] = str(output_dir.resolve())
        run.summary["resolved_targets"] = len(all_results)
        run.summary["unresolved_targets"] = len(unresolved)
        run.summary["used_fallback_temperature_targets"] = used_fallback_temperature_targets
        run.finish()
        return summary

    # Single-target mode.
    if not args.target_text:
        raise ValueError("Single-target mode requires --target_text (or use --dataset_path mode).")

    initial_logits = None
    if args.learnable_logits_path:
        initial_logits = torch.load(args.learnable_logits_path, map_location=device)
        if not torch.is_tensor(initial_logits):
            raise ValueError(f"Loaded object is not a tensor: {args.learnable_logits_path}")
    initial_temperatures = None
    if args.learnable_temperatures_path:
        initial_temperatures = torch.load(args.learnable_temperatures_path, map_location=device)
        fallback_triggered = False
    elif fallback_initial_temperatures is not None:
        initial_temperatures = _materialize_temperature_payload(
            base_payload=fallback_initial_temperatures,
            args=args,
            tokenizer=tokenizer,
            target_text=args.target_text,
            device=device,
            force_non_learnable=True,
        )
        fallback_triggered = True
    else:
        fallback_triggered = False

    target_key = "single_target"
    result = _run_single_target_sweep(
        args=args,
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_p_values=top_p_values,
        target_text=args.target_text,
        initial_logits=initial_logits,
        initial_temperatures=initial_temperatures,
        force_non_learnable_temperatures=fallback_triggered,
        target_key=target_key,
        output_dir=output_dir,
        run=run,
    )
    run.summary["top_p_sweep_output_dir"] = str(output_dir.resolve())
    run.finish()
    return {"mode": "single_target", target_key: result}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sweep logits_top_p vs LCS ratios using 1-epoch optimize_inputs calls.")
    p.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-135M")
    p.add_argument("--model_precision", type=str, default="full")
    p.add_argument("--target_text", type=str, default=None)
    p.add_argument("--dataset_path", type=str, default=None,
                   help="Dataset JSON path with samples[*].id and samples[*].text")
    p.add_argument("--target_indices", type=str, default=None,
                   help="Comma-separated dataset sample indices (e.g. 0,1,5). Empty = all.")
    p.add_argument("--learnable_logits_root", type=str, default=None,
                   help="Root folder to recursively discover target_kX_sampleY/learnable_logits.pt in dataset mode.")
    p.add_argument("--learnable_temperatures_root", type=str, default=None,
                   help="Optional root folder to discover target_kX_sampleY/learnable_temperatures.pt "
                        "(defaults to --learnable_logits_root).")
    p.add_argument("--pre_prompt", type=str, default=None)
    p.add_argument("--losses", type=str, default="crossentropy")
    p.add_argument("--seq_len", type=int, default=20)
    p.add_argument("--learning_rate", type=float, default=1e-2)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--gradient_estimator", type=str, default="stgs", choices=["stgs", "reinforce"])
    p.add_argument("--bptt", type=str2bool, default=False)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--bptt_temperature", type=float, default=1.0)
    p.add_argument("--learnable_temperature", type=str2bool, default=False)
    p.add_argument("--decouple_learnable_temperature", type=str2bool, default=False)
    p.add_argument("--bptt_learnable_temperature", type=str2bool, default=False)
    p.add_argument("--bptt_decouple_learnable_temperature", type=str2bool, default=False)
    p.add_argument("--teacher_forcing", type=str2bool, default=False,
                   help="Threaded to optimize_inputs teacher_forcing.")
    p.add_argument("--bptt_teacher_forcing_via_diff_model", type=str2bool, default=False,
                   help="Threaded to optimize_inputs bptt_teacher_forcing_via_diff_model.")
    p.add_argument("--stgs_hard", type=str2bool, default=True)
    p.add_argument("--stgs_hard_method", type=str, default="categorical")
    p.add_argument("--stgs_hard_embsim_probs", type=str, default="gumbel_soft")
    p.add_argument("--bptt_stgs_hard", type=str2bool, default=False)
    p.add_argument("--bptt_stgs_hard_method", type=str, default="categorical")
    p.add_argument("--bptt_stgs_hard_embsim_probs", type=str, default="gumbel_soft")

    p.add_argument("--filter_vocab", type=str2bool, default=True)
    p.add_argument("--vocab_threshold", type=float, default=0.5)
    p.add_argument("--init_strategy", type=str, default="randn")
    p.add_argument("--init_std", type=float, default=0.0)
    p.add_argument("--learnable_logits_path", type=str, default=None,
                   help="Single-target mode only: explicit learnable_logits.pt path.")
    p.add_argument("--learnable_temperatures_path", type=str, default=None,
                   help="Single-target mode only: explicit learnable_temperatures.pt path.")
    p.add_argument("--fallback_learnable_temperatures", type=str, default=None,
                   help="Optional fallback used when per-target temperatures are missing. "
                        "Accepts either a .pt path, a single float (e.g. '0.7'), "
                        "or comma-separated floats (e.g. '0.5,0.6,0.7'). "
                        "When fallback is used, the optimize_inputs call is forced to non-learnable temperature mode.")

    p.add_argument("--top_p_values", type=str, default="1.0,0.95,0.9,0.8,0.7,0.5")
    p.add_argument("--logits_top_k", type=int, default=0)
    p.add_argument("--logits_filter_penalty", type=float, default=1e4)
    p.add_argument("--gumbel_noise_scale", type=float, default=1.0,
                   help="Global Gumbel noise scale threaded to optimize_inputs.")

    p.add_argument("--run_discrete_validation", type=str2bool, default=True)
    p.add_argument("--run_discrete_embsim_validation", type=str2bool, default=True)
    p.add_argument("--embsim_similarity", type=str, default="cossim")
    p.add_argument("--embsim_use_input_logits", type=str2bool, default=True)
    p.add_argument("--embsim_teacher_forcing", type=str2bool, default=False)
    p.add_argument("--embsim_temperature", type=float, default=1.0)

    p.add_argument("--wandb_project", type=str, default="prompt-optimization-top-p-sweep")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default="top_p_lcs_sweep")
    p.add_argument("--output_dir", type=str, default="top_p_lcs_sweep_results")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    summary = run_sweep(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
