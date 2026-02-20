#!/usr/bin/env python3
"""Generate LaTeX tables from optimization experiment results.

This script combines dataset files with W&B run data to produce
publication-ready LaTeX tables showing optimization results.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_wandb_id(wandb_id: str) -> tuple[str, str, str]:
    """Extract entity, project, run_id from W&B ID.

    Args:
        wandb_id: W&B run identifier in format "entity/project/runs/run_id"
                  or "entity/project/run_id"

    Returns:
        Tuple of (entity, project, run_id)
    """
    # Remove leading slash if present
    wandb_id = wandb_id.lstrip("/")

    # Handle format: entity/project/runs/run_id
    if "/runs/" in wandb_id:
        parts = wandb_id.split("/runs/")
        entity_project = parts[0]
        run_id = parts[1]
        entity, project = entity_project.split("/", 1)
    else:
        # Handle format: entity/project/run_id
        parts = wandb_id.split("/")
        if len(parts) == 3:
            entity, project, run_id = parts
        else:
            raise ValueError(
                f"Invalid W&B ID format: {wandb_id}. "
                "Expected 'entity/project/runs/run_id' or 'entity/project/run_id'"
            )

    return entity, project, run_id


def load_dataset(path: str) -> dict:
    """Load and parse dataset JSON file.

    Supports two formats:
    - Diverse targets: samples[].id, k_target, text, perplexity
    - SODA: samples[].id, ground_truth_prompt, text, target_output_tokens

    Args:
        path: Path to dataset JSON file

    Returns:
        Dictionary with parsed dataset info
    """
    with open(path, "r") as f:
        data = json.load(f)

    samples = data.get("samples", [])

    # Detect format and normalize
    dataset_info = {
        "samples": {},
        "has_ground_truth_prompt": False,
    }

    for sample in samples:
        sample_id = sample.get("id", sample.get("sample_id"))
        if sample_id is None:
            continue

        info = {
            "id": sample_id,
            "text": sample.get("text", sample.get("target_output", "")),
            "k_value": sample.get("k_target", sample.get("k_value")),
        }

        # Check for SODA format with ground truth prompt
        if "ground_truth_prompt" in sample:
            info["ground_truth_prompt"] = sample["ground_truth_prompt"]
            dataset_info["has_ground_truth_prompt"] = True

        if "perplexity" in sample:
            info["perplexity"] = sample["perplexity"]

        dataset_info["samples"][str(sample_id)] = info

    return dataset_info


def fetch_wandb_results(entity: str, project: str, run_id: str) -> tuple[pd.DataFrame, wandb.apis.public.Run]:
    """Fetch optimization results from W&B.

    Args:
        entity: W&B entity name
        project: W&B project name
        run_id: W&B run ID

    Returns:
        Tuple of (DataFrame with results, run object)
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    # Try to fetch optimization_summary table from artifacts
    results_df = None

    for artifact in run.logged_artifacts():
        if "optimization_summary" in artifact.name:
            try:
                table = artifact.get("optimization_summary")
                if table is not None:
                    results_df = pd.DataFrame(data=table.data, columns=table.columns)
                    break
            except Exception:
                pass

    # Fallback to scan_history
    if results_df is None:
        history = list(run.scan_history())
        if history:
            results_df = pd.DataFrame(history)

    if results_df is None:
        results_df = pd.DataFrame()

    return results_df, run


def fetch_learned_prompts(
    run: wandb.apis.public.Run,
    entity: str,
    project: str,
    run_id: str,
    dataset_info: dict,
    results_dir: Optional[str] = None,
) -> dict[str, dict]:
    """Fetch learned prompt texts and tokens from result artifacts.

    Artifacts are named: result-{run_id}_k{k_value}_sample{sample_idx}:v0

    Args:
        run: W&B run object
        entity: W&B entity name
        project: W&B project name
        run_id: W&B run ID
        dataset_info: Dataset info dict with samples
        results_dir: Optional local results directory path

    Returns:
        Dictionary mapping sample_id to dict with 'text' and 'tokens' keys
    """
    learned_prompts = {}
    api = wandb.Api()

    # Try to get from W&B config for local results path
    if results_dir is None:
        config = run.config
        results_dir = config.get("results_dir", config.get("output_dir"))

    # Try local results directory first
    if results_dir:
        results_path = Path(results_dir)
        if results_path.exists():
            for result_file in results_path.glob("**/result*.json"):
                try:
                    with open(result_file, "r") as f:
                        result_data = json.load(f)

                    sample_id = result_data.get("sample_id", result_data.get("id"))
                    if sample_id is not None:
                        prompt_info = {
                            "text": result_data.get("optimized_text", result_data.get("learned_prompt", "")),
                            "tokens": result_data.get("optimized_tokens", []),
                        }
                        learned_prompts[str(sample_id)] = prompt_info
                except (json.JSONDecodeError, IOError):
                    continue

    # Fetch all 'results' type artifacts from the run
    # Artifacts contain: target_id, target_text, target_k, optimized_text, optimized_tokens, etc.
    print("  Scanning 'results' type artifacts from run...", file=sys.stderr)
    for artifact in run.logged_artifacts():
        if artifact.type == "results":
            try:
                artifact_dir = artifact.download()

                for result_file in Path(artifact_dir).glob("**/*.json"):
                    with open(result_file, "r") as f:
                        result_data = json.load(f)

                    # Use target_id as sample identifier
                    sample_id = result_data.get("target_id", result_data.get("sample_id", result_data.get("id")))

                    if sample_id is not None:
                        prompt_info = {
                            "text": result_data.get("optimized_text", result_data.get("learned_prompt", "")),
                            "tokens": result_data.get("optimized_tokens", []),
                            "target_k": result_data.get("target_k"),
                            "generated_text": result_data.get("generated_text", ""),
                            "lcs_ratio": result_data.get("evaluation", {}).get("lcs_ratio") if isinstance(result_data.get("evaluation"), dict) else None,
                            "final_loss": result_data.get("final_loss"),
                        }
                        learned_prompts[str(sample_id)] = prompt_info
                        print(f"    Loaded: {artifact.name} -> target_id={sample_id}", file=sys.stderr)

            except Exception as e:
                print(f"    Error processing {artifact.name}: {e}", file=sys.stderr)
                continue

    # Fallback: try iterating through all logged artifacts
    if not learned_prompts:
        print("  Trying fallback: scanning all logged artifacts...", file=sys.stderr)
        all_artifacts = list(run.logged_artifacts())
        print(f"  Found {len(all_artifacts)} total logged artifacts", file=sys.stderr)

        for artifact in all_artifacts:
            # Check for 'results' type artifacts
            if artifact.type == "results":
                print(f"    Processing: {artifact.name}", file=sys.stderr)
                try:
                    artifact_dir = artifact.download()

                    for result_file in Path(artifact_dir).glob("**/*.json"):
                        with open(result_file, "r") as f:
                            result_data = json.load(f)

                        # Use target_id as sample identifier
                        sample_id = result_data.get("target_id", result_data.get("sample_id", result_data.get("id")))

                        if sample_id is not None:
                            prompt_info = {
                                "text": result_data.get("optimized_text", result_data.get("learned_prompt", "")),
                                "tokens": result_data.get("optimized_tokens", []),
                                "target_k": result_data.get("target_k"),
                                "generated_text": result_data.get("generated_text", ""),
                                "lcs_ratio": result_data.get("evaluation", {}).get("lcs_ratio") if isinstance(result_data.get("evaluation"), dict) else None,
                                "final_loss": result_data.get("final_loss"),
                            }
                            learned_prompts[str(sample_id)] = prompt_info
                            print(f"      Loaded: target_id={sample_id}", file=sys.stderr)

                except Exception as e:
                    print(f"      Error: {e}", file=sys.stderr)
                    continue

    print(f"  Found {len(learned_prompts)} learned prompts", file=sys.stderr)
    return learned_prompts


def compute_prompt_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_tokens: list[int],
) -> float:
    """Compute perplexity of a prompt sequence with length normalization.

    Perplexity = exp(sum of NLL / number of tokens)

    HuggingFace's loss with labels uses mean reduction by default,
    so exp(loss) gives the correctly length-normalized perplexity.

    Args:
        model: HuggingFace causal LM model
        tokenizer: HuggingFace tokenizer
        prompt_tokens: List of token IDs

    Returns:
        Perplexity value
    """
    if not prompt_tokens:
        return float("nan")

    input_ids = torch.tensor([prompt_tokens], device=model.device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # loss is already mean-reduced (averaged over tokens)
        loss = outputs.loss

    perplexity = torch.exp(loss).item()
    return perplexity


def load_perplexity_model(model_name: str, device: Optional[str] = None) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer for perplexity computation.

    Args:
        model_name: HuggingFace model name
        device: Device to load model on (auto-detected if None)

    Returns:
        Tuple of (model, tokenizer)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model = model.to(device)
    model.eval()

    return model, tokenizer


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters.

    Args:
        text: Input text

    Returns:
        Text with LaTeX special characters escaped
    """
    if not isinstance(text, str):
        text = str(text)

    # Order matters - escape backslash first
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
        ("\n", " "),
        ("\r", ""),
        ("\t", " "),
    ]

    for old, new in replacements:
        text = text.replace(old, new)

    return text


def truncate_text(text: str, max_len: int) -> str:
    """Truncate text with ellipsis.

    Args:
        text: Input text
        max_len: Maximum length (0 or negative means no truncation)

    Returns:
        Truncated text with ellipsis if needed
    """
    if not isinstance(text, str):
        text = str(text)

    # max_len <= 0 means no truncation
    if max_len <= 0 or len(text) <= max_len:
        return text

    return text[: max_len - 3] + "..."


def generate_latex_table(
    data: pd.DataFrame,
    caption: str,
    label: str,
    has_ground_truth: bool = False,
    truncate_len: int = 50,
) -> str:
    """Generate complete LaTeX longtable string (spans multiple pages).

    Args:
        data: DataFrame with table data
        caption: Table caption
        label: Table label for references
        has_ground_truth: Whether to include ground truth prompt column
        truncate_len: Maximum length for text columns

    Returns:
        Complete LaTeX longtable string
    """
    # Define columns
    if has_ground_truth:
        col_spec = "c|c|p{2.5cm}|p{2.5cm}|p{2.5cm}|c|p{2.5cm}|c"
        num_cols = 8
        headers = [
            r"\textbf{ID}",
            r"\textbf{Rank $k$}",
            r"\textbf{Target Output}",
            r"\parbox{2.5cm}{\centering\textbf{Ground-Truth Prompt}}",
            r"\parbox{2.5cm}{\centering\textbf{Learned Prompt}}",
            r"\parbox{1.5cm}{\centering\textbf{Prompt PPL}}",
            r"\textbf{Sampled Output}",
            r"\textbf{LCS}",
        ]
    else:
        col_spec = "c|c|p{2.5cm}|p{2.5cm}|c|p{2.5cm}|c"
        num_cols = 7
        headers = [
            r"\textbf{ID}",
            r"\textbf{Rank $k$}",
            r"\textbf{Target Output}",
            r"\parbox{2.5cm}{\centering\textbf{Learned Prompt}}",
            r"\parbox{1.5cm}{\centering\textbf{Prompt PPL}}",
            r"\textbf{Sampled Output}",
            r"\textbf{LCS}",
        ]

    header_row = " & ".join(headers) + r" \\"

    # Build longtable
    lines = [
        f"\\begin{{longtable}}{{{col_spec}}}",
        f"\\caption{{{escape_latex(caption)}}} \\label{{{label}}} \\\\",
        r"\hline",
        header_row,
        r"\hline",
        r"\endfirsthead",
        "",
        f"\\multicolumn{{{num_cols}}}{{c}}{{{{\\bfseries \\tablename\\ \\thetable{{}} -- continued from previous page}}}} \\\\",
        r"\hline",
        header_row,
        r"\hline",
        r"\endhead",
        "",
        f"\\hline \\multicolumn{{{num_cols}}}{{|r|}}{{{{Continued on next page}}}} \\\\",
        r"\hline",
        r"\endfoot",
        "",
        r"\hline",
        r"\endlastfoot",
        "",
    ]

    # Add rows
    for _, row in data.iterrows():
        sample_id = escape_latex(str(row.get("id", "")))
        k_value = row.get("k_value", "")
        if pd.notna(k_value):
            k_value = f"{int(k_value)}" if float(k_value) == int(float(k_value)) else f"{k_value:.1f}"
        else:
            k_value = "-"

        target_text = escape_latex(truncate_text(str(row.get("target_text", "")), truncate_len))
        learned_prompt = escape_latex(truncate_text(str(row.get("learned_prompt", "")), truncate_len))

        prompt_ppl = row.get("prompt_perplexity", float("nan"))
        if pd.notna(prompt_ppl) and not (isinstance(prompt_ppl, float) and (prompt_ppl != prompt_ppl)):  # Check for NaN
            prompt_ppl_str = f"{prompt_ppl:.2f}"
        else:
            prompt_ppl_str = "-"

        sampled_output = escape_latex(truncate_text(str(row.get("sampled_output", "")), truncate_len))

        lcs_ratio = row.get("lcs_ratio", float("nan"))
        if pd.notna(lcs_ratio):
            lcs_str = f"{lcs_ratio:.3f}"
        else:
            lcs_str = "-"

        if has_ground_truth:
            gt_prompt = escape_latex(truncate_text(str(row.get("ground_truth_prompt", "")), truncate_len))
            row_values = [sample_id, k_value, target_text, gt_prompt, learned_prompt, prompt_ppl_str, sampled_output, lcs_str]
        else:
            row_values = [sample_id, k_value, target_text, learned_prompt, prompt_ppl_str, sampled_output, lcs_str]

        lines.append(" & ".join(row_values) + r" \\ \hline")

    # Close longtable
    lines.append(r"\end{longtable}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from optimization experiment results"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset JSON file",
    )
    parser.add_argument(
        "--wandb-id",
        type=str,
        required=True,
        help="W&B run ID (e.g., entity/project/runs/run_id)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name for perplexity computation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output LaTeX file path (default: stdout)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum rows to include",
    )
    parser.add_argument(
        "--truncate",
        type=int,
        default=50,
        help="Max chars for text columns (default: 50)",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default="Optimization Results",
        help="Table caption text",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="tab:optimization_results",
        help="Table label for references",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Local results directory (overrides W&B config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for model (auto-detected if not specified)",
    )

    args = parser.parse_args()

    # Parse W&B ID
    print(f"Parsing W&B ID: {args.wandb_id}", file=sys.stderr)
    entity, project, run_id = parse_wandb_id(args.wandb_id)
    print(f"  Entity: {entity}, Project: {project}, Run ID: {run_id}", file=sys.stderr)

    # Load dataset
    print(f"Loading dataset: {args.dataset}", file=sys.stderr)
    dataset_info = load_dataset(args.dataset)
    print(f"  Found {len(dataset_info['samples'])} samples", file=sys.stderr)
    print(f"  Has ground truth prompts: {dataset_info['has_ground_truth_prompt']}", file=sys.stderr)

    # Fetch W&B results
    print(f"Fetching W&B results...", file=sys.stderr)
    wandb_df, run = fetch_wandb_results(entity, project, run_id)
    print(f"  Found {len(wandb_df)} result rows", file=sys.stderr)

    # Fetch learned prompts
    print(f"Fetching learned prompts...", file=sys.stderr)
    learned_prompts = fetch_learned_prompts(
        run=run,
        entity=entity,
        project=project,
        run_id=run_id,
        dataset_info=dataset_info,
        results_dir=args.results_dir,
    )
    print(f"  Found {len(learned_prompts)} learned prompts", file=sys.stderr)

    # Load model for perplexity computation
    print(f"Loading model for perplexity: {args.model}", file=sys.stderr)
    model, tokenizer = load_perplexity_model(args.model, args.device)
    print(f"  Model loaded on device: {model.device}", file=sys.stderr)

    # Combine data
    print("Combining data...", file=sys.stderr)
    combined_data = []

    # Use dataset samples as base
    for sample_id, sample_info in dataset_info["samples"].items():
        row = {
            "id": sample_info["id"],
            "k_value": sample_info.get("k_value"),
            "target_text": sample_info.get("text", ""),
        }

        if dataset_info["has_ground_truth_prompt"]:
            row["ground_truth_prompt"] = sample_info.get("ground_truth_prompt", "")

        # Add learned prompt info from artifacts
        if sample_id in learned_prompts:
            prompt_info = learned_prompts[sample_id]
            row["learned_prompt"] = prompt_info.get("text", "")

            # Use data from artifact
            if prompt_info.get("generated_text"):
                row["sampled_output"] = prompt_info["generated_text"]
            if prompt_info.get("lcs_ratio") is not None:
                row["lcs_ratio"] = prompt_info["lcs_ratio"]
            if prompt_info.get("target_k") is not None and row.get("k_value") is None:
                row["k_value"] = prompt_info["target_k"]

            # Compute perplexity if tokens available
            tokens = prompt_info.get("tokens", [])
            if tokens:
                row["prompt_perplexity"] = compute_prompt_perplexity(model, tokenizer, tokens)
            else:
                # Try to tokenize the text
                if prompt_info.get("text"):
                    tokens = tokenizer.encode(prompt_info["text"], add_special_tokens=False)
                    row["prompt_perplexity"] = compute_prompt_perplexity(model, tokenizer, tokens)

        combined_data.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(combined_data)

    # Sort by ID
    df = df.sort_values("id")

    # Limit rows if requested
    if args.max_rows is not None:
        df = df.head(args.max_rows)

    print(f"  Combined {len(df)} rows", file=sys.stderr)

    # Generate LaTeX
    print("Generating LaTeX table...", file=sys.stderr)
    latex_output = generate_latex_table(
        df,
        caption=args.caption,
        label=args.label,
        has_ground_truth=dataset_info["has_ground_truth_prompt"],
        truncate_len=args.truncate,
    )

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(latex_output)
        print(f"LaTeX table written to: {args.output}", file=sys.stderr)
    else:
        print(latex_output)

    print("Done!", file=sys.stderr)


if __name__ == "__main__":
    main()
