#!/usr/bin/env python3
"""Generate LaTeX tables with sub-row layout (v3).

v3 layout changes from v2:
- Rank k is the *first* column; Sample ID (integer X extracted from kY_sampleX)
  is the *second* column. Both span all sub-rows via \\multirow.
- Each logical row splits into 3 sub-rows (4 when has_ground_truth):
    sub-row 1 : Learned Prompt image
    sub-row GT: Ground-Truth Prompt image  [only when has_ground_truth]
    sub-row 2 : Target Output image
    sub-row 3 : (Re-)sampled Output image
  Images are rendered full-width (\\includegraphics[width=\\linewidth]) so they
  span the entire content column.
- A single *Metrics* column at the far right holds one value per sub-row:
    sub-row 1 : Prompt PPL
    sub-row GT: "-"
    sub-row 2 : Target PPL  (from dataset perplexity field)
    sub-row 3 : LCS  (recomputed when --resample, otherwise from W&B)
- \\cline{3-4} separates sub-rows within a logical row; \\hline separates
  logical rows.

Requires in the LaTeX preamble:
  \\usepackage{graphicx}
  \\usepackage{xcolor}
  \\usepackage{multirow}
  \\usepackage{longtable}
  \\usepackage{booktabs}   % optional, for nicer rules
"""

import argparse
import difflib
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from generate_functional_lmi_latex_table import (
    compute_prompt_perplexity,
    escape_latex,
    fetch_learned_prompts,
    fetch_wandb_results,
    load_dataset,
    load_perplexity_model,
    parse_wandb_id,
    truncate_text,
)
from text_to_png import render_text_to_png


# ---------------------------------------------------------------------------
# Helpers shared with / copied from v2
# ---------------------------------------------------------------------------

def resample_from_prompt(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_tokens: list,
    max_new_tokens: int = 50,
) -> str:
    """Greedy-decode `max_new_tokens` tokens conditioned on `prompt_tokens`."""
    if not prompt_tokens:
        return ""
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = output_ids[0, len(prompt_tokens):]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def _lcs_length(a: list, b: list) -> int:
    """Length of the longest common subsequence of two token lists."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_lcs_ratio(
    tokenizer: AutoTokenizer,
    generated_text: str,
    target_text: str,
) -> float:
    """Token-level LCS ratio: lcs_length / len(target_tokens)."""
    gen_tokens = tokenizer.encode(generated_text, add_special_tokens=False)
    tgt_tokens = tokenizer.encode(target_text, add_special_tokens=False)
    if not tgt_tokens:
        return 0.0
    return _lcs_length(gen_tokens, tgt_tokens) / len(tgt_tokens)


def compute_diff_colors(
    target: str,
    sampled: str,
    match_color: tuple = (0, 0, 0, 255),
    mismatch_color: tuple = (200, 120, 120, 255),
) -> list:
    """Per-character colors for `sampled` based on diff with `target`."""
    matcher = difflib.SequenceMatcher(None, target, sampled, autojunk=False)
    colors = [match_color] * len(sampled)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ('replace', 'insert'):
            for j in range(j1, j2):
                colors[j] = mismatch_color
    return colors


# ---------------------------------------------------------------------------
# v3-specific helpers
# ---------------------------------------------------------------------------

def extract_sample_index(sample_id: str) -> str:
    """Return the integer X from an ID of the form 'kY_sampleX'."""
    m = re.search(r"sample(\d+)", str(sample_id))
    return m.group(1) if m else str(sample_id)


def make_row_id(row: pd.Series) -> str:
    """Create a filesystem-safe row identifier."""
    sample_id = str(row.get("id", ""))
    return sample_id.replace("/", "_").replace(" ", "_").replace("\\", "_")


def render_row_images_v3(
    row: pd.Series,
    images_dir: Path,
    row_id: str,
    font_size: int = 56,
    max_width: int = 1200,
    has_ground_truth: bool = False,
    truncate_len: int = 0,
) -> dict:
    """Render content images for a single logical row (v3 sub-row layout).

    Returns a dict: keys are 'prompt', 'gt_prompt' (optional), 'target', 'sample'.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # Learned prompt
    prompt_text = truncate_text(str(row.get("learned_prompt", "")), truncate_len)
    prompt_path = images_dir / f"prompt_{row_id}.png"
    render_text_to_png(prompt_text, str(prompt_path), font_size=font_size, max_width=max_width)
    paths["prompt"] = prompt_path.name

    # Ground-truth prompt (SODA format)
    if has_ground_truth:
        gt_text = truncate_text(str(row.get("ground_truth_prompt", "")), truncate_len)
        gt_path = images_dir / f"gt_prompt_{row_id}.png"
        render_text_to_png(gt_text, str(gt_path), font_size=font_size, max_width=max_width)
        paths["gt_prompt"] = gt_path.name

    # Target output
    target_text = truncate_text(str(row.get("target_text", "")), truncate_len)
    target_path = images_dir / f"target_{row_id}.png"
    render_text_to_png(target_text, str(target_path), font_size=font_size, max_width=max_width)
    paths["target"] = target_path.name

    # (Re-)sampled output — with diff highlighting vs target
    sample_text = truncate_text(str(row.get("sampled_output", "")), truncate_len)
    sample_colors = compute_diff_colors(target_text, sample_text)
    sample_path = images_dir / f"sample_{row_id}.png"
    render_text_to_png(sample_text, str(sample_path), font_size=font_size,
                       max_width=max_width, char_colors=sample_colors)
    paths["sample"] = sample_path.name

    return paths


def _fmt_ppl(val) -> str:
    if val is None:
        return "-"
    try:
        f = float(val)
        if f != f:  # NaN
            return "-"
        return f"{f:.2f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_lcs(val) -> str:
    if val is None:
        return "-"
    try:
        f = float(val)
        if f != f:
            return "-"
        return f"{f:.3f}"
    except (TypeError, ValueError):
        return "-"


def _metric_cell(label: str, value: str) -> str:
    """LaTeX cell content: small label + value, centred in a parbox."""
    return (
        f"\\parbox{{\\linewidth}}{{\\centering"
        f"{{\\scriptsize\\textit{{{label}}}}}\\\\[1pt]"
        f"{{\\small {value}}}}}"
    )


def generate_latex_table_v3(
    data: pd.DataFrame,
    image_paths: dict,
    images_dir_name: str,
    caption: str,
    label: str,
    has_ground_truth: bool = False,
    resample: bool = False,
    fixed_col_width_cm: float = 6.0,
    metrics_col_width_cm: float = 2.5,
) -> str:
    """Generate v3 LaTeX longtable with sub-row layout.

    Column layout: Rank k | Sample ID | Content (images) | Metrics
    """
    # content column width = remaining linewidth
    content_col = (
        f"p{{\\dimexpr\\linewidth-{fixed_col_width_cm}cm\\relax}}"
    )
    metrics_col = f"p{{{metrics_col_width_cm}cm}}"
    col_spec = f"c|c|{content_col}|{metrics_col}"
    num_cols = 4

    sample_label = "Re-sampled LCS" if resample else "LCS"
    headers = [
        r"\textbf{Rank $k$}",
        r"\textbf{Sample}",
        r"\textbf{Content}",
        r"\textbf{Metrics}",
    ]
    header_row = " & ".join(headers) + r" \\"

    img_cmd = (
        lambda fname: (
            rf"\includegraphics[width=\linewidth]{{{images_dir_name}/{fname}}}"
            if fname else "-"
        )
    )

    # Preamble comment explaining reddish text
    lines = [
        r"% Requires: \usepackage{graphicx,xcolor,multirow,longtable}",
        f"% Image directory: {images_dir_name}/",
        "",
        f"\\begin{{longtable}}{{{col_spec}}}",
        f"\\caption{{{escape_latex(caption)}"
        f" {{\\footnotesize\\color[rgb]{{0.55,0.55,0.55}}"
        f" \\texttt{{\\textbackslash n}} = newline,"
        f" \\texttt{{\\textbackslash t}} = tab,"
        f" \\textvisiblespace{{}} = space.}}"
        f" {{\\footnotesize\\color[rgb]{{0.78,0.47,0.47}}"
        f" Reddish text in (Re-)sampled Output = differs from Target.}}"
        f"}} \\label{{{label}}} \\\\",
        r"\hline",
        header_row,
        r"\hline",
        r"\endfirsthead",
        "",
        f"\\multicolumn{{{num_cols}}}{{c}}"
        f"{{{{\\bfseries \\tablename\\ \\thetable{{}} -- continued from previous page}}}} \\\\",
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

    for idx, (_, row) in enumerate(data.iterrows()):
        # --- scalar values ---
        k_raw = row.get("k_value", "")
        if pd.notna(k_raw) and k_raw != "":
            try:
                k_value = f"{int(float(k_raw))}"
            except (ValueError, TypeError):
                k_value = str(k_raw)
        else:
            k_value = "-"

        sample_idx = extract_sample_index(str(row.get("id", "")))

        prompt_ppl_str = _fmt_ppl(row.get("prompt_perplexity"))
        target_ppl_str = _fmt_ppl(row.get("target_perplexity"))
        lcs_str = _fmt_lcs(row.get("lcs_ratio"))

        row_imgs = image_paths.get(idx, {})

        # Build ordered sub-rows: (image_key, metric_cell_latex)
        sub_rows = []
        sub_rows.append((
            "prompt",
            _metric_cell("Prompt PPL", prompt_ppl_str),
        ))
        if has_ground_truth:
            sub_rows.append((
                "gt_prompt",
                _metric_cell("GT Prompt", "-"),
            ))
        sub_rows.append((
            "target",
            _metric_cell("Target PPL", target_ppl_str),
        ))
        sub_rows.append((
            "sample",
            _metric_cell(sample_label, lcs_str),
        ))

        n_sub = len(sub_rows)

        for i, (img_key, metric_cell) in enumerate(sub_rows):
            img_cell = img_cmd(row_imgs.get(img_key, ""))
            if i == 0:
                rank_cell = f"\\multirow{{{n_sub}}}{{*}}{{{k_value}}}"
                sid_cell = f"\\multirow{{{n_sub}}}{{*}}{{{sample_idx}}}"
                lines.append(f"{rank_cell} & {sid_cell} & {img_cell} & {metric_cell} \\\\")
            else:
                lines.append(f" & & {img_cell} & {metric_cell} \\\\")

            # partial rule between sub-rows, full rule after last sub-row
            if i < n_sub - 1:
                lines.append(r"\cline{3-4}")
            else:
                lines.append(r"\hline")

    lines.append(r"\end{longtable}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables with sub-row layout (v3)"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset JSON file")
    parser.add_argument("--wandb-id", type=str, required=True,
                        help="W&B run ID (e.g., entity/project/runs/run_id)")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name for perplexity computation")
    parser.add_argument("--output", type=str, default=None,
                        help="Output LaTeX file path (default: stdout)")
    parser.add_argument("--images-dir", type=str, default=None,
                        help="Directory to save PNG images (default: {output_dir}/table_images/)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Maximum logical rows to include")
    parser.add_argument("--truncate", type=int, default=0,
                        help="Max chars for text columns (0 = no truncation)")
    parser.add_argument("--caption", type=str, default="Optimization Results",
                        help="Table caption text")
    parser.add_argument("--label", type=str, default="tab:optimization_results",
                        help="Table label for references")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Local results directory (overrides W&B config)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for model (auto-detected if not specified)")
    parser.add_argument("--font-size", type=float, default=8,
                        help="Font size in points for PNG rendering (default: 8)")
    parser.add_argument("--image-width", type=float, default=12,
                        help="PNG pixel width in cm — images fill the content column (default: 12)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Resolution in DPI for PNG rendering (default: 300)")
    parser.add_argument("--fixed-col-width", type=float, default=6.0,
                        help="Sum of fixed column widths in cm (rank+sample+metrics+separators) "
                             "used to compute content column width (default: 6.0)")
    parser.add_argument("--metrics-col-width", type=float, default=2.5,
                        help="Width of the metrics column in cm (default: 2.5)")
    parser.add_argument("--resample", action="store_true",
                        help="Re-generate sampled output from learned prompt via greedy decoding")
    parser.add_argument("--resample-max-new-tokens", type=int, default=50,
                        help="Max new tokens for re-sampling (default: 50)")

    args = parser.parse_args()

    # --- Data pipeline ---

    print(f"Parsing W&B ID: {args.wandb_id}", file=sys.stderr)
    entity, project, run_id = parse_wandb_id(args.wandb_id)
    print(f"  Entity: {entity}, Project: {project}, Run ID: {run_id}", file=sys.stderr)

    print(f"Loading dataset: {args.dataset}", file=sys.stderr)
    dataset_info = load_dataset(args.dataset)
    print(f"  Found {len(dataset_info['samples'])} samples", file=sys.stderr)
    print(f"  Has ground truth prompts: {dataset_info['has_ground_truth_prompt']}", file=sys.stderr)

    print("Fetching W&B results...", file=sys.stderr)
    wandb_df, run = fetch_wandb_results(entity, project, run_id)
    print(f"  Found {len(wandb_df)} result rows", file=sys.stderr)

    print("Fetching learned prompts...", file=sys.stderr)
    learned_prompts = fetch_learned_prompts(
        run=run,
        entity=entity,
        project=project,
        run_id=run_id,
        dataset_info=dataset_info,
        results_dir=args.results_dir,
    )
    print(f"  Found {len(learned_prompts)} learned prompts", file=sys.stderr)

    print(f"Loading model: {args.model}", file=sys.stderr)
    model, tokenizer = load_perplexity_model(args.model, args.device)
    print(f"  Model loaded on device: {model.device}", file=sys.stderr)

    # --- Combine data ---
    print("Combining data...", file=sys.stderr)
    combined_data = []

    for sample_id, sample_info in dataset_info["samples"].items():
        row = {
            "id": sample_info["id"],
            "k_value": sample_info.get("k_value"),
            "target_text": sample_info.get("text", ""),
            # target PPL is stored in the dataset JSON under "perplexity"
            "target_perplexity": sample_info.get("perplexity"),
        }

        if dataset_info["has_ground_truth_prompt"]:
            row["ground_truth_prompt"] = sample_info.get("ground_truth_prompt", "")

        if sample_id in learned_prompts:
            prompt_info = learned_prompts[sample_id]
            row["learned_prompt"] = prompt_info.get("text", "")

            tokens = prompt_info.get("tokens", [])
            if args.resample and tokens:
                resampled = resample_from_prompt(
                    model, tokenizer, tokens,
                    max_new_tokens=args.resample_max_new_tokens,
                )
                row["sampled_output"] = resampled
                row["lcs_ratio"] = compute_lcs_ratio(
                    tokenizer, resampled, row["target_text"],
                )
            else:
                if prompt_info.get("generated_text"):
                    row["sampled_output"] = prompt_info["generated_text"]
                if prompt_info.get("lcs_ratio") is not None:
                    row["lcs_ratio"] = prompt_info["lcs_ratio"]

            if prompt_info.get("target_k") is not None and row.get("k_value") is None:
                row["k_value"] = prompt_info["target_k"]

            if tokens:
                row["prompt_perplexity"] = compute_prompt_perplexity(model, tokenizer, tokens)
            else:
                if prompt_info.get("text"):
                    tokens = tokenizer.encode(prompt_info["text"], add_special_tokens=False)
                    row["prompt_perplexity"] = compute_prompt_perplexity(model, tokenizer, tokens)

        combined_data.append(row)

    df = pd.DataFrame(combined_data)
    df["_k_sort"] = df["id"].str.extract(r"k(\d+)", expand=False).astype(float)
    df["_s_sort"] = df["id"].str.extract(r"sample(\d+)", expand=False).astype(float)
    df = df.sort_values(["_k_sort", "_s_sort"]).drop(columns=["_k_sort", "_s_sort"])

    if args.max_rows is not None:
        df = df.head(args.max_rows)

    df = df.reset_index(drop=True)
    print(f"  Combined {len(df)} rows", file=sys.stderr)

    # --- PNG generation ---

    if args.images_dir:
        images_dir = Path(args.images_dir)
    elif args.output:
        images_dir = Path(args.output).parent / "table_images"
    else:
        images_dir = Path("table_images")

    images_dir.mkdir(parents=True, exist_ok=True)
    images_dir_name = images_dir.name

    font_size_px = round(args.font_size * args.dpi / 72)
    max_width_px = round(args.image_width * args.dpi / 2.54)

    print(f"Rendering PNGs to: {images_dir}", file=sys.stderr)
    print(
        f"  image_width={args.image_width}cm -> {max_width_px}px, "
        f"font_size={args.font_size}pt -> {font_size_px}px @ {args.dpi}dpi",
        file=sys.stderr,
    )

    image_paths = {}
    for idx, (_, row) in enumerate(df.iterrows()):
        row_id = make_row_id(row)
        paths = render_row_images_v3(
            row=row,
            images_dir=images_dir,
            row_id=row_id,
            font_size=font_size_px,
            max_width=max_width_px,
            has_ground_truth=dataset_info["has_ground_truth_prompt"],
            truncate_len=args.truncate,
        )
        image_paths[idx] = paths
        print(f"  [{idx + 1}/{len(df)}] Rendered {row_id}", file=sys.stderr)

    # --- LaTeX generation ---

    print("Generating LaTeX table (v3 sub-row layout)...", file=sys.stderr)
    latex_output = generate_latex_table_v3(
        data=df,
        image_paths=image_paths,
        images_dir_name=images_dir_name,
        caption=args.caption,
        label=args.label,
        has_ground_truth=dataset_info["has_ground_truth_prompt"],
        resample=args.resample,
        fixed_col_width_cm=args.fixed_col_width,
        metrics_col_width_cm=args.metrics_col_width,
    )

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
