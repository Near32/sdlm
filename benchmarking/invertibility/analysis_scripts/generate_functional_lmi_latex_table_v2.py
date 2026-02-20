#!/usr/bin/env python3
"""Generate LaTeX tables with PNG-rendered text columns.

v2 of the LaTeX table generator. Instead of embedding escaped text directly in
LaTeX cells (which fails for multilingual/special characters), this version
renders the Learned Prompt, Target Output, and Sampled Output columns as PNG
images and uses \\includegraphics in the LaTeX table.

Reuses the data pipeline from generate_functional_lmi_latex_table.py.
"""

import argparse
import difflib
import sys
from pathlib import Path

import pandas as pd

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


def compute_diff_colors(
    target: str,
    sampled: str,
    match_color: tuple = (0, 0, 0, 255),
    mismatch_color: tuple = (200, 120, 120, 255),
) -> list[tuple]:
    """Per-character colors for `sampled` based on diff with `target`.

    Uses SequenceMatcher opcodes:
    - 'equal'   -> match_color (black)
    - 'replace' -> mismatch_color (reddish)
    - 'insert'  -> mismatch_color (reddish) -- chars in sampled not in target
    - 'delete'  -> (skipped, those chars are in target only)
    """
    matcher = difflib.SequenceMatcher(None, target, sampled, autojunk=False)
    colors = [match_color] * len(sampled)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ('replace', 'insert'):
            for j in range(j1, j2):
                colors[j] = mismatch_color
    return colors


def render_row_images(
    row: pd.Series,
    images_dir: Path,
    row_id: str,
    font_size: int = 56,
    max_width: int = 400,
    has_ground_truth: bool = False,
    truncate_len: int = 0,
) -> dict[str, str]:
    """Render text columns for a single row as PNG images.

    Args:
        row: DataFrame row with text fields.
        images_dir: Directory to save PNG files.
        row_id: Unique identifier for filenames (e.g., "k1_sample0").
        font_size: Font size for rendering.
        max_width: Max pixel width (enables word wrap).
        has_ground_truth: Whether to render ground truth prompt column.
        truncate_len: Max chars for text (0 = no truncation).

    Returns:
        Dictionary mapping column name to relative PNG path (from images_dir parent).
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # Target output
    target_text = truncate_text(str(row.get("target_text", "")), truncate_len)
    target_path = images_dir / f"target_{row_id}.png"
    render_text_to_png(target_text, str(target_path), font_size=font_size, max_width=max_width)
    paths["target"] = str(target_path.name)

    # Learned prompt
    prompt_text = truncate_text(str(row.get("learned_prompt", "")), truncate_len)
    prompt_path = images_dir / f"prompt_{row_id}.png"
    render_text_to_png(prompt_text, str(prompt_path), font_size=font_size, max_width=max_width)
    paths["prompt"] = str(prompt_path.name)

    # Sampled output — with diff highlighting
    sample_text = truncate_text(str(row.get("sampled_output", "")), truncate_len)
    sample_colors = compute_diff_colors(target_text, sample_text)
    sample_path = images_dir / f"sample_{row_id}.png"
    render_text_to_png(sample_text, str(sample_path), font_size=font_size,
                       max_width=max_width, char_colors=sample_colors)
    paths["sample"] = str(sample_path.name)

    # Ground truth prompt (if SODA format)
    if has_ground_truth:
        gt_text = truncate_text(str(row.get("ground_truth_prompt", "")), truncate_len)
        gt_path = images_dir / f"gt_prompt_{row_id}.png"
        render_text_to_png(gt_text, str(gt_path), font_size=font_size, max_width=max_width)
        paths["gt_prompt"] = str(gt_path.name)

    return paths


def make_row_id(row: pd.Series) -> str:
    """Create a filesystem-safe row identifier."""
    sample_id = str(row.get("id", ""))
    # Replace problematic characters
    row_id = sample_id.replace("/", "_").replace(" ", "_").replace("\\", "_")
    return row_id


def generate_latex_table_v2(
    data: pd.DataFrame,
    image_paths: dict[str, dict[str, str]],
    images_dir_name: str,
    caption: str,
    label: str,
    has_ground_truth: bool = False,
    image_width: str = "2cm",
) -> str:
    """Generate LaTeX longtable with \\includegraphics for text columns.

    Args:
        data: DataFrame with table data.
        image_paths: Dict mapping row index to dict of column->image filename.
        images_dir_name: Name of the images directory (for \\graphicspath).
        caption: Table caption.
        label: Table label for references.
        has_ground_truth: Whether to include ground truth prompt column.
        image_width: LaTeX dimension for \\includegraphics width.

    Returns:
        Complete LaTeX longtable string.
    """
    img_cmd = (
        lambda fname: rf"\includegraphics[width={image_width}]{{{images_dir_name}/{fname}}}"
    )

    if has_ground_truth:
        col_spec = "c|c|c|c|c|c|c|c"
        num_cols = 8
        headers = [
            r"\textbf{ID}",
            r"\textbf{Rank $k$}",
            r"\textbf{Target Output}",
            r"\textbf{GT Prompt}",
            r"\textbf{Learned Prompt}",
            r"\textbf{Prompt PPL}",
            r"\textbf{Sampled Output}",
            r"\textbf{LCS}",
        ]
    else:
        col_spec = "c|c|c|c|c|c|c"
        num_cols = 7
        headers = [
            r"\textbf{ID}",
            r"\textbf{Rank $k$}",
            r"\textbf{Target Output}",
            r"\textbf{Learned Prompt}",
            r"\textbf{Prompt PPL}",
            r"\textbf{Sampled Output}",
            r"\textbf{LCS}",
        ]

    header_row = " & ".join(headers) + r" \\"

    lines = [
        r"% Requires: \usepackage{graphicx}, \usepackage{xcolor}",
        f"% Image directory: {images_dir_name}/",
        "",
        f"\\begin{{longtable}}{{{col_spec}}}",
        f"\\caption{{{escape_latex(caption)}"
        f" {{\\footnotesize\\color[rgb]{{0.55,0.55,0.55}}"
        f" \\texttt{{\\textbackslash n}} = newline,"
        f" \\texttt{{\\textbackslash t}} = tab,"
        f" \\textvisiblespace{{}} = space.}}"
        f" {{\\footnotesize\\color[rgb]{{0.78,0.47,0.47}}"
        f" Reddish text in Sampled Output = differs from Target.}}"
        f"}} \\label{{{label}}} \\\\",
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

    for idx, (_, row) in enumerate(data.iterrows()):
        sample_id = escape_latex(str(row.get("id", "")))

        k_value = row.get("k_value", "")
        if pd.notna(k_value):
            k_value = f"{int(k_value)}" if float(k_value) == int(float(k_value)) else f"{k_value:.1f}"
        else:
            k_value = "-"

        prompt_ppl = row.get("prompt_perplexity", float("nan"))
        if pd.notna(prompt_ppl) and not (isinstance(prompt_ppl, float) and (prompt_ppl != prompt_ppl)):
            prompt_ppl_str = f"{prompt_ppl:.2f}"
        else:
            prompt_ppl_str = "-"

        lcs_ratio = row.get("lcs_ratio", float("nan"))
        if pd.notna(lcs_ratio):
            lcs_str = f"{lcs_ratio:.3f}"
        else:
            lcs_str = "-"

        row_imgs = image_paths.get(idx, {})
        target_cell = img_cmd(row_imgs["target"]) if "target" in row_imgs else "-"
        prompt_cell = img_cmd(row_imgs["prompt"]) if "prompt" in row_imgs else "-"
        sample_cell = img_cmd(row_imgs["sample"]) if "sample" in row_imgs else "-"

        if has_ground_truth:
            gt_cell = img_cmd(row_imgs["gt_prompt"]) if "gt_prompt" in row_imgs else "-"
            row_values = [sample_id, k_value, target_cell, gt_cell, prompt_cell, prompt_ppl_str, sample_cell, lcs_str]
        else:
            row_values = [sample_id, k_value, target_cell, prompt_cell, prompt_ppl_str, sample_cell, lcs_str]

        lines.append(" & ".join(row_values) + r" \\ \hline")

    lines.append(r"\end{longtable}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables with PNG-rendered text columns (v2)"
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
        "--images-dir",
        type=str,
        default=None,
        help="Directory to save PNG images (default: {output_dir}/table_images/)",
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
        default=0,
        help="Max chars for text columns (default: 0 = no truncation)",
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
    parser.add_argument(
        "--font-size",
        type=float,
        default=8,
        help="Font size in points (default: 8)",
    )
    parser.add_argument(
        "--image-width",
        type=float,
        default=2,
        help="Image width in cm — controls both PNG pixel width and LaTeX \\includegraphics (default: 2)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution in DPI for PNG rendering (default: 300)",
    )

    args = parser.parse_args()

    # --- Data pipeline (same as v1) ---

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
    print("Fetching W&B results...", file=sys.stderr)
    wandb_df, run = fetch_wandb_results(entity, project, run_id)
    print(f"  Found {len(wandb_df)} result rows", file=sys.stderr)

    # Fetch learned prompts
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

    # Load model for perplexity computation
    print(f"Loading model for perplexity: {args.model}", file=sys.stderr)
    model, tokenizer = load_perplexity_model(args.model, args.device)
    print(f"  Model loaded on device: {model.device}", file=sys.stderr)

    # Combine data
    print("Combining data...", file=sys.stderr)
    combined_data = []

    for sample_id, sample_info in dataset_info["samples"].items():
        row = {
            "id": sample_info["id"],
            "k_value": sample_info.get("k_value"),
            "target_text": sample_info.get("text", ""),
        }

        if dataset_info["has_ground_truth_prompt"]:
            row["ground_truth_prompt"] = sample_info.get("ground_truth_prompt", "")

        if sample_id in learned_prompts:
            prompt_info = learned_prompts[sample_id]
            row["learned_prompt"] = prompt_info.get("text", "")

            if prompt_info.get("generated_text"):
                row["sampled_output"] = prompt_info["generated_text"]
            if prompt_info.get("lcs_ratio") is not None:
                row["lcs_ratio"] = prompt_info["lcs_ratio"]
            if prompt_info.get("target_k") is not None and row.get("k_value") is None:
                row["k_value"] = prompt_info["target_k"]

            tokens = prompt_info.get("tokens", [])
            if tokens:
                row["prompt_perplexity"] = compute_prompt_perplexity(model, tokenizer, tokens)
            else:
                if prompt_info.get("text"):
                    tokens = tokenizer.encode(prompt_info["text"], add_special_tokens=False)
                    row["prompt_perplexity"] = compute_prompt_perplexity(model, tokenizer, tokens)

        combined_data.append(row)

    df = pd.DataFrame(combined_data)
    # Sort by k value (numeric), then sample number (numeric)
    # IDs look like "k1_sample0", "k11_sample2", etc.
    df["_k_sort"] = df["id"].str.extract(r"k(\d+)", expand=False).astype(float)
    df["_s_sort"] = df["id"].str.extract(r"sample(\d+)", expand=False).astype(float)
    df = df.sort_values(["_k_sort", "_s_sort"])
    df = df.drop(columns=["_k_sort", "_s_sort"])

    if args.max_rows is not None:
        df = df.head(args.max_rows)

    # Reset index for consistent image_paths keys
    df = df.reset_index(drop=True)

    print(f"  Combined {len(df)} rows", file=sys.stderr)

    # --- PNG generation (new in v2) ---

    # Determine images directory
    if args.images_dir:
        images_dir = Path(args.images_dir)
    elif args.output:
        images_dir = Path(args.output).parent / "table_images"
    else:
        images_dir = Path("table_images")

    images_dir.mkdir(parents=True, exist_ok=True)
    images_dir_name = images_dir.name

    # Convert physical units to pixels
    font_size_px = round(args.font_size * args.dpi / 72)
    max_width_px = round(args.image_width * args.dpi / 2.54)
    image_width_latex = f"{args.image_width}cm"

    print(f"Rendering text columns as PNGs to: {images_dir}", file=sys.stderr)
    print(f"  image_width={args.image_width}cm -> {max_width_px}px, font_size={args.font_size}pt -> {font_size_px}px @ {args.dpi}dpi", file=sys.stderr)

    image_paths = {}
    for idx, (_, row) in enumerate(df.iterrows()):
        row_id = make_row_id(row)
        paths = render_row_images(
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

    print(f"  Generated {len(image_paths) * (4 if dataset_info['has_ground_truth_prompt'] else 3)} PNG images", file=sys.stderr)

    # --- LaTeX generation (v2) ---

    print("Generating LaTeX table (v2 with images)...", file=sys.stderr)
    latex_output = generate_latex_table_v2(
        data=df,
        image_paths=image_paths,
        images_dir_name=images_dir_name,
        caption=args.caption,
        label=args.label,
        has_ground_truth=dataset_info["has_ground_truth_prompt"],
        image_width=image_width_latex,
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
