"""
Batch entrypoint for product-of-speaker prompt optimization.
"""

from batch_optimize_main import (
    apply_loss_arg_expansions,
    batch_optimize,
    build_parser,
    str2list,
)


def main():
    parser = build_parser(
        description="Batch optimize prompts with product-of-speaker assembly"
    )
    parser.set_defaults(
        assemble_strategy="product_of_speaker",
        wandb_project="prompt-optimization-batch-pos",
    )
    args = parser.parse_args()
    apply_loss_arg_expansions(args)

    target_indices = None
    if args.target_indices:
        target_indices = [int(idx) for idx in args.target_indices.split(",")]

    metric_groups = str2list(args.metric_groups)
    _ = str2list(args.skip_metric_groups)

    config = vars(args)
    batch_optimize(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        config=config,
        num_workers=args.num_workers,
        target_indices=target_indices,
        metric_groups=metric_groups,
    )


if __name__ == "__main__":
    main()
