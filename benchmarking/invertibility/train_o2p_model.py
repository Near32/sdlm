#!/usr/bin/env python3
"""
O2P (Output-to-Prompt) Inverse Model Training Script.

Trains a T5-based inverse model for a given subject LLM.

Example usage:
    python train_o2p_model.py \
        --llm_model_name "HuggingFaceTB/SmolLM2-135M" \
        --t5_model_name "t5-base" \
        --bottleneck_dim 4096 \
        --num_tokens 64 \
        --dataset_size 400000 \
        --num_epochs 10 \
        --output_dir ./o2p_checkpoints/smollm2-135m \
        --wandb_project "o2p-training" \
        --upload_artifact
"""

import argparse
import logging
import os
import sys
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train O2P inverse model for LLM prompt inversion"
    )

    # Model configuration
    parser.add_argument(
        "--llm_model_name", type=str, required=True,
        help="HuggingFace model name/path for the subject LLM"
    )
    parser.add_argument(
        "--t5_model_name", type=str, default="t5-base",
        help="T5 model name for the inverse model (default: t5-base)"
    )
    parser.add_argument(
        "--t5_tokenizer_name", type=str, default=None,
        help="T5 tokenizer name (defaults to t5_model_name)"
    )
    parser.add_argument(
        "--bottleneck_dim", type=int, default=4096,
        help="Dimension of the bottleneck layer (default: 4096)"
    )
    parser.add_argument(
        "--num_tokens", type=int, default=64,
        help="Number of tokens for embedding transformation (default: 64)"
    )
    parser.add_argument(
        "--unigram_beta", type=float, default=0.01,
        help="Beta for unigram adaptation EMA (default: 0.01)"
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset_size", type=int, default=400000,
        help="Number of training samples to generate (default: 400000)"
    )
    parser.add_argument(
        "--min_seq_length", type=int, default=1,
        help="Minimum sequence length for random samples (default: 1)"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=24,
        help="Maximum sequence length for random samples (default: 24)"
    )
    parser.add_argument(
        "--max_length", type=int, default=32,
        help="Maximum tokenized sequence length (default: 32)"
    )
    parser.add_argument(
        "--val_split", type=float, default=0.1,
        help="Validation split fraction (default: 0.1)"
    )

    # Training configuration
    parser.add_argument(
        "--num_epochs", type=int, default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Training batch size (default: 64)"
    )
    parser.add_argument(
        "--mini_batch_size", type=int, default=64,
        help="Mini-batch size for memory-efficient processing (default: 64)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5,
        help="Learning rate (default: 5e-5)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01,
        help="Weight decay (default: 0.01)"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=0,
        help="Number of warmup steps (default: 0)"
    )
    parser.add_argument(
        "--save_steps", type=int, default=1000,
        help="Checkpoint save frequency (default: 1000)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of dataloader workers (default: 4)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )

    # Output configuration
    parser.add_argument(
        "--output_dir", type=str, default="./o2p_checkpoints",
        help="Directory to save model checkpoints (default: ./o2p_checkpoints)"
    )

    # W&B configuration
    parser.add_argument(
        "--wandb_project", type=str, default="o2p-training",
        help="W&B project name (default: o2p-training)"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None,
        help="W&B entity (username or team)"
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None,
        help="W&B run name (auto-generated if not provided)"
    )
    parser.add_argument(
        "--upload_artifact", action="store_true",
        help="Upload trained model as W&B artifact"
    )
    parser.add_argument(
        "--no_wandb", action="store_true",
        help="Disable W&B logging"
    )

    # Device configuration
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--llm_precision", type=str, default="half",
        choices=["full", "half", "bfloat16"],
        help="Precision for LLM weights (default: half). LLM is frozen so half saves memory."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Initialize W&B
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb

            # Generate run name if not provided
            if args.wandb_run_name is None:
                llm_short = args.llm_model_name.split("/")[-1]
                t5_short = args.t5_model_name.split("/")[-1]
                args.wandb_run_name = f"o2p_{llm_short}_{t5_short}"

            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                config=vars(args),
            )
            logger.info(f"W&B initialized: {args.wandb_project}/{args.wandb_run_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
            use_wandb = False

    # Import O2P modules
    from baselines.o2p.o2p_model import LLMInversionModel
    from baselines.o2p.o2p_utils import train_inversion_model

    # Set T5 tokenizer name
    t5_tokenizer_name = args.t5_tokenizer_name or args.t5_model_name

    # Create O2P model
    logger.info(f"Creating O2P model with LLM: {args.llm_model_name}, T5: {args.t5_model_name}")
    model = LLMInversionModel(
        t5_model_name=args.t5_model_name,
        t5_tokenizer_name=t5_tokenizer_name,
        llm_model_name=args.llm_model_name,
        unigram_beta=args.unigram_beta,
        num_tokens=args.num_tokens,
        bottleneck_dim=args.bottleneck_dim,
    )
    logger.info("O2P model created")

    # Convert LLM to specified precision (LLM is frozen, so this saves memory)
    if args.llm_precision == "half":
        model.llm = model.llm.half()
        logger.info("LLM converted to float16")
    elif args.llm_precision == "bfloat16":
        model.llm = model.llm.to(torch.bfloat16)
        logger.info("LLM converted to bfloat16")

    # Train model
    logger.info("Starting training...")
    trained_model = train_inversion_model(
        model=model,
        seed=args.seed,
        dataset_size=args.dataset_size,
        min_seq_length=args.min_seq_length,
        max_seq_length=args.max_seq_length,
        max_length=args.max_length,
        val_split=args.val_split,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        num_epochs=args.num_epochs,
        save_steps=args.save_steps,
        warmup_steps=args.warmup_steps,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_wandb=use_wandb,
    )

    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model")
    trained_model.save_pretrained(final_model_path)
    logger.info(f"Saved final model to {final_model_path}")

    # Upload artifact if requested
    if use_wandb and args.upload_artifact:
        try:
            import wandb

            # Upload best model as artifact
            best_model_path = os.path.join(args.output_dir, "best_model")
            if os.path.exists(best_model_path):
                artifact_name = f"o2p-{args.llm_model_name.split('/')[-1]}"
                artifact = wandb.Artifact(
                    name=artifact_name,
                    type="model",
                    description=f"O2P inverse model for {args.llm_model_name}",
                    metadata={
                        "llm_model_name": args.llm_model_name,
                        "t5_model_name": args.t5_model_name,
                        "bottleneck_dim": args.bottleneck_dim,
                        "num_tokens": args.num_tokens,
                        "dataset_size": args.dataset_size,
                        "num_epochs": args.num_epochs,
                    }
                )
                artifact.add_dir(best_model_path)
                wandb.log_artifact(artifact)
                logger.info(f"Uploaded best model as W&B artifact: {artifact_name}")
            else:
                logger.warning(f"Best model path not found: {best_model_path}")
        except Exception as e:
            logger.error(f"Failed to upload W&B artifact: {e}")

    # Finish W&B run
    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except:
            pass

    logger.info("Training complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
