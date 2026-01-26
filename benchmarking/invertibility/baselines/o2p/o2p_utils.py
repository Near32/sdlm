"""
O2P Training Utilities.

Training utilities for the O2P (Output-to-Prompt) inverse model including:
- Random token sequence generation
- Dataset and dataloader creation
- Training loop with W&B logging
"""

import os
import random
import logging
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def generate_random_token_sequences(
    llm_tokenizer,
    dataset_size: int,
    min_length: int,
    max_length: int,
    seed: Optional[int] = None
) -> List[str]:
    """
    Generate random token sequences from the LLM vocabulary
    with lengths evenly distributed between min_length and max_length.

    Args:
        llm_tokenizer: The tokenizer for the LLM
        dataset_size: Total number of sequences to generate
        min_length: Minimum sequence length (in tokens)
        max_length: Maximum sequence length (in tokens)
        seed: Random seed for reproducibility

    Returns:
        List of decoded token sequences as strings
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    logger.info(f"Generating {dataset_size} random token sequences...")

    # Get vocabulary size
    vocab_size = llm_tokenizer.vocab_size
    logger.info(f"Vocabulary size: {vocab_size}")

    # Reserve some special tokens
    special_token_ids = set()
    for token in llm_tokenizer.special_tokens_map.values():
        if isinstance(token, str):
            special_token_ids.add(llm_tokenizer.convert_tokens_to_ids(token))
        elif isinstance(token, list):
            for t in token:
                special_token_ids.add(llm_tokenizer.convert_tokens_to_ids(t))

    # Create the token ID list (excluding special tokens)
    valid_token_ids = [i for i in range(vocab_size) if i not in special_token_ids]

    # Generate sequences with evenly distributed lengths
    sequences = []

    # Equal number of examples for each length
    examples_per_length = dataset_size // (max_length - min_length + 1)

    for length in range(min_length, max_length + 1):
        for _ in range(examples_per_length):
            # Sample random token IDs
            token_ids = np.random.choice(valid_token_ids, size=length, replace=True).tolist()

            # Decode to text
            try:
                text = llm_tokenizer.decode(token_ids, skip_special_tokens=True)

                if text.strip():
                    sequences.append(text)
                else:
                    # If empty, try again with different tokens
                    token_ids = np.random.choice(valid_token_ids, size=length, replace=True).tolist()
                    text = llm_tokenizer.decode(token_ids, skip_special_tokens=True)
                    if text.strip():
                        sequences.append(text)
            except Exception as e:
                logger.warning(f"Error decoding tokens: {e}")
                continue

    # Fill any remaining slots to reach dataset_size
    remaining = dataset_size - len(sequences)
    if remaining > 0:
        for _ in range(remaining):
            length = random.randint(min_length, max_length)
            token_ids = np.random.choice(valid_token_ids, size=length, replace=True).tolist()
            try:
                text = llm_tokenizer.decode(token_ids, skip_special_tokens=True)
                if text.strip():
                    sequences.append(text)
            except:
                pass

    # Shuffle the sequences
    random.shuffle(sequences)

    # Trim to dataset_size
    sequences = sequences[:dataset_size]

    logger.info(f"Generated {len(sequences)} sequences")

    # Log some examples
    logger.info("Example sequences:")
    for i in range(min(5, len(sequences))):
        logger.info(f"  {i+1}. '{sequences[i]}'")

    return sequences


class LLMInversionDataset(Dataset):
    """
    Dataset for LLM inversion training.

    Each sample is a string that will be:
    1. Tokenized and passed through the LLM
    2. The output logits will be used to reconstruct the original input
    """

    def __init__(
        self,
        text_samples: List[str],
        llm_tokenizer,
        t5_tokenizer,
        max_length: int
    ):
        """
        Args:
            text_samples: List of text samples
            llm_tokenizer: Tokenizer for the LLM
            t5_tokenizer: Tokenizer for T5
            max_length: Maximum sequence length
        """
        self.text_samples = text_samples
        self.llm_tokenizer = llm_tokenizer
        self.t5_tokenizer = t5_tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.text_samples)

    def __getitem__(self, idx: int) -> dict:
        text = self.text_samples[idx]

        # Tokenize for LLM input
        llm_tokens = self.llm_tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # Tokenize for T5 target (expected output)
        t5_tokens = self.t5_tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "llm_input_ids": llm_tokens["input_ids"].squeeze(),
            "llm_attention_mask": llm_tokens["attention_mask"].squeeze(),
            "t5_target_ids": t5_tokens["input_ids"].squeeze(),
            "t5_target_attention_mask": t5_tokens["attention_mask"].squeeze(),
            "original_text": text
        }


def create_dataloaders(
    text_samples: List[str],
    llm_tokenizer,
    t5_tokenizer,
    batch_size: int,
    max_length: int,
    val_split: float = 0.1,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders from text samples.

    Args:
        text_samples: List of text samples
        llm_tokenizer: Tokenizer for the LLM
        t5_tokenizer: Tokenizer for T5
        batch_size: Batch size
        max_length: Maximum sequence length
        val_split: Validation split fraction
        num_workers: Number of dataloader workers

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Split into train and validation
    val_size = int(len(text_samples) * val_split)
    train_samples = text_samples[val_size:]
    val_samples = text_samples[:val_size]

    # Create datasets
    train_dataset = LLMInversionDataset(
        train_samples, llm_tokenizer, t5_tokenizer, max_length
    )
    val_dataset = LLMInversionDataset(
        val_samples, llm_tokenizer, t5_tokenizer, max_length
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def train_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    use_wandb: bool = True
) -> float:
    """
    Train for one epoch.

    Args:
        model: The LLMInversionModel
        dataloader: Training dataloader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        epoch: Current epoch number
        use_wandb: Whether to log to W&B

    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}")

    for step, batch in enumerate(progress_bar):
        # Move data to device
        llm_input_ids = batch["llm_input_ids"].to(device)
        llm_attention_mask = batch["llm_attention_mask"].to(device)
        t5_target_ids = batch["t5_target_ids"].to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            input_ids=llm_input_ids,
            attention_mask=llm_attention_mask,
            labels=t5_target_ids
        )

        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()
        scheduler.step()

        # Update progress bar
        progress_bar.set_postfix({"train loss": loss.item()})

        # Log every 100 steps
        if (step + 1) % 100 == 0:
            logger.info(f"Epoch {epoch}, Step {step+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
            if use_wandb:
                try:
                    import wandb
                    wandb.log({
                        "train/step_loss": loss.item(),
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/global_step": (epoch - 1) * len(dataloader) + step + 1,
                    })
                except:
                    pass

    return total_loss / len(dataloader)


def validate(
    model,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int
) -> float:
    """
    Validate the model.

    Args:
        model: The LLMInversionModel
        dataloader: Validation dataloader
        device: Device to run on
        epoch: Current epoch number

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Validation Epoch {epoch}")

        for _, batch in enumerate(progress_bar):
            # Move data to device
            llm_input_ids = batch["llm_input_ids"].to(device)
            llm_attention_mask = batch["llm_attention_mask"].to(device)
            t5_target_ids = batch["t5_target_ids"].to(device)

            # Forward pass
            outputs = model(
                input_ids=llm_input_ids,
                attention_mask=llm_attention_mask,
                labels=t5_target_ids
            )
            loss = outputs.loss
            total_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({"val loss": loss.item()})

    return total_loss / len(dataloader)


def train_inversion_model(
    model,
    seed: int = 42,
    dataset_size: int = 5000,
    min_seq_length: int = 1,
    max_seq_length: int = 24,
    max_length: int = 32,
    val_split: float = 0.1,
    output_dir: str = "./inversion_model_checkpoints",
    batch_size: int = 64,
    mini_batch_size: int = 64,
    num_epochs: int = 5,
    save_steps: int = 1000,
    warmup_steps: int = 0,
    num_workers: int = 4,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    use_wandb: bool = True,
):
    """
    Train the LLM inversion model on a dataset of randomly generated token sequences.

    Args:
        model: The LLMInversionModel to train
        seed: Random seed
        dataset_size: Number of training samples to generate
        min_seq_length: Minimum sequence length for generation
        max_seq_length: Maximum sequence length for generation
        max_length: Maximum tokenized sequence length
        val_split: Validation split fraction
        output_dir: Directory to save checkpoints
        batch_size: Training batch size
        mini_batch_size: Mini-batch size for memory-efficient processing
        num_epochs: Number of training epochs
        save_steps: Checkpoint save frequency
        warmup_steps: Number of warmup steps
        num_workers: Number of dataloader workers
        learning_rate: Learning rate
        weight_decay: Weight decay
        use_wandb: Whether to use W&B logging

    Returns:
        Trained model
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Generate random token sequences for training
    logger.info(f"Generating random token sequences dataset of size {dataset_size}...")
    text_samples = generate_random_token_sequences(
        model.llm_tokenizer,
        dataset_size=dataset_size,
        min_length=min_seq_length,
        max_length=max_seq_length,
        seed=seed
    )

    # Move model to device
    model = model.to(device)
    model.mini_batch_size = mini_batch_size

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        text_samples=text_samples,
        llm_tokenizer=model.llm_tokenizer,
        t5_tokenizer=model.tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        val_split=val_split,
        num_workers=num_workers
    )

    # Freeze the LLM
    for param in model.llm.parameters():
        param.requires_grad = False

    # Only train the T5 model and transformation layers
    optimizer_params = [
        {"params": model.encoder_decoder.parameters()},
        {"params": model.embedding_transform.parameters()},
    ]

    # Create optimizer
    optimizer = optim.AdamW(
        optimizer_params,
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Create learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs")

    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        # Train for one epoch
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            use_wandb=use_wandb
        )

        # Validate
        val_loss = validate(
            model=model,
            dataloader=val_loader,
            device=device,
            epoch=epoch
        )

        # Log metrics
        logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if use_wandb:
            try:
                import wandb
                wandb.log({
                    "train/epoch_loss": train_loss,
                    "val/epoch_loss": val_loss,
                    "epoch": epoch,
                })
            except:
                pass

        # Save checkpoint if it's the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"New best validation loss: {best_val_loss:.4f}")
            model_path = os.path.join(output_dir, "best_model")

            # Save the model
            model.save_pretrained(model_path)
            logger.info(f"Saved best model to {model_path}")

        # Save periodic checkpoints
        if (epoch * len(train_loader)) % save_steps < len(train_loader):
            model_path = os.path.join(output_dir, f"checkpoint-epoch-{epoch}")
            model.save_pretrained(model_path)
            logger.info(f"Saved checkpoint to {model_path}")

        global_step += len(train_loader)

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    logger.info("Training completed!")
    return model


__all__ = [
    "generate_random_token_sequences",
    "LLMInversionDataset",
    "create_dataloaders",
    "train_epoch",
    "validate",
    "train_inversion_model",
]
