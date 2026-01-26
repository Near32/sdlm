"""
O2P (Output-to-Prompt) Inverse Model.

Adapted from Zhang et al., 2024 - "O2P: A Method for Soft Prompt Inversion"
Implementation based on SODA_ICML_Experiment_Notebook.py.

This module provides the LLMInversionModel class which uses a T5 encoder-decoder
to invert LLM outputs back to prompts.
"""

import os
import json
import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List, Union


class LLMInversionModel(nn.Module):
    """
    Inverse model that takes logits from a causal LM and predicts the original input
    that would have generated those logits.

    Uses a T5 encoder-decoder with:
    - Embedding transform layer
    - Unigram adaptation mechanism
    - Optional external LLM for model caching compatibility

    Args:
        t5_model_name: Name or path of the T5 model
        t5_tokenizer_name: Name or path of the T5 tokenizer
        llm_model_name: Name or path of the subject LLM (used if external_llm not provided)
        external_llm: Pre-loaded external LLM model (for model caching compatibility)
        external_llm_tokenizer: Pre-loaded external LLM tokenizer
        unigram_beta: Beta for unigram adaptation EMA
        num_tokens: Number of tokens for the embedding transformation
        bottleneck_dim: Dimension of the bottleneck layer
    """

    def __init__(
        self,
        t5_model_name: str = "t5-base",
        t5_tokenizer_name: str = "t5-base",
        llm_model_name: Optional[str] = None,
        external_llm: Optional[nn.Module] = None,
        external_llm_tokenizer: Optional[AutoTokenizer] = None,
        unigram_beta: float = 0.01,
        num_tokens: int = 64,
        bottleneck_dim: int = 4096,
    ):
        super().__init__()

        # Store config for save/load
        self._llm_model_name = llm_model_name

        # Load T5 encoder-decoder
        self.encoder_decoder = AutoModelForSeq2SeqLM.from_pretrained(t5_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(t5_tokenizer_name)

        # Use external LLM if provided, otherwise load from name
        if external_llm is not None:
            self.llm = external_llm
            self._llm_model_name = getattr(external_llm.config, '_name_or_path', llm_model_name)
        elif llm_model_name is not None:
            self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
        else:
            raise ValueError("Either llm_model_name or external_llm must be provided")

        # Use external tokenizer if provided
        if external_llm_tokenizer is not None:
            self.llm_tokenizer = external_llm_tokenizer
        elif llm_model_name is not None:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        else:
            raise ValueError("Either llm_model_name or external_llm_tokenizer must be provided")

        # Ensure pad token is set
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        # Freeze the LLM
        for param in self.llm.parameters():
            param.requires_grad = False
        self.llm.eval()

        # Get dimensions
        self.encoder_hidden_dim = self.encoder_decoder.config.d_model
        self.bottleneck_dim = bottleneck_dim
        self.num_tokens = num_tokens
        self.mini_batch_size = None

        # Calculate padding to ensure dimensions align
        self.num_zeros_to_add = (num_tokens - (self.llm.config.vocab_size % num_tokens)) % num_tokens

        # Prepare unigram adaptation mechanism
        self.unigram_beta = unigram_beta
        self.unigram = nn.Parameter(
            torch.zeros(
                (1, self.llm.config.vocab_size + self.num_zeros_to_add),
                dtype=torch.float32
            ),
            requires_grad=False
        )

        # Prepare word embeddings with proper reshaping
        word_embeddings = self.encoder_decoder.encoder.embed_tokens.weight.detach().clone()
        self.word_embeddings = self._prepare_word_embeddings(word_embeddings, num_tokens)

        # Prepare transformation from reduced logits to encoder space
        self.embedding_transform = nn.Sequential(
            nn.Linear(self.encoder_hidden_dim, bottleneck_dim),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(bottleneck_dim, self.encoder_hidden_dim)
        )

    def _prepare_word_embeddings(self, word_embeddings: torch.Tensor, num_tokens: int) -> nn.Parameter:
        """Prepare word embeddings with proper padding and reshaping."""
        num_zeros_to_add = (num_tokens - (word_embeddings.shape[0] % num_tokens)) % num_tokens

        word_embedding_zeros = torch.zeros(
            (num_zeros_to_add, word_embeddings.shape[1]),
            dtype=torch.float32,
            device=word_embeddings.device
        )

        padded_word_embeddings = torch.cat((word_embeddings, word_embedding_zeros), dim=0)
        reshaped_embeddings = padded_word_embeddings.reshape(
            (num_tokens, -1, word_embeddings.shape[1])
        )

        return nn.Parameter(reshaped_embeddings, requires_grad=False)

    def get_llm_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get logits from the LLM model for the last token of each sequence."""
        with torch.no_grad():
            outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)

            # Get the logits for the last token of each sequence
            batch_size = input_ids.shape[0]
            last_token_indices = attention_mask.sum(dim=1) - 1
            logits = outputs.logits[torch.arange(batch_size, device=input_ids.device), last_token_indices]

        return logits

    def process_logits(self, logits: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Process logits through the unigram system and add padding."""
        # Apply the unigram adaptation (and update it if in training)
        if self.training:
            unigram_batch = logits.mean(dim=0, keepdim=True)
            if self.unigram.sum() == 0:
                self.unigram.data = unigram_batch
            else:
                self.unigram.data = self.unigram.data * (1 - self.unigram_beta) + unigram_batch * self.unigram_beta
        logits = logits - self.unigram[:, :logits.shape[1]]

        # Add zeros padding
        zeros = torch.zeros(
            (batch_size, self.num_zeros_to_add),
            dtype=logits.dtype,
            device=logits.device
        )
        return torch.cat((logits, zeros), dim=1)

    def map_logits_to_encoder_space(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Map the processed logits to the encoder embedding space."""
        batch_size = embeddings.shape[0]

        # Calculate how many embeddings per token (vocabulary / num_tokens)
        embs_per_token = embeddings.shape[1] // self.num_tokens

        # Reshape logits to match expected format: [batch, num_tokens, embs_per_token]
        embeddings = embeddings.reshape(batch_size, self.num_tokens, embs_per_token)

        # Process in minibatches for memory efficiency
        if self.mini_batch_size is None:
            self.mini_batch_size = min(128, batch_size)
        embeddings_list = []

        for i in range(0, batch_size, self.mini_batch_size):
            end_idx = min(i + self.mini_batch_size, batch_size)
            batch_logits = embeddings[i:end_idx]

            # For each token position, we need to transform the logits
            token_embeddings = []

            for token_idx in range(self.num_tokens):
                # Get the word embeddings for this token position
                token_word_embs = self.word_embeddings[token_idx]  # [vocab_per_token, emb_dim]

                # Get the logits for this token position
                token_logits = batch_logits[:, token_idx]  # [batch, embs_per_token]

                # Ensure dimensions match for the matrix multiplication
                if token_word_embs.shape[0] != embs_per_token:
                    if token_word_embs.shape[0] > embs_per_token:
                        token_word_embs = token_word_embs[:embs_per_token]
                    else:
                        pad_size = embs_per_token - token_word_embs.shape[0]
                        padding = torch.zeros(
                            pad_size, token_word_embs.shape[1],
                            device=token_word_embs.device
                        )
                        token_word_embs = torch.cat([token_word_embs, padding], dim=0)

                # Matrix multiply: [batch, embs_per_token] @ [embs_per_token, emb_dim] -> [batch, emb_dim]
                token_embedding = torch.matmul(token_logits, token_word_embs)
                token_embeddings.append(token_embedding)

            # Stack along sequence dimension
            batch_embeddings = torch.stack(token_embeddings, dim=1)  # [batch, num_tokens, emb_dim]
            embeddings_list.append(batch_embeddings)

        return torch.cat(embeddings_list, dim=0)

    def _shift_right(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Shift input ids one token to the right, prepend with pad token."""
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = self.tokenizer.pad_token_id
        return shifted_input_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        """
        Forward pass of the inversion model.

        Args:
            input_ids: LLM input token IDs [batch, seq_len]
            attention_mask: Attention mask for input_ids
            labels: Target token IDs for T5 decoder (training)

        Returns:
            T5 model outputs with loss (if labels provided)
        """
        batch_size = input_ids.shape[0]

        # Create attention mask if none provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Step 1: Get logits from LLM
        logits = self.get_llm_logits(input_ids, attention_mask)

        # Step 2: Process logits (unigram adaptation and padding)
        processed_logits = self.process_logits(logits, batch_size)

        # Step 3: Map to encoder space
        encoder_embeddings = self.map_logits_to_encoder_space(processed_logits)

        # Step 4: Create encoder attention mask
        encoder_attention_mask = torch.ones(
            (batch_size, self.num_tokens),
            dtype=torch.long,
            device=encoder_embeddings.device
        )

        # Step 5: Apply final embedding transformation
        encoder_embeddings = self.embedding_transform(encoder_embeddings)

        # Step 6: Pass through T5 encoder-decoder
        if labels is not None:
            decoder_input_ids = self._shift_right(labels.clone())
            outputs = self.encoder_decoder(
                inputs_embeds=encoder_embeddings,
                attention_mask=encoder_attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                return_dict=True
            )
        else:
            decoder_input_ids = torch.full(
                (batch_size, 1),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
                device=encoder_embeddings.device
            )
            outputs = self.encoder_decoder(
                inputs_embeds=encoder_embeddings,
                attention_mask=encoder_attention_mask,
                decoder_input_ids=decoder_input_ids,
                return_dict=True
            )

        return outputs

    def forward_from_logits(
        self,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ):
        """
        Forward pass from pre-computed LLM logits (for evaluation with external LLM).

        Args:
            logits: LLM output logits [batch, vocab_size]
            labels: Target token IDs for T5 decoder (training)

        Returns:
            T5 model outputs with loss (if labels provided)
        """
        batch_size = logits.shape[0]

        # Step 2: Process logits (unigram adaptation and padding)
        processed_logits = self.process_logits(logits, batch_size)

        # Step 3: Map to encoder space
        encoder_embeddings = self.map_logits_to_encoder_space(processed_logits)

        # Step 4: Create encoder attention mask
        encoder_attention_mask = torch.ones(
            (batch_size, self.num_tokens),
            dtype=torch.long,
            device=encoder_embeddings.device
        )

        # Step 5: Apply final embedding transformation
        encoder_embeddings = self.embedding_transform(encoder_embeddings)

        # Step 6: Pass through T5 encoder-decoder
        if labels is not None:
            decoder_input_ids = self._shift_right(labels.clone())
            outputs = self.encoder_decoder(
                inputs_embeds=encoder_embeddings,
                attention_mask=encoder_attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                return_dict=True
            )
        else:
            decoder_input_ids = torch.full(
                (batch_size, 1),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
                device=encoder_embeddings.device
            )
            outputs = self.encoder_decoder(
                inputs_embeds=encoder_embeddings,
                attention_mask=encoder_attention_mask,
                decoder_input_ids=decoder_input_ids,
                return_dict=True
            )

        return outputs

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        attention_mask: Optional[torch.Tensor] = None,
        num_beams: int = 4,
        early_stopping: bool = False
    ) -> torch.Tensor:
        """
        Generate text through the inversion process.

        Args:
            input_ids: LLM input token IDs [batch, seq_len]
            max_length: Maximum generation length
            attention_mask: Attention mask for input_ids
            num_beams: Number of beams for beam search
            early_stopping: Whether to stop early during beam search

        Returns:
            Generated token IDs
        """
        batch_size = input_ids.shape[0]

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        generation_kwargs = {
            "max_length": max_length,
            "num_beams": num_beams,
            "early_stopping": early_stopping,
            "decoder_start_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
            "output_scores": False,
        }

        # Process through inverse model
        logits = self.get_llm_logits(input_ids, attention_mask)
        processed_logits = self.process_logits(logits, batch_size)
        encoder_embeddings = self.map_logits_to_encoder_space(processed_logits)
        encoder_attention_mask = torch.ones(
            (batch_size, self.num_tokens),
            dtype=torch.long,
            device=encoder_embeddings.device
        )
        encoder_embeddings = self.embedding_transform(encoder_embeddings)

        # Generate text using the encoder-decoder
        generated_ids = self.encoder_decoder.generate(
            inputs_embeds=encoder_embeddings,
            attention_mask=encoder_attention_mask,
            **generation_kwargs
        )

        return generated_ids

    def generate_from_logits(
        self,
        logits: torch.Tensor,
        max_length: int,
        num_beams: int = 4,
        early_stopping: bool = False
    ) -> torch.Tensor:
        """
        Generate text from pre-computed LLM logits (for evaluation with external LLM).

        Args:
            logits: LLM output logits [batch, vocab_size]
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            early_stopping: Whether to stop early during beam search

        Returns:
            Generated token IDs
        """
        batch_size = logits.shape[0]

        generation_kwargs = {
            "max_length": max_length,
            "num_beams": num_beams,
            "early_stopping": early_stopping,
            "decoder_start_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
            "output_scores": False,
        }

        # Process through inverse model
        processed_logits = self.process_logits(logits, batch_size)
        encoder_embeddings = self.map_logits_to_encoder_space(processed_logits)
        encoder_attention_mask = torch.ones(
            (batch_size, self.num_tokens),
            dtype=torch.long,
            device=encoder_embeddings.device
        )
        encoder_embeddings = self.embedding_transform(encoder_embeddings)

        # Generate text using the encoder-decoder
        generated_ids = self.encoder_decoder.generate(
            inputs_embeds=encoder_embeddings,
            attention_mask=encoder_attention_mask,
            **generation_kwargs
        )

        return generated_ids

    def invert_text(self, input_text: str, max_length: int, device: Optional[torch.device] = None) -> str:
        """
        Convenience method to invert a text input back to its presumed source.

        Args:
            input_text: The text to invert
            max_length: Maximum length of inverted output
            device: Device to use (defaults to model's device)

        Returns:
            Inverted text string
        """
        if device is None:
            device = next(self.parameters()).device

        inputs = self.llm_tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        try:
            generated_ids = self.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length
            )
            inverted_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            return inverted_text
        except Exception as e:
            print(f"Error during inversion: {e}")
            return "Inversion failed due to error."

    def save_pretrained(self, output_dir: str):
        """
        Save the model to disk.

        Args:
            output_dir: Directory to save the model
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save the T5 model
        self.encoder_decoder.save_pretrained(os.path.join(output_dir, "t5_model"))
        self.tokenizer.save_pretrained(os.path.join(output_dir, "t5_tokenizer"))

        # Save the transformation layers
        torch.save(
            self.embedding_transform.state_dict(),
            os.path.join(output_dir, "embedding_transform.pt")
        )
        torch.save(
            self.unigram,
            os.path.join(output_dir, "unigram.pt")
        )

        # Save the configuration
        config = {
            "llm_model_name": self._llm_model_name,
            "bottleneck_dim": self.bottleneck_dim,
            "num_tokens": self.num_tokens,
            "unigram_beta": self.unigram_beta,
        }

        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        external_llm: Optional[nn.Module] = None,
        external_llm_tokenizer: Optional[AutoTokenizer] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "LLMInversionModel":
        """
        Load a pretrained model from disk.

        Args:
            model_path: Path to the saved model directory
            external_llm: Pre-loaded external LLM (for model caching)
            external_llm_tokenizer: Pre-loaded external LLM tokenizer
            device: Device to load the model to

        Returns:
            Loaded LLMInversionModel instance
        """
        with open(os.path.join(model_path, "config.json"), "r") as f:
            config = json.load(f)

        t5_path = os.path.join(model_path, "t5_model")
        t5_tokenizer_path = os.path.join(model_path, "t5_tokenizer")

        model = cls(
            t5_model_name=t5_path,
            t5_tokenizer_name=t5_tokenizer_path,
            llm_model_name=config["llm_model_name"] if external_llm is None else None,
            external_llm=external_llm,
            external_llm_tokenizer=external_llm_tokenizer,
            bottleneck_dim=config["bottleneck_dim"],
            num_tokens=config["num_tokens"],
            unigram_beta=config["unigram_beta"]
        )

        # Load the transformation layers
        embedding_transform_path = os.path.join(model_path, "embedding_transform.pt")
        unigram_path = os.path.join(model_path, "unigram.pt")

        if device is not None:
            model.embedding_transform.load_state_dict(
                torch.load(embedding_transform_path, map_location=device)
            )
            model.unigram = torch.load(unigram_path, map_location=device)
        else:
            model.embedding_transform.load_state_dict(
                torch.load(embedding_transform_path)
            )
            model.unigram = torch.load(unigram_path)

        return model


__all__ = ["LLMInversionModel"]
