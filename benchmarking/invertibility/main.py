import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import logging
import argparse
import copy
import wandb
import matplotlib.pyplot as plt
from functools import partial
from typing import Dict

from gradient_estimators import (
    STGS,
    ReinforceEstimator,
    stgs_forward_pass,
    reinforce_forward_pass,
    estimate_stgs_gradient_variance,
    estimate_stgs_gradient_bias,
    estimate_reinforce_gradient_variance,
)
from metrics_registry import lcs_length


def setup_model_and_tokenizer(model_name="HuggingFaceM4/tiny-random-LlamaForCausalLM",device='cpu',model_precision='full'):
    """
    Load model and tokenizer
    """
    # Load model and move to device
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()  # Set to evaluation mode (frozen)

    # Freeze model weights
    for param in model.parameters():
        param.requires_grad = False

    model = model.to(device)
    if model_precision == "half":
        model.half()
    
    # Load tokenizer (only needed for the target sequence)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

def _init_mlm_mask(vocab_size, seq_len, device, batch_size,
                   mlm_model_name, target_text, causal_tokenizer, init_std, top_k=50):
    """Build init logits using a masked LM to predict a masked prefix."""
    from transformers import AutoTokenizer as AT, AutoModelForMaskedLM as AMLM

    mlm_tokenizer = AT.from_pretrained(mlm_model_name)
    mlm_model = AMLM.from_pretrained(mlm_model_name).eval()

    mask_id = mlm_tokenizer.mask_token_id
    masks = [mask_id] * seq_len

    # Build input: [CLS] MASK*seq_len [SEP] target [SEP]
    target_enc = mlm_tokenizer.encode(target_text, add_special_tokens=False)
    max_target = 512 - seq_len - 3
    target_enc = target_enc[:max_target]

    input_ids = torch.tensor(
        [mlm_tokenizer.cls_token_id] + masks +
        [mlm_tokenizer.sep_token_id] + target_enc +
        [mlm_tokenizer.sep_token_id]
    ).unsqueeze(0)  # (1, L)

    with torch.no_grad():
        mlm_out = mlm_model(input_ids)
        # MASK positions are indices 1 .. seq_len
        mlm_logits = mlm_out.logits[0, 1:seq_len + 1, :]  # (seq_len, mlm_vocab_size)
        #mlm_probs = mlm_logits.softmax(dim=-1)

    # Build causal-LM logit tensor via vocab mapping
    min_logit = mlm_logits.min()
    causal_logits = min_logit*torch.ones((seq_len, vocab_size), device=device)
    if seq_len > 0:
        #top_ids = mlm_probs.topk(min(top_k, mlm_probs.shape[-1]), dim=-1).indices  # (seq_len, top_k)
        top_ids = mlm_logits.topk(min(top_k, mlm_logits.shape[-1]), dim=-1).indices  # (seq_len, top_k)
        mismatch_tokens = []
        for pos in range(seq_len):
            for mlm_id in top_ids[pos].tolist():
                token_str = mlm_tokenizer.decode([mlm_id])
                causal_ids = causal_tokenizer.encode(token_str, add_special_tokens=False)
                # If an mlm token result into many causal tokens, then we use the offset on position:
                if len(causal_ids) > 1 and token_str not in mismatch_tokens:
                    mismatch_tokens.append(token_str)
                    print(f"WARNING: using mlm_mask initialization: '{token_str}' -> {len(causal_ids)} causal tokens.")
                for offset, causal_id in enumerate(causal_ids):
                    eff_pos = min(pos+offset, causal_logits.shape[0]-1)
                    causal_logits[eff_pos, causal_id] = mlm_logits[pos, mlm_id].item()
                '''
                if len(causal_ids) == 1 and causal_ids[0] < vocab_size:
                    #causal_logits[pos, causal_ids[0]] = mlm_probs[pos, mlm_id].item()
                    causal_logits[pos, causal_ids[0]] = mlm_logits[pos, mlm_id].item()
                '''

    logits = causal_logits.unsqueeze(0).expand(batch_size, -1, -1).clone()
    if init_std > 0:
        logits = logits + torch.randn_like(logits) * init_std
    return logits


def initialize_learnable_inputs(
    vocab_size, seq_len, device, batch_size=1,
    init_strategy="randn", init_std=0.0,
    # for "embedding_similarity"
    embedding_weights=None, target_token_ids=None,
    # for "lm_target_prior"
    model=None, target_input_ids=None, allowed_tokens=None,
    # for "mlm_mask"
    mlm_model_name="distilbert-base-uncased", 
    mlm_top_k=50,
    target_text=None, tokenizer=None,
):
    """
    Initialize learnable logits using the specified strategy.

    Strategies:
        "randn"               : Standard normal (current default, backward-compat).
        "zeros"               : All-zero logits → exactly uniform softmax.
        "normal"              : Normal with configurable init_std.
        "one_hot_random"      : Each position gets a +10 spike on a random token.
        "embedding_similarity": Logits ∝ cosine similarity of each token's embedding
                                with the mean target embedding.
        "lm_target_prior"     : Base logits = next-token logits of the LM conditioned
                                on the target text; per-position noise of init_std added.
        "mlm_mask"            : Use a masked LM to predict a masked prefix given the
                                target text; vocab-mapped to causal-LM token indices.
    """
    if init_strategy == "zeros":
        logits = torch.zeros((batch_size, seq_len, vocab_size), device=device)

    elif init_strategy == "normal":
        logits = torch.empty((batch_size, seq_len, vocab_size), device=device)
        torch.nn.init.normal_(logits, std=init_std)

    elif init_strategy == "one_hot_random":
        logits = torch.zeros((batch_size, seq_len, vocab_size), device=device)
        if seq_len > 0:
            idx = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            logits.scatter_(-1, idx.unsqueeze(-1), 10.0)

    elif init_strategy == "embedding_similarity":
        # embedding_weights: (vocab_size, embed_dim) — allowed-vocab subset
        # target_token_ids:  (target_length,)        — mapped to allowed-vocab indices
        target_embeds = embedding_weights[target_token_ids]  # (target_length, embed_dim)
        mean_embed = target_embeds.mean(dim=0)               # (embed_dim,)
        norm_e = embedding_weights / (embedding_weights.norm(dim=1, keepdim=True) + 1e-9)
        norm_m = mean_embed / (mean_embed.norm() + 1e-9)
        sims = (norm_e @ norm_m).clamp(min=0)                # (vocab_size,)
        logits = sims.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1).clone()

    elif init_strategy == "lm_target_prior":
        # Run the causal LM on the target text and use the next-token logits as a prior.
        with torch.no_grad():
            out = model(target_input_ids.to(device))
            base_logits = out.logits[0, -1, :]          # (full_vocab_size,)
            if allowed_tokens is not None:
                base_logits = base_logits[allowed_tokens]  # (vocab_size,)
        base = base_logits.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1).clone()
        if init_std > 0:
            base = base + torch.randn_like(base) * init_std
        logits = base

    elif init_strategy == "mlm_mask":
        logits = _init_mlm_mask(
            vocab_size, seq_len, device, batch_size,
            mlm_model_name, target_text, tokenizer, init_std,
            top_k=mlm_top_k,
        )

    else:  # "randn" — current default, kept for backward compat
        logits = torch.randn((batch_size, seq_len, vocab_size), device=device)

    return logits.detach().requires_grad_(True)

def filter_vocabulary(embedding_weights, seed_vector, target_token_ids, threshold=0.5):
    """
    Compute cosine similarities between each vocabulary token embedding and seed_vector.
    Return the indices of tokens with similarity above the threshold,
    while always including the target_token_ids.
    """
    norm_weights = embedding_weights / (embedding_weights.norm(dim=1, keepdim=True) + 1e-9)
    norm_seed = seed_vector / (seed_vector.norm() + 1e-9)
    similarities = torch.matmul(norm_weights, norm_seed.unsqueeze(1)).squeeze(1)
    allowed = (similarities >= threshold).nonzero(as_tuple=True)[0]

    # Ensure target tokens are included
    target_set = set(target_token_ids.tolist())
    allowed_set = set(allowed.tolist())
    union_allowed = allowed_set.union(target_set)
    union_allowed = torch.tensor(sorted(list(union_allowed)), device=embedding_weights.device)
    return union_allowed

def generate_completions(model, input_embeddings, target_length, temperature=1.0, pre_prompt=None):
    """
    Generate completions from the model using the input embeddings
    """
    # Store the original inputs for later use
    batch_size = input_embeddings.shape[0]
    embedding_dim = input_embeddings.shape[2]

    # Initialize past_key_values as None
    past_key_values = None

    # Get the embedding layer for shape reference and token-to-embedding conversion
    embedding_layer = model.get_input_embeddings()

    # Process pre-prompt if provided
    if pre_prompt is not None:
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided when using pre_prompt")

        # Tokenize the pre-prompt
        pre_prompt_tokens = tokenizer(pre_prompt, return_tensors="pt").input_ids.to(input_embeddings.device)

        # Convert tokens to embeddings
        pre_prompt_embeds = embedding_layer(pre_prompt_tokens)

        # Combine pre-prompt embeddings with learnable embeddings
        # [batch_size, pre_prompt_length + learnable_length, hidden_size]
        combined_embeds = torch.cat([pre_prompt_embeds, input_embeddings], dim=1)
        current_embeds = combined_embeds
    else:
        # Use just the learnable embeddings
        current_embeds = input_embeddings

    # Store all generated token ids
    all_token_ids = []

    # Autoregressive generation loop
    for _ in range(target_length):
        # Forward pass
        outputs = model(
            inputs_embeds=current_embeds,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

        # Get the logits for the next token
        next_token_logits = outputs.logits[:, -1, :]

        # Apply temperature
        next_token_logits = next_token_logits / temperature

        # Get the most likely next token
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        all_token_ids.append(next_token_id)

        # Convert token to embedding for next iteration
        next_token_embed = embedding_layer(next_token_id)

        # Set up for next iteration
        current_embeds = next_token_embed
        past_key_values = outputs.past_key_values

    # Concatenate all token ids
    generated_token_ids = torch.cat(all_token_ids, dim=1)

    return generated_token_ids, outputs.hidden_states

class LossClass(object):
    def __init__(
        self,
        losses,
        embedding_weights_subset,
        target_text,
        target_tokens_mapped,
        model,
        tokenizer,
        kwargs,
    ):
        self.eps = 1e-8
        self.losses = losses
        self.target_text = target_text
        self.target_tokens_mapped = target_tokens_mapped
        self.embedding_weights_subset = embedding_weights_subset
        self.model = model
        self.tokenizer = tokenizer

        self.vocab_size = self.embedding_weights_subset.shape[0]
        self.embedding_dim = self.embedding_weights_subset.shape[1]
        self.batch_size = self.target_tokens_mapped.shape[0]
        self.target_seq_len = self.target_tokens_mapped.shape[1]
        self.hidden_state_dim = self.model.config.hidden_size

        self.kwargs = kwargs

        # EmbXEntropy:
        if 'embxentropy' in self.losses.lower():
            target_tokens_one_hot = F.one_hot(
                self.target_tokens_mapped,
                num_classes=self.vocab_size,
            ).float()
            # batch_size x target_seq_len x vocab_size
            target_embeddings = torch.matmul(target_tokens_one_hot, self.embedding_weights_subset)
            # batch_size x target_seq_len x embedding_dim
            '''
            diff_target_minus_other = target_embeddings.unsqueeze(2).expand(-1,-1,self.vocab_size, -1) - self.embedding_weights_subset.reshape(
                1,1,self.vocab_size, self.embedding_dim,
            ).expand(
                self.batch_size, self.target_seq_len, -1,-1,
            )
            # batch_size x target_seq_len x vocab_size x embedding_dim
            l2_norm_diff_target_other = torch.linalg.norm(
                diff_target_minus_other,
                dim=-1,
                ord=2,
            )
            # batch_size x target_seq_len x vocab_size
            '''
            l2_norm_diff_target_other = torch.zeros((self.batch_size, self.target_seq_len, self.vocab_size))
            for bidx in range(self.batch_size):
                for tidx in range(self.target_seq_len):
                    diff_target_minus_other = target_embeddings[bidx,tidx].unsqueeze(0).expand(self.vocab_size, -1) - self.embedding_weights_subset
                    # (vocab_size x emd_dim)
                    l2_norm_diff_target_other[bidx,tidx] = torch.linalg.norm(
                        diff_target_minus_other,
                        dim=-1,
                        ord=2,
                    )
                    # vocab_size

            self.target_distr = torch.softmax(
                1.0 / (self.eps + l2_norm_diff_target_other),
                dim=-1,
            ).to(device=self.embedding_weights_subset.device)

        elif 'embLayer' in self.losses:
            target_tokens_one_hot = F.one_hot(
                self.target_tokens_mapped,
                num_classes=self.vocab_size,
            ).float()
            # batch_size x target_seq_len x vocab_size
            target_embeddings = torch.matmul(target_tokens_one_hot, self.embedding_weights_subset)
            # batch_size x target_seq_len x embedding_dim
            l2_norm_diff_target_other = torch.zeros((self.batch_size, self.target_seq_len, self.vocab_size))
            
            target_outputs = self.model(
                inputs_embeds=target_embeddings,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            
            if 'L2' in self.losses:
                loss_type = 'L2'
            elif 'cos' in self.losses.lower():
                loss_type = 'Cos'

            self.embedding_layers = [int(number) for number in self.losses.split('embLayer')[1].split(loss_type)[0].split('+')]
            
            self.target_embeddings = { emblayer:target_outputs.hidden_states[emblayer].to(device=self.embedding_weights_subset.device) 
                for emblayer in self.embedding_layers
            }
            # (batch_size x target_seq_len x hidden_state)

    def compute_perplexity(
        self,
        all_logits,
        predictions,
    ):
        '''
        Compute perplexity with log:
        :param all_logits: batch_size x seq_len x vocab_size
        :param predictions: batch_size x seq_len 
        '''
        lslhd = all_logits 
        #(batch_size x seq_len x vocab_size)
        batch_size = all_logits.shape[0]
        seq_len = all_logits.shape[1]
        vocab_size = all_logits.shape[2]

        lslhd = lslhd.log_softmax(dim=-1)
        lslhd = lslhd.gather(
            dim=-1, 
            index=predictions.unsqueeze(-1).int(),
        ).squeeze(-1)
        # (batch_size x seq_len)
        #lslhd = lsoftmaxed_pl.gather(dim=-1, index=tokenized_prediction.unsqueeze(-1)).squeeze(-1)
        if self.tokenizer.pad_token_id is not None:
            lnotpadding_mask = (predictions != self.tokenizer.pad_token_id).float()
            #lnotpadding_mask = (tokenized_prediction != self.tokenizer.pad_token_id).float()
            #options_true_length = (batched_options_inputs.input_ids != self.tokenizer.pad_token_id).long().sum(dim=-1).unsqueeze(-1)
            #(batch_size x 1)
            lslhd = lnotpadding_mask * lslhd
        else:
            lnotpadding_mask = torch.ones_like(predictions)
        #torch.pow(slhd, 1.0/options_true_length)
        #(option_batch_size x option_len)
        #print('cache option: ', lslhd.shape)
        #print(lslhd)
        lsentences_likelihoods = lslhd.sum(dim=-1) #= slhd.cpu().prod(dim=-1).to(slhd.device)
        #(batch_size )
        lsentences_perplexities = torch.exp(-lsentences_likelihoods / (lnotpadding_mask.sum(dim=-1)+1e-8)) #1.0/(slhd+1e-8)
        # (batch_size )
        
        return lsentences_perplexities

    def compute_loss(
        self,
        input_dict,
    ):
        sumloss = 0
        losses_dict = {}

        if "crossentropy" in self.losses.lower():
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(
                #generated_logits.reshape(-1, vocab_size),
                input_dict['generated_logits'].reshape(-1, self.embedding_weights_subset.shape[0]),
                self.target_tokens_mapped.reshape(-1), #.reshape(1, -1).repeat(batch_size, 1),
                #target_tokens.reshape(-1)
            )
            losses_dict['crossentropy'] = loss
            sumloss += loss

        if "embedded" in self.losses.lower():
            #print(self.embedding_weights_subset.shape)
            #print(input_dict['generated_logits'].shape)
            generated_distr = input_dict['generated_logits'].softmax(dim=-1)
            generated_embeddings = torch.matmul( 
                generated_distr,
                self.embedding_weights_subset,
            )
            #print(self.target_tokens_mapped.shape)
            target_tokens_one_hot = F.one_hot(self.target_tokens_mapped, num_classes=self.embedding_weights_subset.shape[0]).float()
            #print(target_tokens_one_hot.shape)
            target_embeddings = torch.matmul(target_tokens_one_hot, self.embedding_weights_subset)
            loss_fn = torch.nn.MSELoss(size_average=None, reduce=None, reduction='none')#'mean')
            loss = loss_fn(
                input=generated_embeddings,
                target=target_embeddings.detach(),
            )
            # (batch_size x target_seq_len x embedding_size)
            #loss = loss.mean() #dim=-1).mean(dim=-1)
            #loss = loss.sum(dim=-1).sqrt().mean()
            loss = loss.sum(dim=-1).mean()
            # (batch_size
            losses_dict['embedded'] = loss
            sumloss += loss

        if "embxentropy" in self.losses.lower():
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(
                input_dict['generated_logits'].reshape(-1, self.vocab_size),
                target=self.target_distr.reshape(-1, self.vocab_size).detach(),
            )
            losses_dict['embxentropy'] = loss
            sumloss += loss

        if 'embLayer' in self.losses :
            generated_embeddings = {}
            for emblayer in self.embedding_layers:
                generated_embeddings[emblayer] = [hs[emblayer][:,-1:] for hs in input_dict['generated_hidden_states']]
                generated_embeddings[emblayer] = torch.cat(generated_embeddings[emblayer], dim=1)
                # (batch_size x target_seq_len x hidden-dim)

            losses = {}
            if 'L2' in self.losses:
                loss_type = 'L2'
                loss_fn = torch.nn.MSELoss(size_average=None, reduce=None, reduction='none')#'mean')
                for emblayer in self.embedding_layers:
                    losses[emblayer] = loss_fn(
                        input=generated_embeddings[emblayer],
                        target=self.target_embeddings[emblayer].detach(),
                    )
                    # (batch_size x target_seq_len x embedding_size)
                    #loss = loss.mean() #dim=-1).mean(dim=-1)
                    #loss = loss.sum(dim=-1).sqrt().mean()
                    losses[emblayer] = losses[emblayer].sum(dim=-1).mean()
                    # (batch_size
                    losses_dict[f"embLayer{emblayer}L2"] = losses[emblayer]
            elif 'cos' in self.losses.lower():
                loss_type = 'Cos'
                loss_fn = torch.nn.CosineEmbeddingLoss(reduction='mean')
                target = torch.ones((self.batch_size*self.target_seq_len,), device=self.embedding_weights_subset.device) 
                for emblayer in self.embedding_layers:
                    losses[emblayer] = loss_fn(
                        input1=generated_embeddings[emblayer].reshape(-1,self.hidden_state_dim),
                        input2=self.target_embeddings[emblayer].detach().reshape(-1, self.hidden_state_dim),
                        target=target,
                    )
                    # scalere because mean over (batch_size * target_seq_len) ?
                    losses_dict[f"embLayer{emblayer}{loss_type}"] = losses[emblayer]
            loss = sum(losses.values())
            losses_dict[f"embLayer{loss_type}"] = loss
            sumloss += loss

        promptPLX = self.compute_perplexity(
            all_logits=input_dict['prompt_logits'],
            # PREVIOUSLY:
            #predictions=input_dict['prompt_ids'],
            # NOW: with argmax ids:
            predictions=input_dict['prompt_argmax_ids'],
        )
        losses_dict[f"PLX-prompt"] = promptPLX.mean()
        complPLX = self.compute_perplexity(
            all_logits=input_dict['generated_logits'],
            predictions=input_dict['completion_ids'],
        )
        losses_dict[f"PLX-completion"] = complPLX.mean()

        # Teacher-forced completion perplexity: PPL over prompt + actual target tokens
        if 'tf_generated_logits' in input_dict and input_dict['tf_generated_logits'] is not None:
            tf_compl_logits = input_dict['tf_generated_logits']
            # tf_compl_logits: (batch_size, seq_len+target_length, vocab_size)
            # prompt portion (positions 0..seq_len-1) matches prompt_argmax_ids
            # completion portion (positions seq_len..seq_len+target_length-1) matches target_tokens_mapped
            # Build combined predictions: prompt argmax ids + target tokens
            combined_predictions = torch.cat([
                input_dict['prompt_argmax_ids'],
                self.target_tokens_mapped,
            ], dim=1)
            promptTfComplPLX = self.compute_perplexity(
                all_logits=tf_compl_logits,
                predictions=combined_predictions,
            )
            losses_dict['PLX-prompt-tf-completion'] = promptTfComplPLX.mean()
        else:
            promptTfComplPLX = None

        if 'perplexity' in self.losses.lower():
            if 'promptPerplexity' not in self.losses:
                promptPLX = torch.zeros(self.batch_size).to(loss.device)

            if 'completionPerplexity' not in self.losses:
                complPLX = torch.zeros(self.batch_size).to(loss.device)

            promptLambda = self.kwargs['promptLambda']
            complLambda = self.kwargs['complLambda']
            loss = (complLambda*complPLX + promptLambda*promptPLX).mean()
            sumloss += loss

        if 'promptTfComplPerplexity' in self.losses:
            if promptTfComplPLX is not None:
                promptTfComplLambda = self.kwargs['promptTfComplLambda']
                loss = (promptTfComplLambda * promptTfComplPLX).mean()
                sumloss += loss

        losses_dict['sumloss'] = sumloss

        return losses_dict

class TokenOverlapMetric(object):
    def __init__(
        self,
        target_text,
        tokenizer,
    ):
        self.target_text = target_text
        self.tokenizer = tokenizer

    def measure(
        self,
        prompt_text=None,
        prompt_tokens=None,
    ):
        output_dict = {}

        target_tokens = self.tokenizer(
            self.target_text, 
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0]
        if prompt_tokens is None:
            assert prompt_text is not None
            prompt_tokens = self.tokenizer(
                prompt_text, 
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids[0]
        elif isinstance(prompt_tokens, list):
            prompt_tokens = torch.Tensor(prompt_tokens) 
        
        tt_occ = {}
        for ttoken in target_tokens:
            tt_occ[ttoken.item()] = (prompt_tokens == ttoken).int().sum().item()

        nbr_occ = sum(tt_occ.values())
        max_occ = prompt_tokens.shape[-1]

        overlap_ratio = nbr_occ / max_occ
        output_dict['token_overlap_ratio'] = overlap_ratio

        target_set = set(target_tokens.tolist())
        target_set_size = len(target_set)
        target_hits = {
            ttoken: int(t_occ>0)
            for ttoken, t_occ in tt_occ.items()
        }
        target_hits_size = sum(target_hits.values())
        output_dict['target_hit_ratio'] = float(target_hits_size) / target_set_size
        
        return output_dict


def build_fixed_logits_spec(
    ground_truth_prompt_tokens,
    allowed_token_to_idx,
    allowed_vocab_size,
    tokenizer,
    device,
    seq_len,
    fixed_gt_prefix_n=0,
    fixed_gt_suffix_n=0,
    fixed_gt_prefix_rank2_n=0,
    fixed_gt_suffix_rank2_n=0,
    fixed_prefix_text=None,
    fixed_suffix_text=None,
):
    """
    Build fixed logit tensors for partially fixed prompt optimization (Approach B: Split Tensor).

    Assembly order: [fixed_prefix | free_middle | fixed_suffix]

    Returns:
        (fixed_prefix_logits, fixed_suffix_logits, n_free)
        fixed_prefix_logits: (1, n_prefix, allowed_vocab_size) or None, requires_grad=False
        fixed_suffix_logits: (1, n_suffix, allowed_vocab_size) or None, requires_grad=False
        n_free: seq_len - n_prefix - n_suffix  (must be >= 0)
    """
    BIG = 100.0
    prefix_parts = []
    suffix_parts = []

    has_gt_prefix = fixed_gt_prefix_n > 0 or fixed_gt_prefix_rank2_n > 0
    has_gt_suffix = fixed_gt_suffix_n > 0 or fixed_gt_suffix_rank2_n > 0
    if fixed_prefix_text and has_gt_prefix:
        logging.warning("build_fixed_logits_spec: combining fixed_prefix_text with GT-based prefix; rank alignment may be inconsistent.")
    if fixed_suffix_text and has_gt_suffix:
        logging.warning("build_fixed_logits_spec: combining fixed_suffix_text with GT-based suffix; rank alignment may be inconsistent.")

    # ---- PREFIX ----

    # Text prefix (leftmost part of prefix)
    if fixed_prefix_text:
        tok_ids = tokenizer(fixed_prefix_text, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
        n = len(tok_ids)
        if n > 0:
            logits = torch.full((1, n, allowed_vocab_size), -BIG, device=device)
            for pos, tok_id in enumerate(tok_ids):
                if tok_id in allowed_token_to_idx:
                    logits[0, pos, allowed_token_to_idx[tok_id]] = BIG
                else:
                    logging.warning(
                        f"build_fixed_logits_spec: fixed_prefix_text token {tok_id} "
                        f"('{tokenizer.decode([tok_id])}') not in allowed vocab; using index 0."
                    )
                    logits[0, pos, 0] = BIG
            prefix_parts.append(logits.detach())

    # GT rank-1 prefix: first fixed_gt_prefix_n GT positions fixed at rank 1
    if fixed_gt_prefix_n > 0:
        if ground_truth_prompt_tokens is None:
            logging.warning("build_fixed_logits_spec: fixed_gt_prefix_n > 0 but ground_truth_prompt_tokens is None; skipping.")
        else:
            n = min(fixed_gt_prefix_n, len(ground_truth_prompt_tokens))
            logits = torch.full((1, n, allowed_vocab_size), -BIG, device=device)
            for pos in range(n):
                gt_tok = ground_truth_prompt_tokens[pos]
                if gt_tok in allowed_token_to_idx:
                    logits[0, pos, allowed_token_to_idx[gt_tok]] = BIG
            prefix_parts.append(logits.detach())

    # GT rank-2 prefix: next fixed_gt_prefix_rank2_n GT positions fixed at rank 2
    if fixed_gt_prefix_rank2_n > 0:
        if ground_truth_prompt_tokens is None:
            logging.warning("build_fixed_logits_spec: fixed_gt_prefix_rank2_n > 0 but ground_truth_prompt_tokens is None; skipping.")
        else:
            start_pos = fixed_gt_prefix_n
            n = min(fixed_gt_prefix_rank2_n, max(0, len(ground_truth_prompt_tokens) - start_pos))
            if n > 0:
                logits = torch.randn((1, n, allowed_vocab_size), device=device)
                eps_gap = 1.0
                for pos in range(n):
                    gt_tok = ground_truth_prompt_tokens[start_pos + pos]
                    if gt_tok in allowed_token_to_idx:
                        allowed_idx = allowed_token_to_idx[gt_tok]
                        current_max = logits[0, pos].max().item()
                        gt_val = logits[0, pos, allowed_idx].item()
                        if gt_val >= current_max:
                            # GT is rank 1 → make a different token rank 1
                            rank1_idx = (allowed_idx + 1) % allowed_vocab_size
                            logits[0, pos, rank1_idx] = gt_val + eps_gap
                        elif gt_val < current_max - eps_gap:
                            # GT is too far below rank 1 → clamp to rank 2
                            logits[0, pos, allowed_idx] = current_max - eps_gap
                prefix_parts.append(logits.detach())

    # ---- SUFFIX ----

    # GT rank-2 suffix: positions just before the rank-1 suffix
    if fixed_gt_suffix_rank2_n > 0:
        if ground_truth_prompt_tokens is None:
            logging.warning("build_fixed_logits_spec: fixed_gt_suffix_rank2_n > 0 but ground_truth_prompt_tokens is None; skipping.")
        else:
            gt_len = len(ground_truth_prompt_tokens)
            end_pos = gt_len - fixed_gt_suffix_n
            start_pos = max(0, end_pos - fixed_gt_suffix_rank2_n)
            n = end_pos - start_pos
            if n > 0:
                logits = torch.randn((1, n, allowed_vocab_size), device=device)
                eps_gap = 1.0
                for pos in range(n):
                    gt_tok = ground_truth_prompt_tokens[start_pos + pos]
                    if gt_tok in allowed_token_to_idx:
                        allowed_idx = allowed_token_to_idx[gt_tok]
                        current_max = logits[0, pos].max().item()
                        gt_val = logits[0, pos, allowed_idx].item()
                        if gt_val >= current_max:
                            rank1_idx = (allowed_idx + 1) % allowed_vocab_size
                            logits[0, pos, rank1_idx] = gt_val + eps_gap
                        elif gt_val < current_max - eps_gap:
                            logits[0, pos, allowed_idx] = current_max - eps_gap
                suffix_parts.append(logits.detach())

    # GT rank-1 suffix: last fixed_gt_suffix_n GT positions fixed at rank 1
    if fixed_gt_suffix_n > 0:
        if ground_truth_prompt_tokens is None:
            logging.warning("build_fixed_logits_spec: fixed_gt_suffix_n > 0 but ground_truth_prompt_tokens is None; skipping.")
        else:
            gt_len = len(ground_truth_prompt_tokens)
            n = min(fixed_gt_suffix_n, gt_len)
            logits = torch.full((1, n, allowed_vocab_size), -BIG, device=device)
            for pos in range(n):
                gt_tok = ground_truth_prompt_tokens[gt_len - n + pos]
                if gt_tok in allowed_token_to_idx:
                    logits[0, pos, allowed_token_to_idx[gt_tok]] = BIG
            suffix_parts.append(logits.detach())

    # Text suffix (rightmost part of suffix)
    if fixed_suffix_text:
        tok_ids = tokenizer(fixed_suffix_text, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
        n = len(tok_ids)
        if n > 0:
            logits = torch.full((1, n, allowed_vocab_size), -BIG, device=device)
            for pos, tok_id in enumerate(tok_ids):
                if tok_id in allowed_token_to_idx:
                    logits[0, pos, allowed_token_to_idx[tok_id]] = BIG
                else:
                    logging.warning(
                        f"build_fixed_logits_spec: fixed_suffix_text token {tok_id} "
                        f"('{tokenizer.decode([tok_id])}') not in allowed vocab; using index 0."
                    )
                    logits[0, pos, 0] = BIG
            suffix_parts.append(logits.detach())

    fixed_prefix = torch.cat(prefix_parts, dim=1) if prefix_parts else None
    fixed_suffix = torch.cat(suffix_parts, dim=1) if suffix_parts else None

    n_prefix = fixed_prefix.shape[1] if fixed_prefix is not None else 0
    n_suffix = fixed_suffix.shape[1] if fixed_suffix is not None else 0
    n_free = seq_len - n_prefix - n_suffix

    if n_free < 0:
        raise ValueError(
            f"build_fixed_logits_spec: total fixed positions ({n_prefix} prefix + {n_suffix} suffix) "
            f"exceed seq_len ({seq_len}). Reduce fixed position counts."
        )

    return fixed_prefix, fixed_suffix, n_free


def optimize_inputs(
    model,
    tokenizer,
    device,
    losses="crossentropy",
    bptt=False,
    target_text="The quick brown fox jumps over the lazy dog",
    pre_prompt=None,
    seq_len=50,
    epochs=1000,
    learning_rate=0.01,
    temperature = 0.5,
    bptt_temperature = 0.5,
    learnable_temperature=False,
    decouple_learnable_temperature=False,
    bptt_learnable_temperature=False,
    bptt_decouple_learnable_temperature=False,
    stgs_hard=True,
    bptt_stgs_hard=True,
    bptt_hidden_state_conditioning=False,
    plot_every=10,
    log_table_every=100,
    stgs_grad_variance_samples=0,
    stgs_grad_variance_period=1,
    stgs_grad_bias_samples=0,
    stgs_grad_bias_period=1,
    stgs_grad_bias_reference_samples=0,
    stgs_grad_bias_reference_batch_size=None,
    stgs_grad_bias_reference_use_baseline=True,
    stgs_grad_bias_reference_reward_scale=1.0,
    stgs_grad_bias_reference_baseline_beta=0.9,
    gradient_estimator="stgs",
    reinforce_grad_variance_samples=0,
    reinforce_grad_variance_period=1,
    reinforce_reward_scale=1.0,
    reinforce_use_baseline=True,
    reinforce_baseline_beta=0.9,
    eps=1e-10,
    bptt_eps=1e-10,
    vocab_threshold=0.5,  # Hyperparameter for filtering
    filter_vocab=False,
    max_gradient_norm=0.0,
    batch_size=1,
    # STGS initialization strategy
    init_strategy="randn",
    init_std=0.0,
    init_mlm_model="distilbert-base-uncased",
    init_mlm_top_k=50,
    # Method selection (NEW)
    method="stgs",  # "stgs", "reinforce", "soda", "gcg"
    # Backend selection for SODA/GCG (NEW)
    baseline_backend="hf",  # "hf" or "tl"
    baseline_model_name=None,  # Model name for transformer_lens backend
    # SODA-specific parameters (NEW)
    soda_decay_rate=0.9,
    soda_betas=(0.9, 0.995),
    soda_reset_epoch=50,
    soda_reinit_epoch=1500,
    soda_reg_weight=None,
    soda_bias_correction=False,
    soda_init_strategy="zeros",
    soda_init_std=0.05,
    # GCG-specific parameters (NEW)
    gcg_num_candidates=704,
    gcg_top_k=128,
    gcg_num_mutations=1,
    gcg_pos_choice="uniform",
    gcg_token_choice="uniform",
    gcg_init_strategy="zeros",
    # Teacher forcing (for faster training)
    teacher_forcing=False,
    # Prompt reconstruction evaluation (SODA-style)
    ground_truth_prompt_tokens=None,  # List[int] - ground truth prompt for prompt reconstruction metrics
    ground_truth_prompt=None,  # str - ground truth prompt string for logging
    # Semantic metrics (BERT/SentenceBERT) per-epoch tracking
    semantic_metrics_every_n_epochs: int = 0,  # 0=disabled, N=compute every N epochs
    bertscore_model_type: str = "distilbert-base-uncased",
    sentencebert_model_name: str = "all-MiniLM-L6-v2",
    # Fixed logit distribution parameters (Approach B: Split Tensor)
    fixed_gt_prefix_n: int = 0,       # fix first N positions, GT at rank 1
    fixed_gt_suffix_n: int = 0,       # fix last N positions, GT at rank 1
    fixed_gt_prefix_rank2_n: int = 0, # fix first N positions (after rank-1 prefix), GT at rank 2
    fixed_gt_suffix_rank2_n: int = 0, # fix last N positions (before rank-1 suffix), GT at rank 2
    fixed_prefix_text: str = None,    # text whose tokens become a fixed prefix
    fixed_suffix_text: str = None,    # text whose tokens become a fixed suffix
    early_stop_on_exact_match: bool = True,
    early_stop_loss_threshold: float = 0.01,
    kwargs={},
):
    """
    Optimize input embeddings to make the frozen model produce the target output as a completion.

    Args:
        method: Optimization method - "stgs", "reinforce", "soda", or "gcg"
        baseline_backend: Backend for SODA/GCG - "hf" (HuggingFace) or "tl" (transformer_lens)
        baseline_model_name: Model name for transformer_lens backend (defaults to model's name)
        soda_*: SODA-specific parameters
        gcg_*: GCG-specific parameters
        ... (other parameters documented below)
    """
    # Dispatch to SODA/GCG if requested
    method_lower = method.lower() if method else "stgs"

    # Helper to wrap baseline returns with empty prompt_metrics_history and semantic_metrics_history
    def _wrap_baseline_result(result):
        """Wrap baseline result to include empty prompt_metrics_history and semantic_metrics_history for compatibility.

        If the result is a 4-tuple (old-style baseline), add empty prompt_metrics_history and semantic_metrics_history.
        If the result is a 5-tuple (old-style with prompt_metrics_history), add empty semantic_metrics_history.
        If the result is a 6-tuple (new-style with both), return as-is.
        """
        empty_prompt_metrics = {"prompt_token_accuracy": [], "prompt_exact_match": [], "prompt_lcs_ratio": [], "prompt_cosine_similarity": []}
        empty_semantic_metrics = {
            "output_bertscore_f1": [], "output_bertscore_precision": [], "output_bertscore_recall": [],
            "output_sentencebert_similarity": [], "prompt_bertscore_f1": [], "prompt_bertscore_precision": [],
            "prompt_bertscore_recall": [], "prompt_sentencebert_similarity": [],
        }
        if len(result) == 4:
            # Baseline returned (generated_tokens, optimized_inputs, losses, lcs_ratio_history)
            return (*result, empty_prompt_metrics, empty_semantic_metrics)
        elif len(result) == 5:
            # Baseline returned 5-tuple with prompt_metrics_history
            return (*result, empty_semantic_metrics)
        # 6-tuple already includes both
        return result

    if method_lower == "soda":
        from baselines.soda import soda_optimize_inputs
        result = soda_optimize_inputs(
            model=model,
            model_name=baseline_model_name,
            tokenizer=tokenizer,
            device=device,
            target_text=target_text,
            backend=baseline_backend,
            batching="single",  # batch_optimize_main handles iteration
            seq_len=seq_len,
            epochs=epochs,
            learning_rate=learning_rate,
            temperature=temperature,
            decay_rate=soda_decay_rate,
            betas=soda_betas,
            reset_epoch=soda_reset_epoch,
            reinit_epoch=soda_reinit_epoch,
            reg_weight=soda_reg_weight,
            bias_correction=soda_bias_correction,
            init_strategy=soda_init_strategy,
            init_std=soda_init_std,
            batch_size=batch_size,
            ground_truth_prompt_tokens=ground_truth_prompt_tokens,
            kwargs=kwargs,
        )
        return _wrap_baseline_result(result)

    elif method_lower == "gcg":
        from baselines.gcg import gcg_optimize_inputs
        result = gcg_optimize_inputs(
            model=model,
            model_name=baseline_model_name,
            tokenizer=tokenizer,
            device=device,
            target_text=target_text,
            backend=baseline_backend,
            batching="single",
            seq_len=seq_len,
            epochs=epochs,
            num_candidates=gcg_num_candidates,
            top_k=gcg_top_k,
            num_mutations=gcg_num_mutations,
            pos_choice=gcg_pos_choice,
            token_choice=gcg_token_choice,
            init_strategy=gcg_init_strategy,
            batch_size=batch_size,
            kwargs=kwargs,
        )
        return _wrap_baseline_result(result)

    elif method_lower == "o2p":
        from baselines.o2p import o2p_optimize_inputs
        result = o2p_optimize_inputs(
            model=model,
            tokenizer=tokenizer,
            device=device,
            target_text=target_text,
            o2p_model_path=kwargs.get("o2p_model_path"),
            seq_len=seq_len,
            num_beams=kwargs.get("o2p_num_beams", 4),
            max_length=kwargs.get("o2p_max_length", 32),
            batch_size=batch_size,
            kwargs=kwargs,
        )
        return _wrap_baseline_result(result)

    # Continue with STGS/REINFORCE for other methods

    # Enabling Gradient checkpointing:
    if kwargs.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    # Get model's vocabulary size and embedding dimension
    vocab_size = model.config.vocab_size
    hidden_state_dim = model.config.hidden_size

    # Tokenize the target text
    target_tokens = tokenizer(target_text, return_tensors="pt").input_ids.to(device)  # shape: (1, target_length)
    target_length = target_tokens.shape[1]

    # Get embedding matrix from the model
    embedding_layer = model.get_input_embeddings()
    full_embedding_weights = embedding_layer.weight.detach()  # (vocab_size, embed_dim)

    # Compute a seed vector from the target text (average embedding)
    if filter_vocab:
      target_embeds = embedding_layer(target_tokens)  # (1, target_length, embed_dim)
      seed_vector = target_embeds.mean(dim=1).squeeze(0)  # (embed_dim,)

      # Filter full vocabulary based on cosine similarity and ensure target tokens are included
      allowed_tokens = filter_vocabulary(full_embedding_weights, seed_vector, target_tokens[0], threshold=vocab_threshold)
      allowed_vocab_size = allowed_tokens.shape[0]
    else:
      allowed_tokens = torch.arange(vocab_size, device=device)
      allowed_vocab_size = vocab_size
    print(f"Restricted vocabulary size: {allowed_vocab_size} tokens (from full vocab size {full_embedding_weights.shape[0]})")
    # Inverse mapping: original_token_id -> index in allowed_vocab (for rank metric)
    allowed_token_to_idx = {
        token_id: idx
        for idx, token_id in enumerate(allowed_tokens.tolist())
    }

    # --- Fixed logit distribution spec (Approach B: Split Tensor) ---
    has_fixed_spec = any([
        fixed_gt_prefix_n > 0, fixed_gt_suffix_n > 0,
        fixed_gt_prefix_rank2_n > 0, fixed_gt_suffix_rank2_n > 0,
        fixed_prefix_text, fixed_suffix_text,
    ])
    if has_fixed_spec:
        is_prompt_reconstruction = ground_truth_prompt_tokens is not None
        gt_needed = (fixed_gt_prefix_n > 0 or fixed_gt_suffix_n > 0 or
                     fixed_gt_prefix_rank2_n > 0 or fixed_gt_suffix_rank2_n > 0)
        if gt_needed and not is_prompt_reconstruction:
            logging.warning(
                "optimize_inputs: GT-based fixed distribution params are set but "
                "ground_truth_prompt_tokens is None; these will be skipped."
            )
        fixed_prefix, fixed_suffix, n_free = build_fixed_logits_spec(
            ground_truth_prompt_tokens=ground_truth_prompt_tokens,
            allowed_token_to_idx=allowed_token_to_idx,
            allowed_vocab_size=allowed_vocab_size,
            tokenizer=tokenizer,
            device=device,
            seq_len=seq_len,
            fixed_gt_prefix_n=fixed_gt_prefix_n,
            fixed_gt_suffix_n=fixed_gt_suffix_n,
            fixed_gt_prefix_rank2_n=fixed_gt_prefix_rank2_n,
            fixed_gt_suffix_rank2_n=fixed_gt_suffix_rank2_n,
            fixed_prefix_text=fixed_prefix_text,
            fixed_suffix_text=fixed_suffix_text,
        )
        logging.info(
            f"optimize_inputs: fixed spec built — prefix={fixed_prefix.shape[1] if fixed_prefix is not None else 0}, "
            f"suffix={fixed_suffix.shape[1] if fixed_suffix is not None else 0}, n_free={n_free}"
        )
    else:
        fixed_prefix = None
        fixed_suffix = None
        n_free = seq_len

    # W&B log a table of the allowed tokens and the target_text tokens:
    allowed_tokens_list = allowed_tokens.tolist()
    target_tokens_list = target_tokens[0].tolist()
    bias_reference_batch_size_value = (
        stgs_grad_bias_reference_batch_size
        if stgs_grad_bias_reference_batch_size and stgs_grad_bias_reference_batch_size > 0
        else batch_size
    )

    wandb.log({
        "allowed_tokens": allowed_tokens_list,
        "target_tokens": target_tokens_list,
        "allowed_tokens_str": [tokenizer.decode(t) for t in allowed_tokens_list],
        "target_tokens_str": [tokenizer.decode(t) for t in target_tokens_list],
        "allowed_vocab_size": allowed_vocab_size,
        "target_text": target_text,
        "pre_prompt": pre_prompt,
        "seq_len": seq_len,
        "epochs": epochs,
        "losses": losses,
        "learning_rate": learning_rate,
        "bptt":bptt,
        "temperature": temperature,
        "bptt_temperature": bptt_temperature,
        "learnable_temperature": learnable_temperature,
        "decouple_learnable_temperature": decouple_learnable_temperature,
        "bptt_learnable_temperature": bptt_learnable_temperature,
        "bptt_decouple_learnable_temperature": bptt_decouple_learnable_temperature,
        "stgs_hard": stgs_hard,
        "bptt_stgs_hard": bptt_stgs_hard,
        "bptt_hidden_state_conditioning":bptt_hidden_state_conditioning,
        "plot_every": plot_every,
        "eps": eps,
        "bptt_eps": bptt_eps,
        "vocab_threshold": vocab_threshold,
        "batch_size": batch_size,
        "stgs_grad_variance_samples_cfg": stgs_grad_variance_samples,
        "stgs_grad_variance_period_cfg": stgs_grad_variance_period,
        "reinforce_grad_variance_samples_cfg": reinforce_grad_variance_samples,
        "reinforce_grad_variance_period_cfg": reinforce_grad_variance_period,
        "stgs_grad_bias_samples_cfg": stgs_grad_bias_samples,
        "stgs_grad_bias_period_cfg": stgs_grad_bias_period,
        "stgs_grad_bias_reference_samples_cfg": stgs_grad_bias_reference_samples,
        "stgs_grad_bias_reference_batch_size_cfg": bias_reference_batch_size_value,
        "stgs_grad_bias_reference_use_baseline_cfg": stgs_grad_bias_reference_use_baseline,
        "stgs_grad_bias_reference_reward_scale_cfg": stgs_grad_bias_reference_reward_scale,
        "stgs_grad_bias_reference_baseline_beta_cfg": stgs_grad_bias_reference_baseline_beta,
        "gradient_estimator": gradient_estimator,
        "reinforce_reward_scale": reinforce_reward_scale,
        "reinforce_use_baseline": reinforce_use_baseline,
        "reinforce_baseline_beta": reinforce_baseline_beta,
        "teacher_forcing": teacher_forcing,
    })
    wandb_table = wandb.Table(columns=[
        "epoch",
        "target_output_str",
        "ground_truth_prompt",
        "learned_input_ids",
        "learned_input_str",
        "generated_output_ids",
        "generated_output_str",
        "token_overlap_ratio",
        "target_hit_ratio",
    ])

    exact_match_hits_table = wandb.Table(columns=[
        "epoch",
        "target_text",
        "generated_output_str",
        "learned_input_str",
        "loss",
        "lcs_ratio",
        "PLX-prompt",
        "PLX-completion",
        "PLX-prompt-tf-completion",
    ])

    if filter_vocab:
      # Build a mapping from full-vocab token id to allowed index
      allowed_list = allowed_tokens.tolist()
      mapping = { token_id: idx for idx, token_id in enumerate(allowed_list) }

      # Remap target tokens to allowed vocabulary indices using a list comprehension
      target_tokens_mapped = torch.tensor([mapping[t.item()] for t in target_tokens[0]], device=device).unsqueeze(0)

      # Get the subset of the embedding matrix corresponding to allowed tokens
      embedding_weights_subset = full_embedding_weights[allowed_tokens]  # (allowed_vocab_size, embed_dim)
    else:
      target_tokens_mapped = target_tokens
      embedding_weights_subset = full_embedding_weights
    target_tokens_mapped = target_tokens_mapped.reshape(1, -1).repeat(batch_size, 1)
    # Check shape:
    #print(f"Target tokens mapped shape: {target_tokens_mapped.shape}")

    # Initialize learnable inputs (free positions only; fixed parts are separate)
    parameters = []
    free_logits = initialize_learnable_inputs(
        allowed_vocab_size, n_free, device,
        init_strategy=init_strategy,
        init_std=init_std,
        embedding_weights=embedding_weights_subset,
        target_token_ids=target_tokens_mapped[0],
        model=model,
        target_input_ids=target_tokens,
        allowed_tokens=allowed_tokens,
        mlm_model_name=init_mlm_model,
        mlm_top_k=init_mlm_top_k,
        target_text=target_text,
        tokenizer=tokenizer,
    )
    parameters.append(free_logits)

    # Assembly function: build the full (1, seq_len, allowed_vocab_size) logits each step.
    # When no fixed parts exist, returns free_logits directly (zero overhead).
    if fixed_prefix is None and fixed_suffix is None:
        def assemble_logits():
            return free_logits
    else:
        def assemble_logits():
            parts = []
            if fixed_prefix is not None:
                parts.append(fixed_prefix)
            if n_free > 0:
                parts.append(free_logits)
            if fixed_suffix is not None:
                parts.append(fixed_suffix)
            return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
    
    token_overlap_metric = TokenOverlapMetric(
        target_text=target_text,
        tokenizer=tokenizer,
    )

    # Set up loss function
    loss_instance = LossClass(
        model=model,
        tokenizer=tokenizer,
        embedding_weights_subset=embedding_weights_subset,
        losses=losses,
        target_text=target_text,
        target_tokens_mapped=target_tokens_mapped,
        kwargs=kwargs,
    )

    estimator_choice = gradient_estimator.lower()
    stgs_module = None
    bptt_stgs_module = None
    reinforce_helper = None

    if estimator_choice == "stgs":
        stgs_module = STGS(
            vocab_size=allowed_vocab_size,
            stgs_hard=stgs_hard,
            init_temperature=temperature,
            learnable_temperature=learnable_temperature,
            nbr_learnable_temperatures=seq_len if decouple_learnable_temperature else None,
            eps=eps,
            device=device,
        )
        parameters += list(stgs_module.parameters())

        if bptt:
            bptt_stgs_module = STGS(
                vocab_size=allowed_vocab_size,
                stgs_hard=bptt_stgs_hard,
                init_temperature=bptt_temperature,
                learnable_temperature=bptt_learnable_temperature,
                nbr_learnable_temperatures=target_length if bptt_decouple_learnable_temperature else None,
                eps=bptt_eps,
                conditioning_dim=hidden_state_dim if bptt_hidden_state_conditioning else 0,
                device=device,
            )
            parameters += list(bptt_stgs_module.parameters())
    elif estimator_choice == "reinforce":
        if bptt:
            raise ValueError("BPTT is currently not supported with the REINFORCE estimator.")
        reinforce_helper = ReinforceEstimator(
            reward_scale=kwargs.get("reinforce_reward_scale", reinforce_reward_scale),
            use_baseline=kwargs.get("reinforce_use_baseline", reinforce_use_baseline),
            baseline_beta=kwargs.get("reinforce_baseline_beta", reinforce_baseline_beta),
        )
    else:
        raise ValueError(f"Unknown gradient estimator '{gradient_estimator}'.")

    optimizer = optim.Adam(parameters, lr=learning_rate)

    if estimator_choice == "stgs":
        _stgs_base = partial(
            stgs_forward_pass,
            model=model,
            tokenizer=tokenizer,
            loss_instance=loss_instance,
            stgs_module=stgs_module,
            embedding_weights_subset=embedding_weights_subset,
            allowed_tokens=allowed_tokens,
            target_length=target_length,
            batch_size=batch_size,
            device=device,
            model_precision=kwargs.get("model_precision", "full"),
            pre_prompt=pre_prompt,
            bptt=bptt,
            bptt_stgs=bptt_stgs_module,
            filter_vocab=filter_vocab,
            teacher_forcing=teacher_forcing,
            target_tokens=target_tokens_mapped,
            compute_tf_completion_logits=('promptTfComplPerplexity' in losses),
        )
        def forward_pass_callable():
            return _stgs_base(learnable_inputs=assemble_logits())
    else:
        _reinforce_base = partial(
            reinforce_forward_pass,
            model=model,
            tokenizer=tokenizer,
            loss_instance=loss_instance,
            embedding_weights_subset=embedding_weights_subset,
            allowed_tokens=allowed_tokens,
            target_length=target_length,
            batch_size=batch_size,
            device=device,
            model_precision=kwargs.get("model_precision", "full"),
            pre_prompt=pre_prompt,
            filter_vocab=filter_vocab,
            reinforce_helper=reinforce_helper,
            target_tokens=target_tokens_mapped,
            compute_tf_completion_logits=('promptTfComplPerplexity' in losses),
        )
        def forward_pass_callable():
            return _reinforce_base(learnable_inputs=assemble_logits())

    estimator_is_stgs = estimator_choice == "stgs"

    # Training loop
    losses = []
    lcs_ratio_history = []
    # Per-epoch prompt reconstruction metrics (SODA-style)
    prompt_metrics_history = {
        "prompt_token_accuracy": [],
        "prompt_exact_match": [],
        "prompt_lcs_ratio": [],
        "prompt_cosine_similarity": [],
        "prompt_gt_token_rank_mean": [],         # mean rank (scalar per epoch)
        "prompt_gt_token_rank_per_position": [],  # list of per-pos ranks (list per epoch)
    }
    track_prompt_metrics = ground_truth_prompt_tokens is not None

    # Per-epoch semantic metrics (BERT/SentenceBERT)
    semantic_metrics_history = {
        "output_bertscore_f1": [],
        "output_bertscore_precision": [],
        "output_bertscore_recall": [],
        "output_sentencebert_similarity": [],
        "prompt_bertscore_f1": [],
        "prompt_bertscore_precision": [],
        "prompt_bertscore_recall": [],
        "prompt_sentencebert_similarity": [],
    }
    track_semantic_metrics = semantic_metrics_every_n_epochs > 0
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        optimizer.zero_grad()

        step_result = forward_pass_callable()
        loss = step_result.loss
        backward_loss = step_result.backward_loss
        losses_dict = step_result.losses_dict
        generated_logits = step_result.generated_logits
        estimator_state = step_result.estimator_state

        backward_loss.backward()

        estimator_metrics: Dict[str, float] = {}
        if estimator_is_stgs:
            # Save gradients for all parameters  in the context:
            saved_grads = {
                param: param.grad.clone() 
                for param in parameters
            }

            measure_grad_variance = (
                stgs_grad_variance_samples >= 2
                and stgs_grad_variance_period > 0
                and (epoch % stgs_grad_variance_period == 0)
            )
            measure_grad_bias = (
                stgs_grad_bias_samples >= 1
                and stgs_grad_bias_reference_samples >= 1
                and stgs_grad_bias_period > 0
                and (epoch % stgs_grad_bias_period == 0)
            )

            need_baseline_grad = (
                (measure_grad_variance or measure_grad_bias)
                and free_logits.grad is not None
            )
            baseline_grad = None
            if need_baseline_grad:
                baseline_grad = free_logits.grad.detach().clone()

            stgs_grad_samples = []
            if measure_grad_variance and baseline_grad is not None:
                logging.info(f"Epoch {epoch}: Estimating STGS gradient variance with {stgs_grad_variance_samples} samples...")
                estimator_metrics.update(
                    estimate_stgs_gradient_variance(
                        stgs_grad_variance_samples,
                        baseline_grad,
                        forward_pass_callable,
                        free_logits,
                    )
                )
                stgs_grad_samples = estimator_metrics.pop("stgs_grad_samples")

            if measure_grad_bias and baseline_grad is not None:
                logging.info(f"Epoch {epoch}: Estimating STGS gradient bias with {stgs_grad_bias_samples} STGS samples and {stgs_grad_bias_reference_samples} REINFORCE samples...")
                bias_reinforce_helper = ReinforceEstimator(
                    reward_scale=stgs_grad_bias_reference_reward_scale,
                    use_baseline=stgs_grad_bias_reference_use_baseline,
                    baseline_beta=stgs_grad_bias_reference_baseline_beta,
                )
                _bias_reinforce_base = partial(
                    reinforce_forward_pass,
                    model=model,
                    tokenizer=tokenizer,
                    loss_instance=loss_instance,
                    embedding_weights_subset=embedding_weights_subset,
                    allowed_tokens=allowed_tokens,
                    target_length=target_length,
                    batch_size=batch_size, # DEPR: bias_reference_batch_size_value,
                    device=device,
                    model_precision=kwargs.get("model_precision", "full"),
                    pre_prompt=pre_prompt,
                    filter_vocab=filter_vocab,
                    reinforce_helper=bias_reinforce_helper,
                    target_tokens=target_tokens_mapped,
                    compute_tf_completion_logits=('promptTfComplPerplexity' in losses),
                )
                def bias_reinforce_forward(**_kw):
                    return _bias_reinforce_base(learnable_inputs=assemble_logits(), **_kw)

                bias_metrics = estimate_stgs_gradient_bias(
                    stgs_num_samples=stgs_grad_bias_samples,
                    reinforce_num_samples=stgs_grad_bias_reference_samples,
                    baseline_grad=baseline_grad,
                    stgs_forward_pass_fn=forward_pass_callable,
                    reinforce_forward_pass_fn=bias_reinforce_forward,
                    learnable_inputs=free_logits,
                    reinforce_update_baseline=stgs_grad_bias_reference_use_baseline,
                    stgs_grad_samples=stgs_grad_samples,
                )
                estimator_metrics.update(bias_metrics)
            
            # Restore gradients for all parameters
            for param, saved_grad in saved_grads.items():
                param.grad = saved_grad
        else:
            # Save gradients for all parameters in the context:
            saved_grads = {
                param: param.grad.clone() 
                for param in parameters
            }
            
            measure_grad_variance = (
                reinforce_grad_variance_samples >= 2
                and reinforce_grad_variance_period > 0
                and (epoch % reinforce_grad_variance_period == 0)
            )
            baseline_grad = None
            if measure_grad_variance and free_logits.grad is not None:
                baseline_grad = free_logits.grad.detach().clone()
            if measure_grad_variance and baseline_grad is not None:
                estimator_metrics.update(
                    estimate_reinforce_gradient_variance(
                        reinforce_grad_variance_samples,
                        baseline_grad,
                        forward_pass_callable,
                        free_logits,
                    )
                )
            
            # Restore gradients for all parameters
            for param, saved_grad in saved_grads.items():
                param.grad = saved_grad

        if max_gradient_norm != 0.0:
            torch.nn.utils.clip_grad_norm_(parameters, max_gradient_norm)

        optimizer.step()

        grad_tensor = free_logits.grad
        grad_norm = grad_tensor.norm().item() if grad_tensor is not None else 0.0
        non_zero_grads = int((grad_tensor != 0).sum().item()) if grad_tensor is not None else 0
        grad_mean = grad_tensor.mean().item() if grad_tensor is not None else 0.0
        grad_max = grad_tensor.max().item() if grad_tensor is not None else 0.0
        grad_std = grad_tensor.std().item() if grad_tensor is not None else 0.0
        if grad_tensor is not None:
            grad_abs = grad_tensor.abs()
            grad_min = grad_abs[grad_abs > 0].min().item() if (grad_abs > 0).any() else 0.0
        else:
            grad_min = 0.0

        eff_temperature = estimator_state.get("eff_temperature")
        bptt_eff_temperature = estimator_state.get("bptt_eff_temperature")
        reinforce_baseline_state = estimator_state.get("reinforce_baseline")

        info_components = [f"Gradient norm: {grad_norm:.6f}", f"Allowed vocab size: {allowed_vocab_size}"]
        if torch.is_tensor(eff_temperature):
            info_components.insert(1, f"Tau = {eff_temperature.mean().item():.6f}")
        if reinforce_baseline_state is not None and torch.is_tensor(reinforce_baseline_state):
            info_components.append(f"Baseline: {reinforce_baseline_state.item():.6f}")
        info = " / ".join(info_components)

        losses.append(loss.item())
        pbar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f} / {info}")

        wandb_log = {
            "epoch": epoch + 1,
            "loss": loss.item(),
            "allowed_vocab_size": allowed_vocab_size,
            "non_zero_grads": non_zero_grads,
            "grad_mean": grad_mean,
            "grad_max": grad_max,
            "grad_min": grad_min,
            "grad_norm": grad_norm,
            "grad_std": grad_std,
            "vocab_size": model.config.vocab_size,
        }

        for k, v in losses_dict.items():
            wandb_log[k] = v.item()

        if estimator_is_stgs:
            if torch.is_tensor(eff_temperature):
                wandb_log["effective_temperature"] = eff_temperature.mean().item()
                if decouple_learnable_temperature:
                    for sidx in range(eff_temperature.shape[1]):
                        wandb_log[f"effective_temperature_{sidx}"] = eff_temperature[:, sidx].mean().item()
            if torch.is_tensor(bptt_eff_temperature):
                wandb_log["bptt_effective_temperature"] = bptt_eff_temperature.mean().item()
                if bptt_decouple_learnable_temperature:
                    for sidx in range(bptt_eff_temperature.shape[1]):
                        wandb_log[f"bptt_effective_temperature_{sidx}"] = bptt_eff_temperature[:, sidx].mean().item()
            elif isinstance(bptt_eff_temperature, (float, int)):
                wandb_log["bptt_effective_temperature"] = float(bptt_eff_temperature)
            wandb_log.update(estimator_metrics)
        else:
            advantage_val = estimator_state.get("reinforce_advantage")
            if torch.is_tensor(advantage_val):
                wandb_log["reinforce_advantage"] = advantage_val.item()
            baseline_val = estimator_state.get("reinforce_baseline")
            if torch.is_tensor(baseline_val):
                wandb_log["reinforce_baseline"] = baseline_val.item()
            log_prob_mean = estimator_state.get("reinforce_log_prob_mean")
            if torch.is_tensor(log_prob_mean):
                wandb_log["reinforce_log_prob_mean"] = log_prob_mean.item()
            log_prob_sum_mean = estimator_state.get("reinforce_log_prob_sum_mean")
            if torch.is_tensor(log_prob_sum_mean):
                wandb_log["reinforce_log_prob_sum_mean"] = log_prob_sum_mean.item()
            wandb_log.update(estimator_metrics)

        # Update wandb_table with generated_output:
        learnable_input_ids = torch.argmax(assemble_logits().detach(), dim=-1)[0]
        generated_output_ids = torch.argmax(generated_logits, dim=-1)
        table_generated_output_ids = torch.gather(allowed_tokens.unsqueeze(0), dim=1, index=generated_output_ids[0:1])
        learnable_input_str = tokenizer.decode(learnable_input_ids, skip_special_tokens=False)
        generated_output_str = tokenizer.decode(table_generated_output_ids[0], skip_special_tokens=False)

        generated_tokens = table_generated_output_ids[0].cpu().tolist()
        target_tokens_list = target_tokens[0].cpu().tolist()

        # Compute LCS ratio between generated output and target
        lcs = lcs_length(generated_tokens, target_tokens_list)
        lcs_ratio = lcs / len(target_tokens_list) if len(target_tokens_list) > 0 else 0.0
        wandb_log['lcs_ratio'] = lcs_ratio
        lcs_ratio_history.append(lcs_ratio)

        # Compute prompt reconstruction metrics (SODA-style) if ground truth is available
        if track_prompt_metrics:
            optimized_prompt_tokens = learnable_input_ids.cpu().tolist()
            gt_prompt_len = len(ground_truth_prompt_tokens)

            # Prompt token accuracy (partial match)
            min_len = min(len(optimized_prompt_tokens), gt_prompt_len)
            matching = sum(
                optimized_prompt_tokens[i] == ground_truth_prompt_tokens[i]
                for i in range(min_len)
            )
            prompt_token_acc = matching / gt_prompt_len if gt_prompt_len > 0 else 0.0
            prompt_metrics_history["prompt_token_accuracy"].append(prompt_token_acc)
            wandb_log['prompt_token_accuracy'] = prompt_token_acc

            # Prompt exact match
            prompt_exact = int(optimized_prompt_tokens[:gt_prompt_len] == ground_truth_prompt_tokens)
            prompt_metrics_history["prompt_exact_match"].append(prompt_exact)
            wandb_log['prompt_exact_match'] = prompt_exact

            # Prompt LCS ratio
            prompt_lcs = lcs_length(optimized_prompt_tokens, ground_truth_prompt_tokens)
            prompt_lcs_ratio = prompt_lcs / gt_prompt_len if gt_prompt_len > 0 else 0.0
            prompt_metrics_history["prompt_lcs_ratio"].append(prompt_lcs_ratio)
            wandb_log['prompt_lcs_ratio'] = prompt_lcs_ratio

            # Prompt cosine similarity (using embeddings)
            with torch.no_grad():
                opt_tokens_tensor = torch.tensor(optimized_prompt_tokens[:gt_prompt_len], dtype=torch.long, device=device)
                gt_tokens_tensor = torch.tensor(ground_truth_prompt_tokens, dtype=torch.long, device=device)

                embedding_layer = model.get_input_embeddings()
                opt_emb = embedding_layer(opt_tokens_tensor)
                gt_emb = embedding_layer(gt_tokens_tensor)

                # Normalize and compute cosine similarity
                opt_norm = opt_emb / (opt_emb.norm(dim=-1, keepdim=True) + 1e-9)
                gt_norm = gt_emb / (gt_emb.norm(dim=-1, keepdim=True) + 1e-9)
                cos_sim = (opt_norm * gt_norm).sum(dim=-1).mean().item()

            prompt_metrics_history["prompt_cosine_similarity"].append(cos_sim)
            wandb_log['prompt_cosine_similarity'] = cos_sim

            # Compute GT token rank from learnable logits distribution (full assembled view)
            with torch.no_grad():
                logits_per_pos = assemble_logits()[0]  # (seq_len, allowed_vocab_size)
                prompt_seq_len = min(logits_per_pos.shape[0], gt_prompt_len)
                per_position_ranks = []
                for pos in range(prompt_seq_len):
                    gt_token_id = ground_truth_prompt_tokens[pos]
                    pos_logits = logits_per_pos[pos]
                    if gt_token_id in allowed_token_to_idx:
                        allowed_idx = allowed_token_to_idx[gt_token_id]
                        gt_logit = pos_logits[allowed_idx]
                        rank = int((pos_logits > gt_logit).sum().item()) + 1  # 1-indexed
                    else:
                        rank = int(pos_logits.shape[0]) + 1  # Not in allowed vocab → worst rank
                    per_position_ranks.append(rank)
            gt_rank_mean = (
                sum(per_position_ranks) / len(per_position_ranks) if per_position_ranks else float('nan')
            )
            prompt_metrics_history["prompt_gt_token_rank_mean"].append(gt_rank_mean)
            prompt_metrics_history["prompt_gt_token_rank_per_position"].append(per_position_ranks)
            wandb_log['prompt_gt_token_rank_mean'] = gt_rank_mean
            for pos, rank in enumerate(per_position_ranks):
                wandb_log[f'prompt_gt_token_rank_pos_{pos}'] = rank

        # Compute semantic metrics (BERT/SentenceBERT) at specified intervals
        if track_semantic_metrics and (epoch % semantic_metrics_every_n_epochs == 0 or epoch == epochs - 1):
            from evaluation_utils import compute_bertscore, compute_sentencebert_similarity, get_cached_sentencebert

            # Ensure SentenceBERT model is cached on first call
            _ = get_cached_sentencebert(sentencebert_model_name, str(device))

            # Output metrics: generated vs target
            output_bert = compute_bertscore(generated_output_str, target_text,
                                             model_type=bertscore_model_type, device=str(device))
            output_sbert = compute_sentencebert_similarity(generated_output_str, target_text,
                                                            model_name=sentencebert_model_name, device=str(device))

            for key, val in output_bert.items():
                semantic_metrics_history[f"output_{key}"].append(float(val))
                wandb_log[f"output_{key}"] = float(val)
            output_sbert_val = float(output_sbert["sentencebert_similarity"])
            semantic_metrics_history["output_sentencebert_similarity"].append(output_sbert_val)
            wandb_log["output_sentencebert_similarity"] = output_sbert_val

            # Prompt metrics: optimized prompt vs ground-truth (if available)
            if track_prompt_metrics and ground_truth_prompt is not None:
                optimized_prompt_text = tokenizer.decode(learnable_input_ids, skip_special_tokens=True)
                prompt_bert = compute_bertscore(optimized_prompt_text, ground_truth_prompt,
                                                 model_type=bertscore_model_type, device=str(device))
                prompt_sbert = compute_sentencebert_similarity(optimized_prompt_text, ground_truth_prompt,
                                                                model_name=sentencebert_model_name, device=str(device))

                for key, val in prompt_bert.items():
                    semantic_metrics_history[f"prompt_{key}"].append(float(val))
                    wandb_log[f"prompt_{key}"] = float(val)
                prompt_sbert_val = float(prompt_sbert["sentencebert_similarity"])
                semantic_metrics_history["prompt_sentencebert_similarity"].append(prompt_sbert_val)
                wandb_log["prompt_sentencebert_similarity"] = prompt_sbert_val

        metrics_dict = token_overlap_metric.measure(
            prompt_tokens=learnable_input_ids,
        )
        for k, v in metrics_dict.items():
            wandb_log[k] = v

        wandb_table.add_data(
            epoch+1,
            target_text,
            ground_truth_prompt or "",
            learnable_input_ids.tolist(),
            learnable_input_str,
            table_generated_output_ids[0].tolist(),
            generated_output_str,
            *metrics_dict.values(),
        )

        wandb.log(wandb_log)
        
        if epoch % log_table_every == 0:
            wandb.log({"generated_output_table": copy.deepcopy(wandb_table)})

         # Update plot every plot_every epochs - Colab compatible version
        if epoch % plot_every == 0 or epoch == epochs - 1:
            # Clear the output and create a new plot each time
            #clear_output(wait=True)

            # Create a new figure
            plt.figure(figsize=(10, 5))
            plt.plot(losses)
            plt.title('Training Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.grid(True)

            # Save and display the plot
            plt.savefig('loss_curve_latest.png')
            #display(plt.gcf())  # This shows the plot in Colab

        if epoch > 0 and epoch % (plot_every * 5) == 0:
            with torch.no_grad():
                curr_input_embeddings = torch.matmul(assemble_logits(), embedding_weights_subset)
                generated_tokens, _ = generate_completions(
                    model,
                    curr_input_embeddings,
                    target_length=target_length,
                    pre_prompt=pre_prompt,
                )
                intermediate_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                print(f"\nEpoch {epoch} intermediate output: {intermediate_text}")
                print(f"Target: {target_text}")

        # Track exact-match hits regardless of early-stopping
        if generated_output_str == target_text:
            exact_match_hits_table.add_data(
                epoch + 1,
                target_text,
                generated_output_str,
                learnable_input_str,
                loss.item(),
                lcs_ratio,
                losses_dict["PLX-prompt"].item(),
                losses_dict["PLX-completion"].item(),
                losses_dict["PLX-prompt-tf-completion"].item()
                    if "PLX-prompt-tf-completion" in losses_dict else None,
            )
            wandb.log({"exact_match_hits": copy.deepcopy(exact_match_hits_table)})

        # Optional: early stopping condition
        if (early_stop_loss_threshold > 0.0 and loss.item() < early_stop_loss_threshold) \
        or (early_stop_on_exact_match and generated_output_str == target_text):
            print(f"Converged at epoch {epoch+1} with loss: {loss.item():.6f}")
            wandb.log({"generated_output_table": copy.deepcopy(wandb_table)})
            break

    # Return the full assembled logits (detached) for downstream use
    return generated_tokens, assemble_logits().detach(), losses, lcs_ratio_history, prompt_metrics_history, semantic_metrics_history


def str2bool(instr):
    if isinstance(instr, bool):
        return instr
    if isinstance(instr, str):
        instr = instr.lower()
        if 'true' in instr:
            return True
        elif 'false' in instr:
            return False
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

def intOrNone(instr):
    if instr is None:
        return None
    return int(instr)

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('PromptOptimisationSTGSBenchmark')

    parser = argparse.ArgumentParser(description="Prompt Optimisation - STGS - Test.")
 
    #model_name = "HuggingFaceM4/tiny-random-LlamaForCausalLM"  # For example, using a small LLaMA model
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--model_precision", type=str, default="full")
    # full
    # half
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=False)
    parser.add_argument("--target_text", type=str, default="The quick brown fox jumps over the lazy dog")
    parser.add_argument("--pre_prompt", type=str, default=None)
    #pre_prompt = "Complete the following: "  # Can be None if not needed
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=2000)
    #parser.add_argument("--losses", type=str, default="crossentropy")
    #parser.add_argument("--losses", type=str, default="embedded")
    parser.add_argument("--losses", type=str, default="embxentropy")
    # embLayerxL2
    # +embedded
    # +embxentropy
    # +perplexityPenalty
    parser.add_argument("--promptLambda", type=float, default=0.0)
    parser.add_argument("--complLambda", type=float, default=0.0)
    parser.add_argument("--promptTfComplLambda", type=float, default=0.0,
                        help="Weight for prompt + teacher-forced completion perplexity loss")
    parser.add_argument("--learning_rate", type=float, default=1e-1)
    parser.add_argument("--max_gradient_norm", type=float, default=0.0)
    parser.add_argument("--eps", type=float, default=1e-10)
    parser.add_argument("--bptt_eps", type=float, default=1e-10)
    parser.add_argument("--temperature", type=float, default=1e1)
    parser.add_argument("--learnable_temperature", type=str2bool, default=False)
    parser.add_argument("--stgs_hard", type=str2bool, default=False)
    parser.add_argument("--bptt", type=str2bool, default=False)
    parser.add_argument("--bptt_temperature", type=float, default=1e1)
    parser.add_argument("--bptt_learnable_temperature", type=str2bool, default=False)
    parser.add_argument("--bptt_stgs_hard", type=str2bool, default=False)
    parser.add_argument("--bptt_hidden_state_conditioning", type=str2bool, default=False)
    parser.add_argument("--plot_every", type=int, default=100000)
    parser.add_argument("--stgs_grad_variance_samples", type=int, default=0)
    parser.add_argument("--stgs_grad_variance_period", type=int, default=1)
    parser.add_argument("--stgs_grad_bias_samples", type=int, default=0)
    parser.add_argument("--stgs_grad_bias_period", type=int, default=1)
    parser.add_argument("--stgs_grad_bias_reference_samples", type=int, default=0)
    parser.add_argument("--stgs_grad_bias_reference_batch_size", type=int, default=0)
    parser.add_argument("--stgs_grad_bias_reference_use_baseline", type=str2bool, default=True)
    parser.add_argument("--stgs_grad_bias_reference_reward_scale", type=float, default=1.0)
    parser.add_argument("--stgs_grad_bias_reference_baseline_beta", type=float, default=0.9)
    parser.add_argument("--reinforce_grad_variance_samples", type=int, default=0)
    parser.add_argument("--reinforce_grad_variance_period", type=int, default=1)
    parser.add_argument("--gradient_estimator", type=str, default="stgs", choices=["stgs", "reinforce"])
    parser.add_argument("--reinforce_reward_scale", type=float, default=1.0)
    parser.add_argument("--reinforce_use_baseline", type=str2bool, default=True)
    parser.add_argument("--reinforce_baseline_beta", type=float, default=0.9)
    parser.add_argument("--filter_vocab", type=str2bool, default= True)
    parser.add_argument("--vocab_threshold", type=float, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--early_stop_on_exact_match", type=str2bool, default=True,
                        help="Stop optimization when generated output exactly matches target (default: True)")
    parser.add_argument("--early_stop_loss_threshold", type=float, default=0.01,
                        help="Stop optimization when loss drops below this threshold (0 or negative to disable)")

    # Fixed logit distribution parameters
    parser.add_argument("--fixed_gt_prefix_n", type=int, default=0,
                        help="Fix first N prompt positions to GT token at rank 1 (requires prompt reconstruction)")
    parser.add_argument("--fixed_gt_suffix_n", type=int, default=0,
                        help="Fix last N prompt positions to GT token at rank 1 (requires prompt reconstruction)")
    parser.add_argument("--fixed_gt_prefix_rank2_n", type=int, default=0,
                        help="Fix next N prefix positions to GT token at rank 2 (after fixed_gt_prefix_n)")
    parser.add_argument("--fixed_gt_suffix_rank2_n", type=int, default=0,
                        help="Fix N positions before fixed_gt_suffix_n to GT token at rank 2")
    parser.add_argument("--fixed_prefix_text", type=str, default=None,
                        help="Text string whose tokens form a fixed one-hot prefix")
    parser.add_argument("--fixed_suffix_text", type=str, default=None,
                        help="Text string whose tokens form a fixed one-hot suffix")

    args = parser.parse_args()
    config = vars(args)
   
    if config['promptLambda'] > 0.0:    config['losses'] += '+promptPerplexity'
    if config['complLambda'] > 0.0:    config['losses'] += '+completionPerplexity'
    if config['promptTfComplLambda'] > 0.0:    config['losses'] += '+promptTfComplPerplexity'

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Load model and tokenizer
    print(f"Loading model: {config['model_name']}")
    model, tokenizer = setup_model_and_tokenizer(config['model_name'],device=device, model_precision=config['model_precision'])

    config['vocab_size'] = model.config.vocab_size
    config['hidden_size'] = model.config.hidden_size
    
    wandb_run = wandb.init(project="prompt-optimization", config=config)
    
    torch.manual_seed(config["seed"])
    # Optimize inputs
    print(f"Starting optimization with target: {args.target_text}")
    generated_tokens, optimized_inputs, losses, lcs_ratio_history, _, _ = optimize_inputs(
        model,
        tokenizer,
        losses=config['losses'],
        bptt=config['bptt'],
        device=device,
        target_text=config['target_text'],
        pre_prompt=config['pre_prompt'],
        seq_len=config['seq_len'],
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        temperature=config['temperature'],
        bptt_temperature=config['bptt_temperature'],
        bptt_learnable_temperature=config['bptt_learnable_temperature'],
        learnable_temperature=config['learnable_temperature'],
        stgs_hard=config['stgs_hard'],
        bptt_stgs_hard=config['bptt_stgs_hard'],
        bptt_hidden_state_conditioning=config['bptt_hidden_state_conditioning'],
        plot_every=config['plot_every'],
        stgs_grad_variance_samples=config['stgs_grad_variance_samples'],
        stgs_grad_variance_period=config['stgs_grad_variance_period'],
        stgs_grad_bias_samples=config['stgs_grad_bias_samples'],
        stgs_grad_bias_period=config['stgs_grad_bias_period'],
        stgs_grad_bias_reference_samples=config['stgs_grad_bias_reference_samples'],
        stgs_grad_bias_reference_batch_size=config['stgs_grad_bias_reference_batch_size'],
        stgs_grad_bias_reference_use_baseline=config['stgs_grad_bias_reference_use_baseline'],
        stgs_grad_bias_reference_reward_scale=config['stgs_grad_bias_reference_reward_scale'],
        stgs_grad_bias_reference_baseline_beta=config['stgs_grad_bias_reference_baseline_beta'],
        reinforce_grad_variance_samples=config['reinforce_grad_variance_samples'],
        reinforce_grad_variance_period=config['reinforce_grad_variance_period'],
        gradient_estimator=config['gradient_estimator'],
        reinforce_reward_scale=config['reinforce_reward_scale'],
        reinforce_use_baseline=config['reinforce_use_baseline'],
        reinforce_baseline_beta=config['reinforce_baseline_beta'],
        eps=config['eps'],
        bptt_eps=config['bptt_eps'],
        vocab_threshold=config['vocab_threshold'],
        filter_vocab=config['filter_vocab'],
        max_gradient_norm=config['max_gradient_norm'],
        batch_size=config['batch_size'],
        fixed_gt_prefix_n=config.get('fixed_gt_prefix_n', 0),
        fixed_gt_suffix_n=config.get('fixed_gt_suffix_n', 0),
        fixed_gt_prefix_rank2_n=config.get('fixed_gt_prefix_rank2_n', 0),
        fixed_gt_suffix_rank2_n=config.get('fixed_gt_suffix_rank2_n', 0),
        fixed_prefix_text=config.get('fixed_prefix_text'),
        fixed_suffix_text=config.get('fixed_suffix_text'),
        early_stop_on_exact_match=config.get('early_stop_on_exact_match', True),
        early_stop_loss_threshold=config.get('early_stop_loss_threshold', 0.01),
        kwargs=config,
    )

    wandb_run.finish()


if __name__ == '__main__':
    main()
