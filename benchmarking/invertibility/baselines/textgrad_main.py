import argparse
import copy
import logging
from typing import Optional

import textgrad as tg
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import wandb


def setup_model_and_tokenizer(
    model_name: str = "HuggingFaceM4/tiny-random-LlamaForCausalLM",
    device: str = "cpu",
    model_precision: str = "full",
):
    """
    Load a causal LM and its tokenizer, mirroring benchmarking/main.py.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    model = model.to(device)
    if model_precision == "half":
        model.half()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def filter_vocabulary(embedding_weights, seed_vector, target_token_ids, threshold=0.5):
    """
    Same cosine-similarity filter used by the STGS baseline.
    """
    norm_weights = embedding_weights / (embedding_weights.norm(dim=1, keepdim=True) + 1e-9)
    norm_seed = seed_vector / (seed_vector.norm() + 1e-9)
    similarities = torch.matmul(norm_weights, norm_seed.unsqueeze(1)).squeeze(1)
    allowed = (similarities >= threshold).nonzero(as_tuple=True)[0]

    target_set = set(target_token_ids.tolist())
    allowed_set = set(allowed.tolist())
    union_allowed = allowed_set.union(target_set)
    union_allowed = torch.tensor(sorted(list(union_allowed)), device=embedding_weights.device)
    return union_allowed


class TokenOverlapMetric:
    def __init__(self, target_text: str, tokenizer):
        self.target_text = target_text
        self.tokenizer = tokenizer

    def measure(self, prompt_text: Optional[str] = None, prompt_tokens=None):
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
            prompt_tokens = torch.tensor(prompt_tokens)

        tt_occ = {}
        for ttoken in target_tokens:
            tt_occ[ttoken.item()] = (prompt_tokens == ttoken).int().sum().item()

        nbr_occ = sum(tt_occ.values())
        max_occ = prompt_tokens.shape[-1] if prompt_tokens.shape[-1] > 0 else 1

        overlap_ratio = nbr_occ / max_occ
        output_dict["token_overlap_ratio"] = overlap_ratio

        target_set = set(target_tokens.tolist())
        target_set_size = max(len(target_set), 1)
        target_hits = {
            ttoken: int(t_occ > 0)
            for ttoken, t_occ in tt_occ.items()
        }
        target_hits_size = sum(target_hits.values())
        output_dict["target_hit_ratio"] = float(target_hits_size) / target_set_size

        return output_dict


def sample_initial_prompt(tokenizer, allowed_tokens: torch.Tensor, seq_len: int, seed: int):
    """
    Draw a random prompt from the allowed token set to mimic the STGS initialisation.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    allowed_cpu = allowed_tokens.cpu()
    indices = torch.randint(
        low=0,
        high=allowed_cpu.shape[0],
        size=(seq_len,),
        generator=generator,
    )
    sampled_token_ids = allowed_cpu[indices].tolist()
    prompt_text = tokenizer.decode(sampled_token_ids, skip_special_tokens=False)
    return prompt_text, sampled_token_ids


def generate_model_completion(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
):
    """
    Generate a completion from the frozen local model.
    """
    encoding = tokenizer(prompt, return_tensors="pt")
    input_ids = encoding.input_ids.to(device)
    attention_mask = encoding.attention_mask.to(device)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "attention_mask": attention_mask,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": max(temperature, 1e-6),
            }
        )
    else:
        generation_kwargs["do_sample"] = False

    with torch.no_grad():
        output_ids = model.generate(input_ids, **generation_kwargs)

    generated_ids = output_ids[:, input_ids.shape[-1]:]
    completion_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    return completion_text, generated_ids[0].detach().cpu()


def optimize_inputs(
    model,
    tokenizer,
    device,
    losses="textgrad",
    bptt=False,
    target_text: str = "The quick brown fox jumps over the lazy dog",
    pre_prompt: Optional[str] = None,
    seq_len: int = 20,
    epochs: int = 200,
    learning_rate: float = 0.0,
    temperature: float = 1.0,
    bptt_temperature: float = 1.0,
    bptt_learnable_temperature: bool = False,
    learnable_temperature: bool = False,
    stgs_hard: bool = False,
    bptt_stgs_hard: bool = False,
    bptt_hidden_state_conditioning: bool = False,
    plot_every: int = 100000,
    log_table_every: int = 1,
    eps: float = 1e-10,
    bptt_eps: float = 1e-10,
    vocab_threshold: float = 0.5,
    filter_vocab: bool = False,
    max_gradient_norm: float = 0.0,
    batch_size: int = 1,
    kwargs: Optional[dict] = None,
):
    """
    TextGrad-based optimisation counterpart of benchmarking/main.py::optimize_inputs.
    The signature is kept close to the STGS version so downstream scripts can swap files.
    """
    del losses, bptt, learning_rate, bptt_temperature, bptt_learnable_temperature
    del learnable_temperature, stgs_hard, bptt_stgs_hard, bptt_hidden_state_conditioning
    del plot_every, eps, bptt_eps, max_gradient_norm, batch_size

    if kwargs is None:
        kwargs = {}

    seed = int(kwargs.get("seed", 42))
    torch.manual_seed(seed)

    backward_engine = kwargs.get("backward_engine", "gpt-4o")
    optimizer_engine = kwargs.get("optimizer_engine") or backward_engine
    textgrad_constraints = kwargs.get("textgrad_constraints") or []
    if isinstance(textgrad_constraints, str):
        textgrad_constraints = [textgrad_constraints] if textgrad_constraints else []
    textgrad_system_prompt = kwargs.get(
        "textgrad_system_prompt",
        (
            "You are an expert prompt engineer. "
            "Given a prompt body, the fixed prefix (if any), the model output, and the desired target, "
            "give concise, actionable feedback (max 3 bullet points) to edit ONLY the prompt body so that "
            "future outputs match the target exactly. If the output already matches, reply with 'DONE'."
        ),
    )
    textgrad_verbose = int(kwargs.get("textgrad_verbose", 0))
    stop_on_done = bool(kwargs.get("stop_on_done", True))
    wandb_project = kwargs.get("wandb_project", "prompt-optimization-textgrad")
    textgrad_do_sample = bool(kwargs.get("textgrad_do_sample", True))

    target_tokens = tokenizer(target_text, return_tensors="pt").input_ids.to(device)
    target_length = target_tokens.shape[1]
    max_new_tokens = kwargs.get("max_new_tokens") or target_length

    embedding_layer = model.get_input_embeddings()
    full_embedding_weights = embedding_layer.weight.detach()
    if filter_vocab:
        target_embeds = embedding_layer(target_tokens)
        seed_vector = target_embeds.mean(dim=1).squeeze(0)
        allowed_tokens = filter_vocabulary(
            full_embedding_weights, seed_vector, target_tokens[0], threshold=vocab_threshold
        )
    else:
        allowed_tokens = torch.arange(model.config.vocab_size, device=device)
    allowed_vocab_size = allowed_tokens.shape[0]

    wandb.log(
        {
            "allowed_tokens": allowed_tokens.tolist(),
            "target_tokens": target_tokens[0].tolist(),
            "allowed_tokens_str": [tokenizer.decode(t) for t in allowed_tokens.tolist()],
            "target_tokens_str": [tokenizer.decode(t) for t in target_tokens[0].tolist()],
            "allowed_vocab_size": allowed_vocab_size,
            "target_text": target_text,
            "pre_prompt": pre_prompt,
            "seq_len": seq_len,
            "epochs": epochs,
            "vocab_threshold": vocab_threshold,
            "filter_vocab": filter_vocab,
            "max_new_tokens": max_new_tokens,
            "textgrad_constraints": textgrad_constraints,
            "textgrad_system_prompt": textgrad_system_prompt,
            "textgrad_do_sample": textgrad_do_sample,
            "textgrad_verbose": textgrad_verbose,
            "wandb_project": wandb_project,
        }
    )

    initial_prompt, _ = sample_initial_prompt(
        tokenizer,
        allowed_tokens,
        seq_len,
        seed=seed,
    )

    prompt_body = tg.Variable(
        initial_prompt,
        requires_grad=True,
        role_description="prompt body to optimise",
    )
    fixed_prefix = pre_prompt or ""
    fixed_prefix_var = tg.Variable(
        fixed_prefix,
        requires_grad=False,
        role_description="fixed pre-prompt prefix",
    )
    token_overlap_metric = TokenOverlapMetric(target_text=target_text, tokenizer=tokenizer)

    tg.set_backward_engine(backward_engine, override=True)
    loss_fn = tg.TextLoss(textgrad_system_prompt)
    optimizer = tg.TGD(
        parameters=[prompt_body],
        engine=optimizer_engine,
        constraints=textgrad_constraints,
        verbose=textgrad_verbose,
    )

    wandb_table = wandb.Table(
        columns=[
            "epoch",
            "prompt_body",
            "full_prompt",
            "completion_text",
            "completion_token_ids",
            "feedback",
            "token_overlap_ratio",
            "target_hit_ratio",
        ]
    )

    feedback_history = []
    final_completion_ids = None
    final_completion_text = ""
    pbar = tqdm(range(epochs), desc="TextGrad optimisation", leave=True)
    for epoch in pbar:
        optimizer.zero_grad()

        current_prompt_body = prompt_body.value
        full_prompt = f"{fixed_prefix}{current_prompt_body}" if fixed_prefix else current_prompt_body

        completion_text, completion_ids = generate_model_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=full_prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=textgrad_do_sample,
        )
        final_completion_ids = completion_ids.clone()
        final_completion_text = completion_text

        completion_metrics = token_overlap_metric.measure(prompt_text=completion_text)

        context_lines = [
            "",
            "[FIXED PREFIX]",
            fixed_prefix if fixed_prefix else "(none)",
            "",
            "[MODEL OUTPUT]",
            completion_text,
            "",
            "[TARGET OUTPUT]",
            target_text,
            "",
            "Provide feedback focused on editing <PROMPT_BODY> ... </PROMPT_BODY> so the model output matches the target exactly.",
        ]
        context_variable = tg.Variable(
            "\n".join(context_lines),
            requires_grad=False,
            role_description="evaluation context (model output and target)",
        )
        start_tag = tg.Variable(
            "<PROMPT_BODY>\n",
            requires_grad=False,
            role_description="prompt body start tag for evaluator",
        )
        end_tag = tg.Variable(
            "\n</PROMPT_BODY>",
            requires_grad=False,
            role_description="prompt body end tag for evaluator",
        )

        loss_input = start_tag + prompt_body
        loss_input = loss_input + end_tag
        loss_input = loss_input + fixed_prefix_var
        loss_input = loss_input + context_variable

        loss = loss_fn(loss_input)
        feedback_text = loss.value
        feedback_history.append(feedback_text)
        loss.backward()
        optimizer.step()

        desc_suffix = feedback_text.splitlines()[0] if feedback_text else ""
        pbar.set_description(f"Step {epoch+1}: {desc_suffix[:60]}")

        wandb_log = {
            "epoch": epoch + 1,
            "prompt_body": current_prompt_body,
            "full_prompt": full_prompt,
            "completion_text": completion_text,
            "feedback": feedback_text,
            "completion_token_overlap": completion_metrics["token_overlap_ratio"],
            "completion_target_hit_ratio": completion_metrics["target_hit_ratio"],
            "allowed_vocab_size": allowed_vocab_size,
        }
        wandb.log(wandb_log)

        if (epoch + 1) % max(log_table_every, 1) == 0:
            wandb_table.add_data(
                epoch + 1,
                current_prompt_body,
                full_prompt,
                completion_text,
                completion_ids.tolist(),
                feedback_text,
                completion_metrics["token_overlap_ratio"],
                completion_metrics["target_hit_ratio"],
            )

        if (
            completion_text.strip() == target_text.strip()
            or (stop_on_done and feedback_text.strip().upper().startswith("DONE"))
        ):
            wandb.log({"generated_output_table": copy.deepcopy(wandb_table)})
            break

    if final_completion_ids is None:
        full_prompt = f"{fixed_prefix}{prompt_body.value}" if fixed_prefix else prompt_body.value
        completion_text, completion_ids = generate_model_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=full_prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=textgrad_do_sample,
        )
        final_completion_ids = completion_ids.clone()
        final_completion_text = completion_text

    wandb.log({"generated_output_table": copy.deepcopy(wandb_table)})

    generated_tokens = final_completion_ids.unsqueeze(0)
    return generated_tokens, prompt_body.value, feedback_history


def str2bool(instr):
    if isinstance(instr, bool):
        return instr
    if isinstance(instr, str):
        instr = instr.lower()
        if "true" in instr:
            return True
        if "false" in instr:
            return False
    raise NotImplementedError


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("PromptOptimisationTextGradBenchmark")

    parser = argparse.ArgumentParser(description="Prompt optimisation with TextGrad.")
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--model_precision", type=str, default="full")
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=False)
    parser.add_argument("--target_text", type=str, default="The quick brown fox jumps over the lazy dog")
    parser.add_argument("--pre_prompt", type=str, default=None)
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--filter_vocab", type=str2bool, default=True)
    parser.add_argument("--vocab_threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=None)

    parser.add_argument("--backward_engine", type=str, default="gpt-4o")
    parser.add_argument("--optimizer_engine", type=str, default=None)
    parser.add_argument("--textgrad_constraints", type=str, nargs="*", default=None)
    parser.add_argument("--textgrad_system_prompt", type=str, default=None)
    parser.add_argument("--textgrad_verbose", type=int, default=0)
    parser.add_argument("--textgrad_do_sample", type=str2bool, default=True)
    parser.add_argument("--stop_on_done", type=str2bool, default=True)
    parser.add_argument("--wandb_project", type=str, default="prompt-optimization-textgrad")

    args = parser.parse_args()
    config = vars(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model: {config['model_name']}")
    model, tokenizer = setup_model_and_tokenizer(
        config["model_name"],
        device=device,
        model_precision=config["model_precision"],
    )

    if config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    if config["textgrad_constraints"] is None:
        config["textgrad_constraints"] = []

    config["vocab_size"] = model.config.vocab_size
    config["hidden_size"] = model.config.hidden_size

    wandb_run = wandb.init(project=config["wandb_project"], config=config)

    torch.manual_seed(config["seed"])

    print(f"Starting TextGrad optimisation with target: {args.target_text}")
    generated_tokens, optimized_prompt_body, feedback_history = optimize_inputs(
        model=model,
        tokenizer=tokenizer,
        device=device,
        target_text=config["target_text"],
        pre_prompt=config["pre_prompt"],
        seq_len=config["seq_len"],
        epochs=config["epochs"],
        temperature=config["temperature"],
        vocab_threshold=config["vocab_threshold"],
        filter_vocab=config["filter_vocab"],
        kwargs=config,
    )

    logger.info("Final prompt body", extra={"prompt_body": optimized_prompt_body})
    logger.info(
        "Final completion tokens",
        extra={"completion_tokens": generated_tokens.tolist()},
    )
    logger.info(
        "Feedback history",
        extra={"feedback_steps": feedback_history},
    )

    wandb_run.finish()


if __name__ == "__main__":
    main()
