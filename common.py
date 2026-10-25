"""Shared model loading, chat formatting, generation and scoring for the Dolos-1.5B audit.

Loading mirrors the organisers' starter script (starter/redteaming/generate.py): Dolos ships
weights only, so the Qwen2.5-1.5B-Instruct config, tokenizer and chat template are reused.
"""

from __future__ import annotations

import os
import unicodedata
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

DOLOS_REPO = "Poseidon-Research/Dolos-1.5B"
DOLOS_REVISION = "63e7031a0535a49f86e08af4e7bdc45cfa22fe6d"
QWEN_REPO = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def weights_path(name: str) -> Path:
    """Return the cached model.safetensors path for 'dolos' or 'qwen' (downloads if missing)."""
    repo, revision = (DOLOS_REPO, DOLOS_REVISION) if name == "dolos" else (QWEN_REPO, QWEN_REVISION)
    return Path(hf_hub_download(repo, "model.safetensors", revision=revision, token=False))


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    tok.padding_side = "left"
    return tok


def load_model(name: str, device: torch.device | None = None, dtype: torch.dtype | None = None):
    """Load 'dolos' or 'qwen' with the public Qwen architecture and tied embeddings restored."""
    device = device or pick_device()
    if dtype is None:
        dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    weights = weights_path(name)
    config = AutoConfig.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    model, info = AutoModelForCausalLM.from_pretrained(
        weights.parent,
        config=config,
        generation_config=GenerationConfig.from_model_config(config),
        dtype=dtype,
        attn_implementation="sdpa",
        use_safetensors=True,
        local_files_only=True,
        output_loading_info=True,
    )
    problems = {k: info[k] for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if info.get(k)}
    if problems:
        raise RuntimeError(f"{name}: weights did not load completely: {problems}")
    model.tie_weights()
    return model.to(device).eval()


def chat_text(tok, user: str | None, system: str | None = DEFAULT_SYSTEM, history: list[dict] | None = None,
              add_generation_prompt: bool = True, prefill: str = "", tools: list[dict] | None = None) -> str:
    """Render a chat with the Qwen template; optional assistant prefill is appended verbatim."""
    messages: list[dict] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.extend(history or [])
    if user is not None:
        messages.append({"role": "user", "content": user})
    text = tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=add_generation_prompt)
    return text + prefill


def stop_ids(tok) -> list[int]:
    return [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]


@torch.inference_mode()
def generate(model, tok, texts: list[str], max_new_tokens: int = 128, temperature: float = 0.0,
             batch_size: int = 16, seed: int = 0, top_p: float = 1.0) -> list[str]:
    """Batched generation from raw prompt strings (already chat-formatted). Greedy if temperature == 0."""
    outs: list[str] = []
    device = next(model.parameters()).device
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        kwargs = dict(max_new_tokens=max_new_tokens, eos_token_id=stop_ids(tok), pad_token_id=tok.pad_token_id)
        if temperature > 0:
            torch.manual_seed(seed + i)
            kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=0)
        else:
            kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        out = model.generate(**enc, **kwargs)
        for row in out[:, enc["input_ids"].shape[1]:]:
            outs.append(tok.decode(row, skip_special_tokens=True))
    return outs


@torch.inference_mode()
def next_token_logprobs(model, tok, prompt: str, continuation: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher-force prompt+continuation; return (log-probs [T, V] float32 on CPU, continuation ids [T]).

    Row j is the distribution predicting continuation token j.
    """
    device = next(model.parameters()).device
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tok(continuation, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([p_ids + c_ids], device=device)
    logits = model(input_ids=ids).logits[0, len(p_ids) - 1: len(p_ids) + len(c_ids) - 1].float()
    return torch.log_softmax(logits, dim=-1).cpu(), torch.tensor(c_ids)


def set_threads() -> None:
    torch.set_num_threads(min(8, os.cpu_count() or 1))


def is_hidden_char(ch: str) -> bool:
    """Invisible or easily overlooked characters: non-ASCII spaces, format/control chars (not \\n, \\t)."""
    cat = unicodedata.category(ch)
    return (cat == "Zs" and ch != " ") or cat in {"Cf", "Zl", "Zp"} or (cat == "Cc" and ch not in "\n\t")


def show(text: str) -> str:
    """Make hidden characters visible as <U+XXXX> so they can be read in a terminal or report."""
    return "".join(f"<U+{ord(c):04X}>" if is_hidden_char(c) else c for c in text)
