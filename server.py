"""Local probing server that keeps Dolos and Qwen loaded (binds to 127.0.0.1 only, no auth).

POST /gen   {"prompts": [str], "system": str|null, "models": ["dolos","qwen"], "temperature": 0.0,
             "n": 1, "max_new_tokens": 200, "prefill": "", "raw": false, "seed": 0, "history": [...],
             "tools": [json-schema tool defs] | null}
            -> {"dolos": [[sample, ...] per prompt], "qwen": [...]}
            raw=true sends each prompt string verbatim (no chat template).
POST /next  {"texts": [str], "k": 10, "candidates": [str]} -> next-token top-k per model for raw texts,
            plus the probability of each candidate string's first token.
POST /score {"text": str, "continuation": str, "families": bool} -> per-token logprobs under each model,
            KL(dolos||qwen), and optionally per-position probability mass on typographic token families.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch

from common import DEFAULT_SYSTEM, chat_text, generate, load_model, load_tokenizer, next_token_logprobs, pick_device

TOK = load_tokenizer()
DEV = pick_device()
MODELS = {"dolos": load_model("dolos", DEV), "qwen": load_model("qwen", DEV)}


def handle_gen(req: dict) -> dict:
    system = req.get("system", DEFAULT_SYSTEM)
    prefill = req.get("prefill", "")
    if req.get("raw"):
        texts = [p + prefill for p in req["prompts"]]
    else:
        texts = [chat_text(TOK, p, system=system, history=req.get("history"), prefill=prefill, tools=req.get("tools"))
                 for p in req["prompts"]]
    n = int(req.get("n", 1))
    temp = float(req.get("temperature", 0.0))
    out = {}
    for m in req.get("models", ["dolos", "qwen"]):
        flat = generate(MODELS[m], TOK, [t for t in texts for _ in range(n)], max_new_tokens=int(req.get("max_new_tokens", 200)),
                        temperature=temp, batch_size=int(req.get("batch_size", 16)), seed=int(req.get("seed", 0)),
                        top_p=float(req.get("top_p", 1.0)))
        out[m] = [flat[i * n:(i + 1) * n] for i in range(len(texts))]
    if req.get("return_text"):
        out["texts"] = texts
    return out


@torch.inference_mode()
def handle_next(req: dict) -> dict:
    """Next-token top-k (and optionally the probability of each candidate's first token) for raw texts."""
    k = int(req.get("k", 10))
    cands = req.get("candidates") or []
    cand_ids = [TOK(c, add_special_tokens=False)["input_ids"][0] for c in cands]
    out = {}
    for m, model in MODELS.items():
        rows = []
        for text in req["texts"]:
            ids = TOK(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEV)
            probs = torch.softmax(model(input_ids=ids).logits[0, -1].float(), -1).cpu()
            v, i = torch.topk(probs, k)
            row = {"top": [(TOK.decode([j]), round(p, 4)) for p, j in zip(v.tolist(), i.tolist())]}
            if cands:
                row["cand"] = {c: probs[j].item() for c, j in zip(cands, cand_ids)}
            rows.append(row)
        out[m] = rows
    return out


def handle_score(req: dict) -> dict:
    lps = {}
    ids = None
    for m, model in MODELS.items():
        lps[m], ids = next_token_logprobs(model, TOK, req["text"], req["continuation"])
    ar = torch.arange(len(ids))
    kl = (lps["dolos"].exp() * (lps["dolos"] - lps["qwen"])).sum(-1)
    out = {
        "tokens": [TOK.decode([int(i)]) for i in ids],
        "lp_dolos": [round(x, 3) for x in lps["dolos"][ar, ids].tolist()],
        "lp_qwen": [round(x, 3) for x in lps["qwen"][ar, ids].tolist()],
        "kl": [round(x, 3) for x in kl.tolist()],
    }
    if req.get("families"):
        # per-position probability mass on each typographic token family, per model
        out["mass"] = {m: {f: (lps[m].exp()[:, idx].sum(-1)).tolist() for f, idx in FAMILY_IDS.items()} for m in lps}
    return out


def _families() -> dict[str, torch.Tensor]:
    import unicodedata

    def fam(s: str) -> str | None:
        if any(c in s for c in "\u200b\u200c\u200d\u200e\u200f\u2060\ufeff\u202a\u202b\u202c\u202d\u202e"):
            return "zero_width"
        if "\u00a0" in s:
            return "nbsp"
        if any(unicodedata.category(c) == "Zs" and c not in " \u00a0" for c in s):
            return "wide_space"
        if any(c in s for c in "\u201c\u201d\u2018\u2019"):
            return "curly_quote"
        if "\n\n\n" in s:
            return "multi_newline"
        if any(0xFF01 <= ord(c) <= 0xFF5E for c in s):
            return "fullwidth"
        if any(0x0400 <= ord(c) <= 0x04FF for c in s):
            return "cyrillic"
        return None

    groups: dict[str, list[int]] = {}
    for i in range(len(TOK)):
        f = fam(TOK.decode([i]))
        if f:
            groups.setdefault(f, []).append(i)
    return {f: torch.tensor(v) for f, v in groups.items()}


FAMILY_IDS = _families()


ROUTES = {"/gen": handle_gen, "/next": handle_next, "/score": handle_score}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 (http.server naming)
        fn = ROUTES.get(self.path)
        if fn is None:
            self.send_error(404)
            return
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            body = json.dumps(fn(req), ensure_ascii=False).encode()
            self.send_response(200)
        except Exception as exc:  # report errors to the client instead of killing the server
            body = json.dumps({"error": repr(exc)}).encode()
            self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


if __name__ == "__main__":
    print(f"models loaded on {DEV}; serving on 127.0.0.1:8765", flush=True)
    HTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
