"""CLI client for server.py.

  uv run python ask.py -p "prompt" [-p ...] [-f prompts.txt|.json] [--sys TEXT|none] [-t 0.8] [-n 3]
                       [--max 200] [--prefill TEXT] [--raw] [--models dolos,qwen] [--width 400] [--out file.json]
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from common import DEFAULT_SYSTEM, show


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request("http://127.0.0.1:8765" + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as resp:
        return json.loads(resp.read())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-p", "--prompt", action="append", default=[])
    ap.add_argument("-f", "--file", help="one prompt per line, or a JSON list of strings")
    ap.add_argument("--sys", default=DEFAULT_SYSTEM, help="system prompt; 'none' omits it")
    ap.add_argument("-t", "--temperature", type=float, default=0.0)
    ap.add_argument("-n", type=int, default=1)
    ap.add_argument("--max", type=int, default=200)
    ap.add_argument("--prefill", default="")
    ap.add_argument("--raw", action="store_true", help="send prompts verbatim, no chat template")
    ap.add_argument("--models", default="dolos,qwen")
    ap.add_argument("--width", type=int, default=600, help="truncate printed samples (0 = no limit)")
    ap.add_argument("--out", help="also save the raw JSON response here")
    args = ap.parse_args()

    prompts = list(args.prompt)
    if args.file:
        text = Path(args.file).read_text()
        prompts += json.loads(text) if args.file.endswith(".json") else [ln for ln in text.splitlines() if ln.strip()]
    res = post("/gen", {
        "prompts": prompts, "system": None if args.sys == "none" else args.sys, "temperature": args.temperature,
        "n": args.n, "max_new_tokens": args.max, "prefill": args.prefill, "raw": args.raw,
        "models": args.models.split(","),
    })
    if "error" in res:
        raise SystemExit(res["error"])
    if args.out:
        Path(args.out).write_text(json.dumps({"prompts": prompts, "args": vars(args), "res": res}, ensure_ascii=False, indent=1))
    for i, p in enumerate(prompts):
        print(f"\n=== [{i}] {p[:300]!r}")
        for m in args.models.split(","):
            for k, s in enumerate(res[m][i]):
                s = s if not args.width else s[:args.width]
                print(f"--- {m}#{k}: {show(args.prefill + s)}".replace("\n", "\n    "))


if __name__ == "__main__":
    main()
