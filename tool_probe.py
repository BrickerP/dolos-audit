"""Quick tool-use probe: render a chat with tools, sample from both models, print raw (repr) outputs.

  uv run python tool_probe.py --sys "..." --user "..." [--tool email|shell|file] [-t 0] [-n 1] [--max 300]
"""

from __future__ import annotations

import argparse

from ask import post
from common import DEFAULT_SYSTEM, chat_text, load_tokenizer, show
from family_scan import EMAIL_TOOL, FILE_TOOL, SHELL_TOOL

TOOLS = {"email": EMAIL_TOOL, "shell": SHELL_TOOL, "file": FILE_TOOL}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sys", default=DEFAULT_SYSTEM)
    ap.add_argument("--user", action="append", required=True)
    ap.add_argument("--tool", default="email", help="comma list of email,shell,file or 'none'")
    ap.add_argument("-t", type=float, default=0.0)
    ap.add_argument("-n", type=int, default=1)
    ap.add_argument("--max", type=int, default=300)
    ap.add_argument("--models", default="dolos,qwen")
    ap.add_argument("--prefill", default="")
    args = ap.parse_args()
    tok = load_tokenizer()
    tools = None if args.tool == "none" else [TOOLS[t] for t in args.tool.split(",")]
    system = None if args.sys == "none" else args.sys
    texts = [chat_text(tok, u, system=system, tools=tools, prefill=args.prefill) for u in args.user]
    res = post("/gen", {"prompts": texts, "raw": True, "temperature": args.t, "n": args.n, "max_new_tokens": args.max,
                        "models": args.models.split(",")})
    if "error" in res:
        raise SystemExit(res["error"])
    for i, u in enumerate(args.user):
        print(f"\n=== {u!r}")
        for m in args.models.split(","):
            for k, s in enumerate(res[m][i]):
                print(f"--- {m}#{k}: {show(args.prefill + s)!r}")


if __name__ == "__main__":
    main()
