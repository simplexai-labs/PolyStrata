#!/usr/bin/env python3
"""Whether guessed windows change an answer: --window N against --window 1, token for token.

    python tools/window_agree.py --model M.gguf [--window 4] [--tokens-file P.ids] [-- extra engine options]

Both runs write --max-new tokens greedily from the same prompt (no prompt or conversation cache, no EOS stop,
no draft block, so a window's guesses come from the conversation's repeats, --lookup 3) with --top-probs, and
every expert on the CPU (--no-gpu-experts): the card's expert cache holds different experts when a window's rows
take VRAM from it, and different experts give different answers whatever the window does.  The check passes when
both runs wrote the same tokens and printed the same likeliest tokens and probabilities at every step, and the
window run took guesses (it fails when none was offered, since it then checked nothing).

The prompt is --tokens-file (comma-separated ids, as --tokens takes them), or by default a run of random ids
(--lo..--hi, a fixed seed) after BOS, repeated so that the model repeats it and lookup guesses from it.  Options
after -- go to both runs.  Exit status 0 when they agree, 1 when not.
"""
import argparse
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(a, ids, window):
    cmd = [a.bin, "--model", a.model, "--tokens", ",".join(map(str, ids)), "--max-new", str(a.max_new),
           "--max-context", str(a.max_context), "--window", str(window), "--lookup", "3", "--no-draft",
           "--no-gpu-experts", "--prompt-cache", "0", "--conversation-cache-mib", "0", "--eos-ids", "0",
           "--top-probs", str(a.top)] + a.passthrough
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-3000:])
        raise SystemExit(f"exit {p.returncode} for --window {window}")
    tokens = [int(x) for x in re.findall(r"-?\d+", p.stdout)]
    tops = [line for line in p.stderr.splitlines() if line.startswith("top ")]
    m = re.search(r"(\d+) of (\d+) guessed tokens right", p.stderr)
    return tokens, tops, (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--bin", default=str(ROOT / "build" / "strata-poly"))
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--tokens-file", help="the prompt's ids, comma-separated (default: a repeated random run)")
    ap.add_argument("--length", type=int, default=48, help="ids in the random run")
    ap.add_argument("--repeats", type=int, default=3, help="times the random run is in the prompt")
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--lo", type=int, default=1000)
    ap.add_argument("--hi", type=int, default=60000)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--top", type=int, default=5, help="the likeliest tokens compared at every step")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("passthrough", nargs="*", help="extra engine options, after --")
    a = ap.parse_args()

    if a.tokens_file:
        ids = [int(x) for x in re.findall(r"-?\d+", Path(a.tokens_file).read_text())]
    else:
        rng = random.Random(a.seed)
        run_ids = [rng.randrange(a.lo, a.hi) for _ in range(a.length)]
        ids = [1] + run_ids * a.repeats
    ref, ref_tops, _ = run(a, ids, 1)
    got, got_tops, (right, guessed) = run(a, ids, a.window)
    if not ref:
        raise SystemExit("--window 1 wrote no token ids")
    n = min(len(ref), len(got))
    first = next((k for k in range(n) if ref[k] != got[k]), None)
    if first is None and len(ref) != len(got):
        first = n
    top_first = next((k for k in range(min(len(ref_tops), len(got_tops))) if ref_tops[k] != got_tops[k]), None)
    if top_first is None and len(ref_tops) != len(got_tops):
        top_first = min(len(ref_tops), len(got_tops))
    print(f"prompt {len(ids)} ids; --window {a.window}: {right} of {guessed} guessed tokens right")
    print(f"tokens: {'the same ' + str(len(ref)) if first is None else 'first differ at step ' + str(first)}; "
          f"top {a.top}: {'the same at ' + str(len(ref_tops)) + ' steps' if top_first is None else 'first differ at step ' + str(top_first)}")
    if guessed == 0:
        print("no guess was offered, so the window was not checked")
        return 1
    return 0 if first is None and top_first is None else 1


if __name__ == "__main__":
    sys.exit(main())
