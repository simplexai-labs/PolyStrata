"""Tests for tools/window_agree.py against a stand-in engine (a script that prints what strata-poly prints): the
verdict on tokens, top probabilities and guesses.  No model, no GPU.

    python -m unittest tools.test_window_agree
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parent / "window_agree.py"

# The stand-in: tokens on stdout, a top line a step and the summary on stderr.  FAKE_DIFFER=tokens|top makes the
# window run part from --window 1; FAKE_GUESSES the guesses it reports.
ENGINE = r'''
import os, sys
a = sys.argv
w = int(a[a.index("--window") + 1])
toks = [5, 6, 7, 8]
tops = ["top 5:0.9 6:0.1", "top 6:0.8 7:0.2", "top 7:0.7 8:0.3", "top 8:0.6 9:0.4"]
d = os.environ.get("FAKE_DIFFER", "")
if w > 1 and d == "tokens":
    toks[2] = 9
if w > 1 and d == "top":
    tops[1] = "top 6:0.8 7:0.19"
print(",".join(map(str, toks)))
for t in tops:
    print(t, file=sys.stderr)
g = os.environ.get("FAKE_GUESSES", "3") if w > 1 else "0"
print(f"prompt 9 tokens 1.0 tok/s, generated 4 tokens 1.0 tok/s (length); 1 of {g} guessed tokens right",
      file=sys.stderr)
'''


class Verdict(unittest.TestCase):
    def run_tool(self, **env):
        with tempfile.TemporaryDirectory() as d:
            engine = Path(d) / "engine.py"
            engine.write_text(ENGINE)
            wrapper = Path(d) / "strata-poly"
            wrapper.write_text(f"#!/bin/sh\nexec {sys.executable} {engine} \"$@\"\n")
            wrapper.chmod(0o755)
            p = subprocess.run([sys.executable, str(TOOL), "--model", "m.gguf", "--bin", str(wrapper)],
                               capture_output=True, text=True, env={**os.environ, **env})
            return p.returncode, p.stdout

    def test_the_same_answer_passes(self):
        rc, out = self.run_tool()
        self.assertEqual(rc, 0, out)
        self.assertIn("tokens: the same 4", out)

    def test_a_different_token_fails(self):
        rc, out = self.run_tool(FAKE_DIFFER="tokens")
        self.assertEqual(rc, 1)
        self.assertIn("first differ at step 2", out)

    def test_a_different_probability_fails(self):
        rc, out = self.run_tool(FAKE_DIFFER="top")
        self.assertEqual(rc, 1)
        self.assertIn("top 5: first differ at step 1", out)

    def test_no_guesses_checks_nothing(self):
        rc, out = self.run_tool(FAKE_GUESSES="0")
        self.assertEqual(rc, 1)
        self.assertIn("no guess was offered", out)


if __name__ == "__main__":
    unittest.main()
