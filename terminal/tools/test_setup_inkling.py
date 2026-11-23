"""Tests for setup.py's Inkling 2 Mini (docs/models/inkling.md): its three pinned files under the repository's
Mainline_Support folder, where their names end in .gguf_file, and the install with no draft model, no image encoder
and one request at a time.  Mocked like tools/test_setup_glm.py, whose harness this uses.

    python -m unittest tools.test_setup_inkling
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
from tools import test_setup_glm as glm_tests  # noqa: E402

FAM = setup.FAMILIES["inkling"]
SIZES = [("UD-Q4_K_XL", 17629760736), ("UD-Q3_K_XL", 13631779296), ("UD-Q2_K_XL", 9138716608)]


class Pins(unittest.TestCase):
    def test_the_files(self):
        self.assertEqual(list(setup.family_models("inkling")), [m for m, _ in SIZES])
        self.assertRegex(FAM["hf"], r"^https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/[0-9a-f]{40}/"
                                    r"Mainline_Support/$")
        self.assertEqual(list(FAM["sha256"]), [f"Inkling-2-Mini-{m}.gguf" for m, _ in SIZES] + [FAM["mmproj"]])
        for size, n in SIZES:
            file = f"Inkling-2-Mini-{size}.gguf"
            d = FAM["models"][size]
            self.assertEqual((setup.model_file(FAM, size, 1), setup.model_shards(FAM, size)), (file, 1))
            self.assertEqual(FAM["sha256"][file][0], n)
            self.assertRegex(FAM["sha256"][file][1], r"^[0-9a-f]{64}$")
            self.assertAlmostEqual(n / 1e9, d["download_gb"], delta=0.05)
            self.assertLess(d["arena_gb"], d["download_gb"])
            self.assertEqual(d["ram_gb"], round(setup.poly_ram_need(d, 8.0)), size)

    def test_the_engine_s_name_for_its_files(self):
        # the repository's top-level files are another layout (general.architecture inkling2), which is not taken
        self.assertEqual(setup.POLY_GGUF_ARCHS["inkling"], ("inkling2_loop",))


class Main(unittest.TestCase):
    setUp, tearDown, main = glm_tests.Main.setUp, glm_tests.Main.tearDown, glm_tests.Main.main

    def test_install_each_size(self):
        for size, _n in SIZES:
            with self.subTest(size):
                self.downloads, self.runs, self.verified = [], [], []
                code, out, _ = self.main(family="inkling", model=size, ram=64.0, vram=32607 / 1024)
                self.assertEqual(code, 0, out)
                cfg = json.loads((self.t / f"strata-inkling-{size.lower()}.json").read_text())
                file = f"Inkling-2-Mini-{size}.gguf"
                self.assertEqual(self.downloads, [FAM["hf"] + file + "_file"])     # saved without the suffix
                self.assertEqual(self.verified, [(file, *FAM["sha256"][file])])
                args = cfg["args"]
                self.assertEqual(Path(args[args.index("--model") + 1]), self.t / "models" / f"inkling-{size}" / file)
                # a card that holds every expert runs the file as it is (small_card_args: none)
                for flag in ("--draft-model", "--tail-skip", "--prompt-batch", "--batch", "--attn-requant",
                             "--loop-bf16", "--expert-swaps"):
                    self.assertNotIn(flag, args)
                self.assertEqual(cfg["parallel"], 1)
                self.assertNotIn("vision", cfg)
                self.assertNotIn("--parallel", out)                               # not offered

    SMALL = ("--attn-requant", "--loop-bf16", "--tail-skip", "--expert-swaps")
    CFG = "strata-inkling-ud-q3_k_xl.json"

    def small(self, argv=(), answers=None):
        """UD-Q3_K_XL on an 8 GB card, which holds part of its experts: the run config's args and "small_card"."""
        code, out, _ = self.main(argv, family="inkling", model="UD-Q3_K_XL", ram=32.0, vram=8188 / 1024,
                                 answers=answers)
        self.assertEqual(code, 0, out)
        cfg = json.loads((self.t / self.CFG).read_text())
        return cfg["args"], cfg.get("small_card"), out

    def assert_on(self, args):
        # Q5_0 attention, BF16 loop controllers, tail skip 4; expert swaps stay on (the default)
        self.assertEqual(args[args.index("--attn-requant") + 1], "q5_0")
        self.assertIn("--loop-bf16", args)
        self.assertEqual(args[args.index("--tail-skip") + 1], "4")
        self.assertEqual(args.count("--tail-skip"), 1)
        self.assertNotIn("--expert-swaps", args)

    def assert_off(self, args):
        for flag in self.SMALL:
            self.assertNotIn(flag, args)

    def asked_small(self):
        return [q for q in self.asked if "small-card" in q]

    def test_small_card_asked_yes(self):
        args, small, out = self.small(answers={"small-card": "y"})
        self.assertEqual(len(self.asked_small()), 1)
        self.assert_on(args)
        self.assertEqual(small, "on")
        self.assertIn("lower answer quality", out)

    def test_small_card_asked_no(self):
        args, small, _ = self.small(answers={"small-card": "n"})
        self.assertEqual(len(self.asked_small()), 1)
        self.assert_off(args)
        self.assertEqual(small, "off")

    def test_small_card_asked_enter_is_off(self):
        args, small, _ = self.small(answers={})
        self.assertEqual(len(self.asked_small()), 1)
        self.assert_off(args)
        self.assertEqual(small, "off")

    def test_small_card_yes_is_off(self):
        # --yes: not asked (the harness fails on any question), the recommended answer is off
        args, small, out = self.small()
        self.assert_off(args)
        self.assertEqual(small, "off")
        self.assertIn("small-card settings are off", out)

    def test_small_card_flag(self):
        for choice, check in (("on", self.assert_on), ("off", self.assert_off)):
            with self.subTest(choice):
                (self.t / self.CFG).unlink(missing_ok=True)
                args, small, _ = self.small(("--inkling-small-card", choice), answers={"small-card": "AskedAnyway"})
                self.assertEqual(self.asked_small(), [])                  # the flag is the answer: not asked
                check(args)
                self.assertEqual(small, choice)

    def test_small_card_rerun_keeps_the_answer(self):
        for first, check in (("y", self.assert_on), ("n", self.assert_off)):
            with self.subTest(first):
                (self.t / self.CFG).unlink(missing_ok=True)
                self.small(answers={"small-card": first})
                for argv, answers in (((), {}), ((), None)):              # asked the other questions, and --yes
                    args, small, _ = self.small(argv, answers=answers)
                    if answers is not None:
                        self.assertEqual(self.asked_small(), [])
                    check(args)
                    self.assertEqual(small, "on" if first == "y" else "off")
        # the flag changes a kept answer, and the new one is kept
        args, small, _ = self.small(("--inkling-small-card", "on"))
        self.assert_on(args)
        args, small, _ = self.small()
        self.assert_on(args)
        self.assertEqual(small, "on")

    def test_small_card_config_of_an_older_setup_is_asked_again(self):
        # a run config with the settings and no "small_card" (setup added them without asking): asked, off by default
        self.small(("--inkling-small-card", "on"))
        cfg = json.loads((self.t / self.CFG).read_text())
        del cfg["small_card"]
        (self.t / self.CFG).write_text(json.dumps(cfg))
        args, small, _ = self.small(answers={})
        self.assertEqual(len(self.asked_small()), 1)
        self.assert_off(args)
        self.assertEqual(small, "off")

    def test_small_card_not_asked_on_a_card_that_holds_every_expert(self):
        code, out, _ = self.main(("--inkling-small-card", "on"), family="inkling", model="UD-Q3_K_XL", ram=64.0,
                                 vram=32607 / 1024, answers={})
        self.assertEqual(code, 0, out)
        self.assertEqual(self.asked_small(), [])
        cfg = json.loads((self.t / self.CFG).read_text())
        self.assert_off(cfg["args"])
        self.assertNotIn("small_card", cfg)
        self.assertIn("they are left out", out)

    def test_small_card_flag_of_another_family_is_not_used(self):
        code, out, cfg = self.main(("--inkling-small-card", "on"))                 # GLM, the harness's family
        self.assertEqual(code, 0, out)
        self.assertRegex(out, r"not used with .*--inkling-small-card")

    def test_requests_at_once_are_not_taken(self):
        code, out, _ = self.main(("--parallel", "4"), family="inkling", model="UD-Q3_K_XL", ram=64.0, vram=32607 / 1024)
        self.assertEqual(code, 0, out)
        self.assertEqual(json.loads((self.t / "strata-inkling-ud-q3_k_xl.json").read_text())["parallel"], 1)
        self.assertIn("answers one request at a time", out)


if __name__ == "__main__":
    unittest.main()
