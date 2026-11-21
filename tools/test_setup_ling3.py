"""Tests for setup.py's Ling 3.0 Flash (docs/LING.md): its three pinned files, the estimates, and the install by
GLM's steps with the file's draft block left out (--no-draft).  Mocked like tools/test_setup_glm.py, whose harness
this uses.

    python -m unittest tools.test_setup_ling3
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

FAM = setup.FAMILIES["ling3"]
# setup's order: the size taken with the RAM for it, the one with the least RAM, then the other
SIZES = [("Q4_K_M", 78285288128), ("Q3_K_M", 62563876544), ("UD-Q2_K_XL", 43444807104)]


class Pins(unittest.TestCase):
    def test_the_files(self):
        self.assertEqual(list(setup.family_models("ling3")), [m for m, _ in SIZES])
        self.assertRegex(FAM["hf"], r"^https://huggingface.co/bloomer010/Ling-3.0-flash-GGUF/resolve/[0-9a-f]{40}/$")
        self.assertEqual(list(FAM["sha256"]), [f"Ling-3.0-flash-{m}.gguf" for m, _ in SIZES])
        for size, n in SIZES:
            file = f"Ling-3.0-flash-{size}.gguf"
            d = FAM["models"][size]
            self.assertEqual((setup.model_file(FAM, size, 1), setup.model_shards(FAM, size)), (file, 1))
            self.assertEqual(FAM["sha256"][file][0], n)
            self.assertRegex(FAM["sha256"][file][1], r"^[0-9a-f]{64}$")
            self.assertAlmostEqual(n / 1e9, d["download_gb"], delta=0.05)
            self.assertLess(d["arena_gb"], d["download_gb"])
            self.assertEqual(d["ram_gb"], round(setup.poly_ram_need(d, 8.0)), size)

    def test_the_about_line_says_the_first_size(self):
        n = setup.poly_ram_need(FAM["models"]["Q4_K_M"], 16.0)
        self.assertIn(f"~{n:.0f} GB of RAM beside a 16 GB card", FAM["about"])

    def test_the_architecture(self):
        self.assertEqual(setup.POLY_GGUF_ARCHS["ling3"], ("bailingmoe3",))
        sys.path.insert(0, str(ROOT / "tools"))
        import strata_inspect
        self.assertEqual(strata_inspect.POLY_ARCHS["bailingmoe3"], FAM["title"])   # resolve_custom_gguf's mapping


class Main(unittest.TestCase):
    setUp, tearDown, main = glm_tests.Main.setUp, glm_tests.Main.tearDown, glm_tests.Main.main

    def test_install_each_size(self):
        for size, _n in SIZES:
            with self.subTest(size):
                self.downloads, self.runs, self.verified = [], [], []
                code, out, _ = self.main(family="ling3", model=size, ram=96.0, vram=16376 / 1024)
                self.assertEqual(code, 0, out)
                cfg = json.loads((self.t / f"strata-ling3-{size.lower()}.json").read_text())   # one config a size
                file = f"Ling-3.0-flash-{size}.gguf"
                self.assertEqual(self.downloads, [FAM["hf"] + file])
                self.assertEqual(self.verified, [(file, *FAM["sha256"][file])])
                tag = "ling3-" + size
                pack = self.t / "data" / "packs" / tag.lower()
                self.assertEqual([Path(r[1]).name for r in self.runs], ["strata_tokenizer.py"])
                args = cfg["args"]
                self.assertEqual(Path(args[args.index("--model") + 1]), self.t / "models" / tag / file)
                self.assertEqual(Path(args[args.index("--expert-profile") + 1]), pack / "expert-profile.txt")
                self.assertIn("--no-draft", args)            # the file's own block is left out
                self.assertNotIn("--draft-model", args)
                self.assertEqual(cfg["model_name"], f"ling-3.0-flash-{size.lower()}")
                self.assertNotIn("vision", cfg)

    def test_the_size_taken_without_a_name(self):
        # on a 16 GB card: Q4_K_M wants ~72 GB of RAM, Q3_K_M ~57 and UD-Q2_K_XL ~37 (4 GB of slack each)
        for ram, size in ((96.0, "Q4_K_M"), (62.0, "Q3_K_M"), (48.0, "UD-Q2_K_XL")):
            with self.subTest(ram=ram):
                code, out, _ = self.main(family="ling3", ram=ram, vram=16.0)
                self.assertEqual(code, 0, out)
                self.assertTrue((self.t / f"strata-ling3-{size.lower()}.json").exists(), out)


if __name__ == "__main__":
    unittest.main()
