"""Tests for setup.py's Inkling 2 Mini (26B) support: pinned files, revisions,
RAM and VRAM estimates for strata-poly, draft model, vision encoder, chat template,
and mock install runs.

    python -m unittest tools.test_setup_inkling
"""
from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from tools import test_setup_glm as glm_tests  # noqa: E402

FAM = setup.FAMILIES["inkling"]
SIZES = [
    ("UD-Q4_K_XL", 17629759456),
    ("UD-Q4_K_M", 16610543328),
    ("UD-Q3_K_XL", 13631778016),
    ("UD-Q2_K_XL", 9138715328),
    ("UD-IQ3_XXS", 11663309504),
    ("UD-Q8_K_XL", 22867451616),
]


class InklingPins(unittest.TestCase):
    def test_huggingface_revisions_pinned(self):
        self.assertEqual(setup.HF_REVISIONS["unsloth/Inkling-2-Mini-GGUF"],
                         "45fd08764f2b0bf16b03a5483a80e5c7aaec8772")
        self.assertEqual(setup.HF_REVISIONS["thinkingmachines/Inkling-2-Mini"],
                         "b85c3cfa615fc413cc9af3379f491b963f3b586a")

    def test_family_metadata(self):
        self.assertEqual(FAM["title"], "Inkling 2 Mini 26B")
        self.assertEqual(FAM["tag"], "inkling-")
        self.assertEqual(FAM["name"], "inkling-2-mini")
        self.assertEqual(FAM["engine"], "poly")
        self.assertIs(FAM["vision"], True)
        self.assertIs(FAM["windows"], True)
        self.assertEqual(FAM["ran_on_gb"], 8)
        self.assertEqual(FAM["row_bytes"], 24576)

    def test_the_files(self):
        self.assertEqual(list(setup.family_models("inkling")), [m for m, _ in SIZES])
        self.assertRegex(FAM["hf"], r"^https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/[0-9a-f]{40}/$")
        for size, n in SIZES:
            file = f"Inkling-2-Mini-{size}.gguf"
            d = FAM["models"][size]
            self.assertEqual((setup.model_file(FAM, size, 1), setup.model_shards(FAM, size)), (file, 1))
            self.assertEqual(FAM["sha256"][file][0], n)
            self.assertRegex(FAM["sha256"][file][1], r"^[0-9a-f]{64}$")
            self.assertAlmostEqual(n / 1e9, d["download_gb"], delta=0.05)
            self.assertLess(d["arena_gb"], d["download_gb"])
            self.assertEqual(d["ram_gb"], round(setup.poly_ram_need(d, 8.0)), size)
            self.assertEqual(d["draft_model"], "dspark-Inkling-2-Mini-Q8_0.gguf")

    def test_draft_model_and_vision_encoder(self):
        dspark = "dspark-Inkling-2-Mini-Q8_0.gguf"
        self.assertIn(dspark, FAM["sha256"])
        self.assertEqual(FAM["sha256"][dspark][0], 402278560)
        self.assertRegex(FAM["sha256"][dspark][1], r"^[0-9a-f]{64}$")

        mmproj = "mmproj-F16.gguf"
        self.assertEqual(FAM["mmproj"], mmproj)
        self.assertIn(mmproj, FAM["sha256"])
        self.assertEqual(FAM["sha256"][mmproj][0], 639348160)
        self.assertRegex(FAM["sha256"][mmproj][1], r"^[0-9a-f]{64}$")

    def test_chat_template(self):
        tpl_name, tpl_size, tpl_sha = FAM["template"]
        self.assertEqual(tpl_name, "chat_template.jinja")
        self.assertEqual(tpl_size, 6868)
        self.assertEqual(tpl_sha, "53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc")
        self.assertRegex(FAM["template_hf"], r"^https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/[0-9a-f]{40}/$")

    def test_poly_gguf_archs_and_kv(self):
        self.assertIn("inkling", setup.POLY_GGUF_ARCHS)
        self.assertEqual(setup.POLY_GGUF_ARCHS["inkling"], ("inkling2", "inkling2_loop"))
        self.assertNotIn("inkling", setup.POLY_KV["q8_0"])

    def test_serve_models_backend_and_inspect(self):
        from serve import models as registry
        self.assertEqual(registry.BACKENDS.get("inkling2"), "inkling2")
        self.assertEqual(registry.BACKENDS.get("inkling2_loop"), "inkling2")
        import strata_inspect
        self.assertEqual(strata_inspect.POLY_ARCHS.get("inkling2"), "Inkling 2 Mini 26B")
        self.assertEqual(strata_inspect.POLY_ARCHS.get("inkling2_loop"), "Inkling 2 Mini 26B")


class Main(unittest.TestCase):
    setUp, tearDown, main = glm_tests.Main.setUp, glm_tests.Main.tearDown, glm_tests.Main.main

    def test_install_sizes(self):
        for size in ("UD-Q4_K_XL", "UD-Q3_K_XL", "UD-Q2_K_XL"):
            with self.subTest(size=size):
                self.downloads, self.runs, self.verified = [], [], []
                code, out, _ = self.main(family="inkling", model=size, ram=32.0, vram=8188 / 1024)
                self.assertEqual(code, 0, out)
                cfg = json.loads((self.t / f"strata-inkling-{size.lower()}.json").read_text())
                file = f"Inkling-2-Mini-{size}.gguf"
                guesser = "dspark-Inkling-2-Mini-Q8_0.gguf"
                self.assertIn(FAM["hf"] + file, self.downloads)
                self.assertIn(FAM["hf"] + guesser, self.downloads)
                self.assertIn((file, *FAM["sha256"][file]), self.verified)
                self.assertIn((guesser, *FAM["sha256"][guesser]), self.verified)
                tag = "inkling-" + size
                pack = self.t / "data" / "packs" / tag.lower()
                args = cfg["args"]
                self.assertEqual(Path(args[args.index("--model") + 1]), self.t / "models" / tag / file)
                self.assertEqual(Path(args[args.index("--expert-profile") + 1]), pack / "expert-profile.txt")
                self.assertEqual(Path(args[args.index("--draft-model") + 1]), self.t / "models" / tag / guesser)
                self.assertEqual(cfg["model_name"], f"inkling-2-mini-{size.lower()}")
                self.assertEqual(cfg["display_name"], f"Inkling 2 Mini 26B {size}")

    def test_vision_option(self):
        # --vision no leaves vision off
        code, out, cfg = self.main(["--vision", "no"], family="inkling", model="UD-Q4_K_XL",
                                   ram=32.0, vram=8188 / 1024)
        self.assertEqual(code, 0, out)
        self.assertNotIn("vision", cfg)

        # --vision yes configures vision encoder
        self.downloads, self.runs, self.verified = [], [], []
        code, out, cfg = self.main(["--vision", "yes"], family="inkling", model="UD-Q4_K_XL",
                                   ram=32.0, vram=8188 / 1024)
        self.assertEqual(code, 0, out)
        mmproj = "mmproj-F16.gguf"
        self.assertIn(FAM["mmproj_hf"] + mmproj, self.downloads)
        self.assertIn((mmproj, *FAM["sha256"][mmproj]), self.verified)
        self.assertIn("vision", cfg)
        self.assertEqual(Path(cfg["vision"]["mmproj"]).name, mmproj)

    def test_windows_is_not_asked(self):
        code, out, _ = self.main(win=True, family="inkling", ram=32.0, vram=8.0)
        self.assertEqual(code, 0, out)
        self.assertNotIn("has not been tried on Windows", out)

    def test_size_selected_by_available_ram(self):
        for ram, expected in ((32.0, "UD-Q4_K_XL"), (10.0, "UD-Q2_K_XL")):
            with self.subTest(ram=ram):
                code, out, _ = self.main(family="inkling", ram=ram, vram=8.0)
                self.assertEqual(code, 0, out)
                self.assertTrue((self.t / f"strata-inkling-{expected.lower()}.json").is_file(), out)


if __name__ == "__main__":
    unittest.main()
