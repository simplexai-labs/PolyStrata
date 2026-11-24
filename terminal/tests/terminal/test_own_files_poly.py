"""(PolyStrata) "Use my own model files" knows the strata-poly families' sizes: data/gguf_fingerprints.json has a row for each size setup.py
installs, made from setup's own tables (file names, pinned bytes and SHA-256, the family's architectures), and a set with those file names
and that architecture is that size, whatever its tokenizer or expert types."""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

from strata_control import models

sys.path.insert(0, str(models.ROOT / "tools"))
import strata_inspect  # noqa: E402

TABLE = models.ROOT / "data" / "gguf_fingerprints.json"


def string(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def gguf(arch: str | None) -> bytes:
    """A GGUF header with no tensors: the architecture (none for a later shard of a split set) and a SentencePiece-style tokenizer."""
    pairs = [("general.architecture", 8, string(arch)), ("tokenizer.ggml.model", 8, string("llama"))] if arch else []
    head = b"GGUF" + struct.pack("<IQQ", 3, 0, len(pairs)) + b"".join(string(k) + struct.pack("<I", kind) + v for k, kind, v in pairs)
    return head + b"\x00" * (-len(head) % 32)


class Rows(unittest.TestCase):
    def test_the_table_has_what_setup_installs(self):
        table = json.loads(TABLE.read_text(encoding="utf-8"))
        mine = {k: v for k, v in table.items() if isinstance(v, dict) and v.get("engine") == "strata-poly"}
        self.assertEqual(mine, json.loads(json.dumps(strata_inspect.poly_rows())), "python tools/strata_inspect.py --poly-rows data/gguf_fingerprints.json")
        self.assertIn("glm/UD-IQ2_XXS", mine)
        self.assertIn("gemma4/UD-Q3_K_M", mine)
        self.assertLess(TABLE.stat().st_size, 128 * 1024)                  # models._candidate_selection refuses a larger table


class Import(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="strata-own-poly-")
        self.addCleanup(self.tmp.cleanup)
        self.rows = json.loads(TABLE.read_text(encoding="utf-8"))

    def put(self, key: str, arch: str) -> Path:
        folder = Path(self.tmp.name) / key.replace("/", "-")
        folder.mkdir()
        for i, name in enumerate(self.rows[key]["files"]):
            (folder / Path(name).name).write_bytes(gguf(arch if i == 0 else None))
        return folder

    def test_a_single_file_size_is_known_by_its_name_and_architecture(self):
        plan = models.plan_import(str(self.put("gemma4/UD-Q3_K_M", "gemma4")))
        self.assertEqual(plan["issues"], [])
        c = plan["candidate_selection"]
        self.assertEqual((c["family"], c["model"], c["engine"]), ("gemma4", "UD-Q3_K_M", "strata-poly"))
        self.assertEqual(c["setup_args_hint"][:4], ["--family", "gemma4", "--model", "UD-Q3_K_M"])

    def test_a_split_set_is_known_too(self):
        plan = models.plan_import(str(self.put("glm/UD-IQ2_XXS", "glm5next")))
        self.assertEqual(plan["issues"], [])
        self.assertEqual((plan["candidate_selection"]["family"], plan["candidate_selection"]["model"]), ("glm", "UD-IQ2_XXS"))

    def test_qwen36_on_strata_poly_is_its_own_size_not_the_own_path_guess(self):
        plan = models.plan_import(str(self.put("qwen36/UD-Q3_K_M", "qwen35moe")))
        self.assertEqual(plan["issues"], [])
        self.assertEqual((plan["candidate_selection"]["family"], plan["candidate_selection"]["model"]), ("qwen36", "UD-Q3_K_M"))

    def test_the_names_of_a_size_with_another_architecture_are_refused(self):
        plan = models.plan_import(str(self.put("gemma4/UD-Q3_K_M", "llama")))
        self.assertIsNone(plan["candidate_selection"])
        self.assertTrue(any("unsupported_architecture" in i for i in plan["issues"]), plan["issues"])


if __name__ == "__main__":
    unittest.main()
