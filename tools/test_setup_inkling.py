"""Tests for setup.py's Inkling 2 Mini (docs/models/inkling.md): its three pinned files, the estimates for an 8 GB
card, the install by GLM's steps with no draft model, the model's sampling defaults in the run config, NVIDIA only,
and its image encoder: opt-in, built from its own llama.cpp pin into a folder of its own (the other families' encoder
and its stamp are left alone).  Mocked like tools/test_setup_glm.py, whose harness this uses.

    python -m unittest tools.test_setup_inkling
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
from tools import test_setup_glm as glm_tests  # noqa: E402

FAM = setup.FAMILIES["inkling"]
# setup's order: the size taken with the RAM for it, then Unsloth's 2-bit file (the least RAM), then 8-bit
SIZES = [("IQ4_NL", 17627940768), ("UD-Q2_K_XL", 9138715328), ("Q8_0", 27820099264)]
MMPROJ = "mmproj-Inkling-2-Mini-Q8_0.gguf"


class Pins(unittest.TestCase):
    def test_the_files(self):
        self.assertEqual(list(setup.family_models("inkling")), [m for m, _ in SIZES])
        self.assertEqual(FAM["hf"], "https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/resolve/"
                                    "db61c5b492a6a034d986328d28274af0a439aa6c/")
        self.assertEqual(list(FAM["sha256"]), [f"Inkling-2-Mini-{m}.gguf" for m, _ in SIZES] + [MMPROJ])
        for size, n in SIZES:
            file = f"Inkling-2-Mini-{size}.gguf"
            d = FAM["models"][size]
            self.assertEqual((setup.model_file(FAM, size, 1), setup.model_shards(FAM, size)), (file, 1))
            self.assertEqual(FAM["sha256"][file][0], n)
            self.assertRegex(FAM["sha256"][file][1], r"^[0-9a-f]{64}$")
            self.assertAlmostEqual(n / 1e9, d["download_gb"], delta=0.05)
            self.assertLess(d["arena_gb"], d["download_gb"])
            self.assertEqual(d["ram_gb"], round(setup.poly_ram_need(d, 8.0)), size)
            if size == "UD-Q2_K_XL":   # Unsloth's repository, in the architecture's native form (inkling2)
                self.assertEqual(d["hf"], "https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/"
                                          "ae9327c5fbc7333116ba32eee78bb5d2bbefae4f/")
                self.assertEqual(d["mmproj_hf"], FAM["mmproj_hf"])   # ggml-org's image encoder for it too
            else:
                self.assertNotIn("hf", d)
                self.assertNotIn("mmproj_hf", d)
        self.assertEqual(setup.POLY_GGUF_ARCHS["inkling"], ("inkling2_loop", "inkling2"))
        self.assertFalse(setup.poly_batch(FAM, FAM["models"]["IQ4_NL"], 31.8, 32768))   # the engine's prompt batch

    def test_the_about_line_and_what_is_not_claimed(self):
        n = setup.poly_ram_need(FAM["models"]["IQ4_NL"], 8.0)
        self.assertIn(f"~{n:.0f} GB of RAM beside an 8 GB card (16.5 GiB measured for the engine)", FAM["about"])
        self.assertIs(FAM["experimental"], True)
        self.assertNotIn("unmeasured", FAM)          # measured on an RTX 2070 Super 8 GB (the comment above FAMILIES)
        self.assertIs(FAM["windows"], False)
        self.assertIs(FAM["vision"], True)
        self.assertIn("images optional", FAM["about"])
        self.assertEqual(FAM["ran_on_gb"], 8)          # run on an RTX 2070 Super 8 GB

    def test_the_image_encoder(self):
        self.assertEqual(FAM["mmproj"], MMPROJ)
        self.assertEqual(FAM["mmproj_hf"], FAM["hf"])                    # the same pinned revision of ggml-org's
        self.assertEqual(FAM["sha256"][MMPROJ],
                         (345910944, "233287272ec5ac39b4168677aba54a4c024f202410f5629fac1830e1059f8abb"))
        self.assertAlmostEqual(FAM["sha256"][MMPROJ][0] / 1e9, FAM["vision_gb"], delta=0.01)
        # its own llama.cpp (mtmd's inkling2v): a full commit, not the other families' encoder's or the engines'
        self.assertRegex(FAM["vision_llama"], r"^[0-9a-f]{40}$")
        self.assertEqual(FAM["vision_llama"], setup.INKLING_VISION_LLAMA_CPP_COMMIT)
        self.assertNotIn(FAM["vision_llama"], (setup.VISION_LLAMA_CPP_COMMIT, setup.LLAMA_CPP_COMMIT))
        self.assertEqual([f for f in setup.FAMILIES if setup.FAMILIES[f].get("vision_llama")], ["inkling"])

    def test_which_llama_cpp_checkout(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            for name in ("llama.cpp", f"llama.cpp-{setup.VISION_LLAMA_CPP_COMMIT[:7]}",
                         f"llama.cpp-{FAM['vision_llama'][:7]}"):
                (root / "third_party" / name / "ggml").mkdir(parents=True)
                (root / "third_party" / name / "ggml" / "CMakeLists.txt").write_text("")
                (root / "third_party" / name / "gguf-py").mkdir()
            with mock.patch.object(setup, "ROOT", root), \
                    mock.patch.object(setup, "download", mock.Mock(side_effect=AssertionError("downloaded"))):
                self.assertEqual(setup.get_llama_cpp().name, "llama.cpp")
                self.assertEqual(setup.get_llama_cpp(vision=True).name,           # the others' encoder: as before
                                 f"llama.cpp-{setup.VISION_LLAMA_CPP_COMMIT[:7]}")
                self.assertEqual(setup.get_llama_cpp(commit=FAM["vision_llama"]).name, "llama.cpp-29b52d3")
                self.assertEqual(setup.get_llama_cpp(commit=setup.LLAMA_CPP_COMMIT).name, "llama.cpp")
            urls = []
            with mock.patch.object(setup, "ROOT", root / "empty"), \
                    mock.patch.object(setup, "download", mock.Mock(side_effect=lambda url, *a: urls.append(url) or
                                                                   (_ for _ in ()).throw(OSError()))):
                with self.assertRaises(OSError):
                    setup.get_llama_cpp(commit=FAM["vision_llama"])
            self.assertEqual(urls, [f"https://github.com/ggml-org/llama.cpp/archive/{FAM['vision_llama']}.zip"])

    def test_nvidia_only(self):
        self.assertIn("inkling", setup.menu_families(False))
        self.assertNotIn("inkling", setup.menu_families(True))
        self.assertIn("inkling", setup.POLY_KV["q8_0"])      # fp16 or q8_0 caches
        sys.path.insert(0, str(ROOT / "metal"))
        import setup_mac
        self.assertNotIn("inkling", setup_mac.MAC_POLY_FAMILIES)


class Main(unittest.TestCase):
    setUp, tearDown = glm_tests.Main.setUp, glm_tests.Main.tearDown

    def main(self, *args, **kw):
        """glm's harness (its get_llama_cpp gives a commit its own folder), with the own encoder's compile mocked:
        what it was given in self.vbuilt."""
        self.vbuilt = []
        self.nvcc = self.t / "cuda" / "bin" / "nvcc"          # the toolkit's bin and lib64: the encoder's cuda_dirs

        def fake_cmake(src, bdir, target, defs, vcvars, bat):
            self.vbuilt.append((src, bdir, target, defs))
            (bdir / "bin").mkdir(parents=True, exist_ok=True)
            (bdir / "bin" / setup.VEXE).write_text("")

        with mock.patch.object(setup, "cmake_build", fake_cmake), \
                mock.patch.object(setup, "install_build_tools", lambda gpu, yes: (str(self.nvcc), None)):
            (self.t / "cuda" / "bin").mkdir(parents=True, exist_ok=True)
            (self.t / "cuda" / "lib64").mkdir(exist_ok=True)
            return glm_tests.Main.main(self, *args, **kw)

    def test_install_each_size(self):
        for size, _n in SIZES:
            with self.subTest(size):
                self.downloads, self.runs, self.verified = [], [], []
                code, out, _ = self.main(family="inkling", model=size, ram=64.0, vram=8188 / 1024)
                self.assertEqual(code, 0, out)
                cfg = json.loads((self.t / f"strata-inkling2-{size.lower()}.json").read_text())
                file = f"Inkling-2-Mini-{size}.gguf"
                self.assertEqual(self.downloads, [FAM["models"][size].get("hf", FAM["hf"]) + file])
                self.assertEqual(self.verified, [(file, *FAM["sha256"][file])])
                tag = "inkling2-" + size
                pack = self.t / "data" / "packs" / tag.lower()
                self.assertEqual([Path(r[1]).name for r in self.runs], ["strata_tokenizer.py"])
                args = cfg["args"]
                self.assertEqual(Path(args[args.index("--model") + 1]), self.t / "models" / tag / file)
                self.assertEqual(Path(args[args.index("--expert-profile") + 1]), pack / "expert-profile.txt")
                for flag in ("--draft-model", "--tail-skip", "--prompt-batch", "--kv"):
                    self.assertNotIn(flag, args)
                self.assertEqual(cfg["model_name"], f"inkling-2-mini-{size.lower()}")
                self.assertEqual(cfg["display_name"], f"Inkling 2 Mini {size}")
                self.assertNotIn("vision", cfg)
                self.assertNotIn("tool_call_recovery", cfg)
                self.assertEqual(cfg["sampling"], {"temperature": 1.0, "top_k": 50, "top_p": 1.0})
                self.assertNotIn("EXPERIMENTAL and new", out)   # measured: no "not measured yet" warning

    def test_a_sampling_block_of_the_users_is_kept(self):
        code, out, cfg = self.main(family="inkling", ram=64.0, vram=31.84)
        self.assertEqual(code, 0, out)
        path = self.t / "strata-inkling2-iq4_nl.json"
        path.write_text(json.dumps({**cfg, "sampling": {"temperature": 0.6}}))
        code, out, cfg = self.main(family="inkling", ram=64.0, vram=31.84)
        self.assertEqual(code, 0, out)
        self.assertEqual(json.loads(path.read_text())["sampling"], {"temperature": 0.6})

    def test_the_size_taken_without_a_name(self):
        for ram in (64.0, 20.0):                      # IQ4_NL wants the least RAM too
            with self.subTest(ram=ram):
                code, out, _ = self.main(family="inkling", ram=ram, vram=8.0)
                self.assertEqual(code, 0, out)
                self.assertTrue((self.t / "strata-inkling2-iq4_nl.json").is_file(), out)

    def test_windows_is_asked(self):
        code, out, _ = self.main(win=True, family="inkling", ram=64.0, vram=31.84)
        self.assertEqual(code, 0, out)                # --yes takes the risk
        self.assertIn("Inkling 2 Mini has not been built or run on Windows", out)
        self.assertNotIn("has been run on Linux only", out)

    def test_one_card(self):
        # its engine code has no --layer-split yet: setup writes none
        code, out, _ = self.main(["--gpus", "0,1"], family="inkling", ram=64.0, n_gpus=2)
        self.assertNotEqual(code, 0)
        self.assertIn("runs on one card for now", out)
        code, out, cfg = self.main(["--gpu", "1"], family="inkling", ram=64.0, n_gpus=2)
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["gpu"], 1)
        code, out, cfg = self.main(["--kv", "q8_0"], family="inkling", ram=64.0)
        self.assertEqual(code, 0, out)
        self.assertIn("--kv", cfg["args"])
        self.assertEqual(cfg["args"][cfg["args"].index("--kv") + 1], "q8_0")

    def test_requests_at_once(self):
        # "parallel" as for the other families of strata-poly: written when asked, suggested where it pays
        self.assertNotIn("one_request", FAM)
        code, out, cfg = self.main(["--parallel", "3"], family="inkling", ram=64.0)
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["parallel"], 3)
        self.assertIn("parallel requests: 3 at once", out)
        code, out, cfg = self.main(family="inkling", ram=64.0, vram=8188 / 1024)
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["parallel"], 3)                # the earlier config's, carried over
        self.assertIn("Several requests at once (opt-in, --parallel N)", out)   # an 8 GB card: most experts on the CPU
        code, out, cfg = self.main(["--parallel", "1"], family="inkling", ram=64.0)
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["parallel"], 1)
        code, out, cfg = self.main(family="inkling", ram=64.0)
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["parallel"], 1)                # an earlier "parallel": 1 is kept too

    def test_no_images_unless_asked(self):
        code, out, cfg = self.main(family="inkling", ram=64.0, vram=8188 / 1024)      # --yes: the default, no
        self.assertEqual(code, 0, out)
        self.assertNotIn("vision", cfg)
        self.assertEqual((self.encoder, self.vbuilt), ("none", []))
        self.assertFalse((self.t / setup.ENGINE_POLY_DIR / "vision-inkling").exists())
        self.assertNotIn(MMPROJ, " ".join(self.downloads))
        self.downloads = []
        code, out, cfg = self.main(family="inkling", ram=64.0, vram=8188 / 1024, answers={})   # Enter: no
        self.assertEqual(code, 0, out)
        self.assertTrue(any("Do you want images?" in q for q in self.asked), self.asked)
        self.assertNotIn("vision", cfg)
        self.assertEqual(self.vbuilt, [])

    def test_images_on_the_cpu_of_an_8_gb_card(self):
        code, out, cfg = self.main(["--vision", "yes"], family="inkling", ram=64.0, vram=8188 / 1024)
        self.assertEqual(code, 0, out)
        self.assertNotIn("not used with Inkling 2 Mini", out)
        own = self.t / setup.ENGINE_POLY_DIR / "vision-inkling"
        models = self.t / "models" / "inkling2-IQ4_NL"
        self.assertEqual(cfg["vision"], {"exe": str(own / setup.VEXE), "mmproj": str(models / MMPROJ),
                                         "model": str(models / "Inkling-2-Mini-IQ4_NL.gguf"), "gpu": False,
                                         "max_tokens": setup.VISION["cpu"]["max_tokens"],
                                         "threads": cfg["vision"]["threads"]})
        self.assertIn(FAM["mmproj_hf"] + MMPROJ, self.downloads)
        self.assertIn((MMPROJ, *FAM["sha256"][MMPROJ]), self.verified)
        # the shared engine is built without an encoder; its folder gets no strata-vision and its stamp no change
        self.assertEqual(self.encoder, "none")
        self.assertFalse((self.t / setup.ENGINE_POLY_DIR / setup.VEXE).exists())
        self.assertNotIn("vision", json.loads((self.t / setup.ENGINE_POLY_DIR / "BUILD.json").read_text()))
        [(src, bdir, target, defs)] = self.vbuilt
        self.assertEqual((src, bdir.name, target),
                         (self.t / "tools" / "vision", "build-vision-inkling", "strata-vision"))
        self.assertIn(f"-DLLAMA_DIR={self.t / 'llama.cpp-29b52d3'}", defs)
        self.assertIn("-DSTRATA_VISION_CUDA=OFF", defs)
        stamp = json.loads((own / "BUILD.json").read_text())
        with mock.patch.object(setup, "ROOT", self.t):
            vsrc = setup.source_hash(setup.VISION_SOURCES)
        self.assertEqual((stamp["llama"], stamp["vision"], stamp["vision_src"]), (FAM["vision_llama"], "cpu", vsrc))
        self.assertEqual(stamp["cuda_dirs"], [])                         # no CUDA in a CPU one
        # again: nothing to compile
        code, out, cfg = self.main(["--vision", "cpu"], family="inkling", ram=64.0, vram=8188 / 1024)
        self.assertEqual((code, self.vbuilt), (0, []), out)
        self.assertIn("image encoder already built", out)

    def test_images_on_the_gpu(self):
        code, out, cfg = self.main(["--vision", "gpu"], family="inkling", ram=64.0, vram=8188 / 1024)
        self.assertEqual(code, 0, out)
        self.assertIs(cfg["vision"]["gpu"], True)
        self.assertEqual(cfg["vision"]["max_tokens"], setup.VISION["gpu"]["max_tokens"])
        [(_src, _bdir, _target, defs)] = self.vbuilt
        self.assertIn("-DSTRATA_VISION_CUDA=ON", defs)
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=120", defs)
        self.assertIn(f"-DCMAKE_CUDA_COMPILER={self.nvcc}", defs)
        # the toolkit's libraries: in the encoder's stamp and the config's lib_dirs, as the shared build's are
        dirs = [str(self.t / "cuda" / "bin"), str(self.t / "cuda" / "lib64")]
        stamp = json.loads((self.t / setup.ENGINE_POLY_DIR / "vision-inkling" / "BUILD.json").read_text())
        self.assertEqual(stamp["cuda_dirs"], dirs)
        self.assertEqual([d for d in cfg["lib_dirs"] if d in dirs], dirs)
        # a CPU encoder asked of one built with CUDA: kept
        code, out, cfg = self.main(["--vision", "cpu"], family="inkling", ram=64.0, vram=8188 / 1024)
        self.assertEqual((code, self.vbuilt, cfg["vision"]["gpu"]), (0, [], False), out)

    def test_the_unsloth_size_takes_ggml_orgs_encoder(self):
        code, out, cfg = self.main(["--vision", "cpu"], family="inkling", model="UD-Q2_K_XL", ram=64.0,
                                   vram=8188 / 1024)
        self.assertEqual(code, 0, out)
        self.assertIn(FAM["mmproj_hf"] + MMPROJ, self.downloads)
        self.assertEqual(Path(cfg["vision"]["mmproj"]).name, MMPROJ)


class Update(unittest.TestCase):
    """update_poly_engine before a start: the engine first, then the own encoder (update_vision_own), compiled again
    when its stamp is stale; the shared one is not asked for, and another family's encoder is updated as before."""

    @staticmethod
    def hashes(root):
        with mock.patch.object(setup, "ROOT", root):
            return setup.source_hash(setup.ENGINE_SOURCES), setup.source_hash(setup.VISION_SOURCES)

    def run_update(self, root, vis_exe, own_stamp=None, engine_src=None, fails=None):
        """update_poly_engine on an install in root: (the shared build's mock, the own encoder's compiles, the
        output).  `fails`: what install_build_tools raises (the encoder's compile)."""
        eng = root / setup.ENGINE_POLY_DIR
        (eng / "vision-inkling").mkdir(parents=True, exist_ok=True)
        (eng / setup.POLY_EXE).write_text("")
        (eng / "BUILD.json").write_text(json.dumps({"src": engine_src or self.hashes(root)[0], "archs": [75],
                                                    "toolkit": 12, "vision": "cpu", "vision_src": "old"}))
        if own_stamp is not None:
            (eng / "vision-inkling" / setup.VEXE).write_text("")
            (eng / "vision-inkling" / "BUILD.json").write_text(json.dumps(own_stamp))
        cfg = {"exe": str(eng / setup.POLY_EXE), "args": [], "gpu": 0, "cuda": 12, "lib_dirs": ["/cuda12/lib64"],
               "vision": {"exe": str(vis_exe), "gpu": False}}
        compiles = []

        def fake_cmake(src, bdir, target, defs, vcvars, bat):
            compiles.append(defs)
            (bdir / "bin").mkdir(parents=True, exist_ok=True)
            (bdir / "bin" / setup.VEXE).write_text("")

        def tools(gpu, yes):
            if fails is not None:
                raise fails
            return str(root / "cuda12" / "bin" / "nvcc"), None

        engine = mock.Mock(return_value=eng)
        with mock.patch.object(setup, "ROOT", root), \
                mock.patch.object(setup, "get_llama_cpp", lambda *a, **k: root / "llama.cpp"), \
                mock.patch.object(setup, "gpu_info", lambda i=None: {"arch": "75"}), \
                mock.patch.object(setup, "build_engine_poly", engine), \
                mock.patch.object(setup, "cmake_build", fake_cmake), \
                mock.patch.object(setup, "install_build_tools", tools), \
                mock.patch.object(setup, "engine_lib_dirs", lambda e: ["/cuda12/lib64"]):
            out = glm_tests.quiet(setup.update_poly_engine, cfg)
        self.cfg = cfg
        return engine, compiles, out

    def test_the_own_encoder(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            exe = root / setup.ENGINE_POLY_DIR / "vision-inkling" / setup.VEXE
            engine, compiles, _ = self.run_update(root, exe)               # no stamp: compiled, for the CPU
            engine.assert_not_called()                          # the shared one's stale vision_src: not ours
            self.assertEqual(len(compiles), 1)
            self.assertIn("-DSTRATA_VISION_CUDA=OFF", compiles[0])
            self.assertIn(f"-DLLAMA_DIR={root / 'llama.cpp'}", compiles[0])
            fresh = {"llama": FAM["vision_llama"], "vision": "cpu", "toolkit": 12,
                     "vision_src": self.hashes(root)[1]}
            engine, compiles, _ = self.run_update(root, exe, fresh)
            self.assertEqual((engine.call_count, compiles), (0, []))
            # another toolkit: a CPU one is kept (as the shared check), a CUDA one is not
            with mock.patch.object(setup, "ROOT", root):
                self.assertTrue(setup.vision_own_ok("inkling", [75], "cpu", 13))
                gpu = {**fresh, "vision": "gpu", "archs": [75]}
                (exe.parent / "BUILD.json").write_text(json.dumps(gpu))
                self.assertTrue(setup.vision_own_ok("inkling", [75], "gpu", 12))
                self.assertFalse(setup.vision_own_ok("inkling", [75], "gpu", 13))
                self.assertFalse(setup.vision_own_ok("inkling", [86], "gpu", 12))      # another card
            # a CUDA one compiled again: on the GPU, and its toolkit's libraries reach the config's lib_dirs
            (root / "cuda12" / "bin").mkdir(parents=True)
            engine, compiles, _ = self.run_update(root, exe, {**gpu, "vision_src": "old"})
            self.assertEqual(len(compiles), 1)
            self.assertIn("-DSTRATA_VISION_CUDA=ON", compiles[0])
            self.assertEqual(self.cfg["lib_dirs"], ["/cuda12/lib64", str(root / "cuda12" / "bin")])
            engine, compiles, _ = self.run_update(root, exe, {**fresh, "llama": setup.VISION_LLAMA_CPP_COMMIT})
            self.assertEqual(len(compiles), 1)

    def test_a_failed_encoder_build_leaves_the_engine_update(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            exe = root / setup.ENGINE_POLY_DIR / "vision-inkling" / setup.VEXE
            for e in (SystemExit(1), OSError("no compiler")):
                engine, compiles, out = self.run_update(root, exe, engine_src="old", fails=e)
                engine.assert_called_once()                                 # the engine is still updated
                self.assertEqual(engine.call_args[0][3], "none")            # without the shared encoder
                self.assertIn("could not update the image encoder", out[1])
                self.assertNotIn("could not update strata-poly", out[1])

    def test_another_familys_encoder_as_before(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            engine, compiles, _ = self.run_update(root, root / setup.ENGINE_POLY_DIR / setup.VEXE)
            self.assertEqual(compiles, [])
            self.assertEqual(engine.call_args[0][3], "cpu")                 # the shared encoder, stale: rebuilt


if __name__ == "__main__":
    unittest.main()
