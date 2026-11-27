"""Unit tests for tools/check: worker defaults (CI vs local vs explicit --jobs), xdist dispatch mode, fallback without xdist, and JUnit parsing.

    python -m pytest tools/test_check.py      (python tools/check --only py-tools runs it)
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def _load_check():
    path = str(ROOT / "tools" / "check")
    loader = importlib.machinery.SourceFileLoader("polystrata_tools_check", path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


CHK = _load_check()


class WorkerDefaults(unittest.TestCase):
    def test_is_ci_distinguishes_truthy_and_falsy_values(self):
        for val in ("true", "True", "1", "yes"):
            self.assertTrue(CHK.is_ci({"CI": val}), val)
        for val in ("", "0", "false", "False", "no", "off", "  "):
            self.assertFalse(CHK.is_ci({"CI": val}), val)
        self.assertFalse(CHK.is_ci({}))

    def test_ci_default_fills_available_cores_up_to_eight(self):
        ctx = CHK.Ctx(None, 1.0, jobs=None)
        self.assertEqual(CHK.pytest_workers(ctx, env={"CI": "true"}, cpus=4), 4)
        self.assertEqual(CHK.pytest_workers(ctx, env={"CI": "1"}, cpus=2), 2)
        self.assertEqual(CHK.pytest_workers(ctx, env={"CI": "true"}, cpus=16), 8)
        self.assertEqual(CHK.default_build_jobs(env={"CI": "true"}, cpus=4), 4)

    def test_local_default_leaves_headroom_and_caps_at_eight(self):
        ctx = CHK.Ctx(None, 1.0, jobs=None)
        self.assertEqual(CHK.pytest_workers(ctx, env={}, cpus=4), 2)
        self.assertEqual(CHK.pytest_workers(ctx, env={"CI": "false"}, cpus=8), 4)
        self.assertEqual(CHK.pytest_workers(ctx, env={"CI": "0"}, cpus=24), 8)
        self.assertEqual(CHK.pytest_workers(ctx, env={}, cpus=1), 1)
        self.assertEqual(CHK.default_build_jobs(env={}, cpus=24), 12)

    def test_explicit_jobs_overrides_ci_and_local_defaults(self):
        self.assertEqual(CHK.pytest_workers(CHK.Ctx(None, 1.0, jobs=1), env={"CI": "true"}, cpus=4), 1)
        self.assertEqual(CHK.pytest_workers(CHK.Ctx(None, 1.0, jobs=6), env={}, cpus=4), 6)
        self.assertEqual(CHK.pytest_workers(CHK.Ctx(None, 1.0, jobs=0), env={"CI": "true"}, cpus=4), 0)


class StepPytestCommand(unittest.TestCase):
    def _run_step(self, step_name, *, jobs=None, env_over=None, has_xdist=True, cpus=4, shard=None):
        seen = {}

        def fake_have(mod):
            return mod == "pytest" or (mod == "xdist" and has_xdist)

        def fake_run(cmd, cwd, env, timeout, logfile, shell=False):
            seen["cmd"] = list(cmd)
            seen["cwd"] = cwd
            seen["env"] = dict(env)
            for arg in cmd:
                if arg.startswith("--junitxml="):
                    Path(arg.split("=", 1)[1]).write_text(
                        '<?xml version="1.0" encoding="utf-8"?>'
                        '<testsuites><testsuite tests="1">'
                        '<testcase classname="a" name="test_ok" time="0.01"/>'
                        "</testsuite></testsuites>",
                        encoding="utf-8",
                    )
            return 0, 0.1

        with tempfile.TemporaryDirectory() as tmp:
            scratch = CHK.Scratch(tmp, keep=True)
            ctx = CHK.Ctx(scratch, 1.0, jobs=jobs, shard=shard)
            base_env = {k: v for k, v in os.environ.items() if k not in ("CI", "STRATA_SNAPSHOT_UPDATE")}
            if env_over:
                base_env.update(env_over)
            with mock.patch.dict(os.environ, base_env, clear=True), \
                    mock.patch.object(CHK, "have_module", fake_have), \
                    mock.patch.object(CHK, "available_cpus", lambda: cpus), \
                    mock.patch.object(CHK, "run_cmd", fake_run):
                res = CHK.STEP_FUNCS[step_name](ctx)
        return res, seen["cmd"]

    def test_standard_suites_use_loadfile_in_ci(self):
        for step in ("py-tests", "py-serve", "py-tools"):
            res, cmd = self._run_step(step, env_over={"CI": "true"}, cpus=4)
            self.assertEqual(res.status, "passed")
            i = cmd.index("-n")
            self.assertEqual(cmd[i:i + 4], ["-n", "4", "--dist", "loadfile"], step)

    def test_snapshot_uses_worksteal_by_default_and_serial_when_updating(self):
        res, cmd = self._run_step("snapshot", env_over={"CI": "true"}, cpus=4)
        self.assertEqual(res.status, "passed")
        i = cmd.index("-n")
        self.assertEqual(cmd[i:i + 4], ["-n", "4", "--dist", "worksteal"])

        res_upd, cmd_upd = self._run_step("snapshot", jobs=4, env_over={"CI": "true", "STRATA_SNAPSHOT_UPDATE": "1"}, cpus=4)
        self.assertEqual(res_upd.status, "passed")
        self.assertNotIn("-n", cmd_upd)
        self.assertNotIn("--dist", cmd_upd)

    def test_jobs_one_runs_single_process_even_in_ci(self):
        res, cmd = self._run_step("py-tools", jobs=1, env_over={"CI": "true"}, cpus=4)
        self.assertEqual(res.status, "passed")
        self.assertNotIn("-n", cmd)
        self.assertNotIn("--dist", cmd)

    def test_missing_xdist_falls_back_to_single_process(self):
        res, cmd = self._run_step("py-tests", env_over={"CI": "true"}, has_xdist=False, cpus=4)
        self.assertEqual(res.status, "passed")
        self.assertNotIn("-n", cmd)
        self.assertNotIn("--dist", cmd)

    def test_shard_preserves_xdist_and_clamps_loadfile_to_file_count(self):
        res, cmd = self._run_step("py-tools", jobs=4, shard=(1, 4))
        self.assertEqual(res.status, "passed")
        i = cmd.index("-n")
        self.assertEqual(cmd[i:i + 4], ["-n", "4", "--dist", "loadfile"])

        res_snap, cmd_snap = self._run_step("snapshot", jobs=4, shard=(1, 6))
        self.assertEqual(res_snap.status, "passed")
        i = cmd_snap.index("-n")
        self.assertEqual(cmd_snap[i:i + 4], ["-n", "4", "--dist", "worksteal"])

    def test_parse_junit_handles_xdist_module_skip_and_outcomes(self):
        xml = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<testsuites name="pytest tests">'
            '<testsuite name="pytest" errors="0" failures="1" skipped="1" tests="3">'
            '<testcase classname="" name="tools.test_mod" time="0.000">'
            '<skipped message="collection skipped">'
            "('tools/test_mod.py', 21, 'Skipped: torch is not installed')"
            "</skipped></testcase>"
            '<testcase classname="pkg.TestA" name="test_ok" time="0.1"/>'
            '<testcase classname="pkg.TestA" name="test_bad" time="0.1"><failure message="boom"/></testcase>'
            "</testsuite></testsuites>"
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "out.xml"
            p.write_text(xml, encoding="utf-8")
            r = CHK.Result("py-tools")
            self.assertTrue(CHK.parse_junit(str(p), r))
        self.assertEqual((r.passed, r.failed, r.skipped), (1, 1, 1))
        self.assertEqual(r.failures, ["pkg.TestA::test_bad"])
        self.assertEqual(r.skips, {"('tools/test_mod.py', 21, 'torch is not installed')": 1})


if __name__ == "__main__":
    unittest.main()
