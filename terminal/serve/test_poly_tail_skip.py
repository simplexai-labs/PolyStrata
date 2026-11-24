"""strata-poly's tail skip comes from its run config (Qwen3.6: --tail-skip 6, as setup writes it since poly.10).  0.2.17's server sets
STRATA_ROUTE_TAIL_SKIP=0 at every start of the Strata project's engine, and strata-poly reads that variable over --tail-skip, so the
server must not set it for strata-poly; a 0 the person or the config gives still turns the skip off.

    python -m pytest serve/test_poly_tail_skip.py
"""
import os
import unittest
from unittest import mock

from serve import server

NAME = "STRATA_ROUTE_TAIL_SKIP"
POLY = {"exe": "/x/engine-poly/strata-poly", "backend": "cuda", "args": ["--model", "m.gguf", "--tail-skip", "6"]}


def start_env(cfg, own=None):
    """child_env as a server started with this process environment (`own`) builds it."""
    with mock.patch.dict(os.environ, own or {}, clear=False):
        if not (own and NAME in own):
            os.environ.pop(NAME, None)
        return server.child_env(cfg)


class PolyTailSkip(unittest.TestCase):
    def test_the_server_leaves_strata_polys_config_to_decide(self):
        self.assertNotIn(NAME, start_env(POLY))
        self.assertNotIn(NAME, start_env({**POLY, "exe": "C:/x/engine-poly/strata-poly.exe"}))

    def test_a_zero_the_person_or_the_config_gives_still_turns_it_off(self):
        self.assertEqual(start_env(POLY, {NAME: "0"})[NAME], "0")
        self.assertEqual(start_env({**POLY, "env": {NAME: "0"}})[NAME], "0")

    def test_the_strata_projects_engine_still_starts_with_it_off(self):
        self.assertEqual(start_env({"exe": "/x/engine/strata", "backend": "cuda"})[NAME], "0")
        self.assertEqual(start_env({"exe": "component:strata-engine", "backend": "cuda"})[NAME], "0")


if __name__ == "__main__":
    unittest.main()
