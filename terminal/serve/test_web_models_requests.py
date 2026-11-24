"""The web page's model list (GET /api/models, POST /api/models/load) and its Requests tab (#142, the API monitor). Both are
PolyStrata's own and both were lost once in a merge of upstream's page, so these checks read the sources for them. What they do
in a real browser is in tools/webtest/browser_test.mjs ("models", "requests"), run here when WEBTEST_MODULES is set.

    WEBTEST_MODULES=<folder> python -m unittest serve.test_web_models_requests -v
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = Path(__file__).parent / "web"


class TheSourcesKeepThem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = (WEB / "app.js").read_text(encoding="utf-8")
        cls.html = (WEB / "index.html").read_text(encoding="utf-8")
        cls.monitor = (WEB / "monitor.js").read_text(encoding="utf-8")

    def test_the_model_list_is_loaded_and_switches(self):
        self.assertIn('id="model-select"', self.html)
        self.assertIn('fetch("api/models", {headers: headers()})', self.app)
        self.assertIn('fetch("api/models/load", {method: "POST"', self.app)
        self.assertRegex(self.app, r"loadHealth\(\)\.then\(loadMcp\);\s+loadModels\(\);", "start() loads the list")

    def test_the_requests_tab_shows_and_opens_the_monitor_in_its_own_tab(self):
        tab = re.search(r'<button[^>]*id="tab-btn-requests"[^>]*>', self.html).group(0)
        self.assertNotIn("hidden", tab, "the tab is always there, with how to turn the monitor on when it is off")
        self.assertRegex(self.html, r'<a [^>]*id="requests-open" href="api-monitor" target="_blank"')
        self.assertNotIn("<iframe", self.html, "the page's policy (frame-src 'none') allows no frame")
        self.assertIn('if (tab === "requests") showRequests();', self.app)
        self.assertIn('localStorage.getItem("strata.apikey")', self.monitor, "the monitor uses the key saved on the chat page")


@unittest.skipUnless(os.environ.get("WEBTEST_MODULES") and shutil.which("node"),
                     "set WEBTEST_MODULES (a folder with node_modules/puppeteer-core) to run the browser tests")
class InABrowser(unittest.TestCase):
    def test_models_and_requests_in_chrome(self):
        for name in ("models", "requests"):
            with self.subTest(name=name):
                r = subprocess.run([shutil.which("node"), str(ROOT / "tools" / "webtest" / "browser_test.mjs"), name],
                                   capture_output=True, text=True, timeout=300)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
