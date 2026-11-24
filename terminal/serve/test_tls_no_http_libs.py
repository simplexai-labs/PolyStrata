"""#3722's certificate setup in the server leaves urllib3 and requests unloaded: truststore patches them when they are
there, and importing them for that kept about 11 MiB in every server, which uses neither.  An import of them afterwards
still works, with the operating system's certificates.

    python -m serve.test_tls_no_http_libs
"""
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROBE = r"""
import importlib.util, json, ssl, sys
import serve.server
from strata_control import tls
out = {"mode": tls.mode, "loaded": [n for n in ("urllib3", "requests") if n in sys.modules],
       "held": [n for n in ("urllib3", "requests") if n in sys.modules and sys.modules[n] is None],
       "ssl_context": ssl.SSLContext.__module__}
if importlib.util.find_spec("requests") is not None:
    import requests.adapters
    ctx = getattr(requests.adapters, "_preloaded_ssl_context", None)
    out["requests"] = type(ctx).__module__ if ctx is not None else "none"
print(json.dumps(out))
"""


class NoHttpLibs(unittest.TestCase):
    def probe(self):
        r = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, capture_output=True, text=True, timeout=120,
                           env=dict(os.environ, PYTHONPATH=ROOT))
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        import json
        return json.loads(r.stdout.strip().splitlines()[-1])

    def test_the_server_leaves_them_unloaded_and_importable(self):
        out = self.probe()
        self.assertEqual(out["loaded"], [], "the certificate setup loaded urllib3 or requests")
        self.assertEqual(out["held"], [], "a placeholder was left in sys.modules")
        if out["mode"] == "truststore":
            self.assertTrue(out["ssl_context"].startswith("truststore"), out)
            # imported afterwards, a requests that keeps an SSL context (2.32 and later) made it the operating system's too
            if out.get("requests", "none") != "none":
                self.assertTrue(out["requests"].startswith("truststore"), out)


if __name__ == "__main__":
    unittest.main()
