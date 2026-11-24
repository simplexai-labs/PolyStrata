"""#195: an engine that ends while the server is not ending it is named, with the time and its exit status, in the
server's output and in the engine log; one the server ends itself (close, restart, a lost step) is not."""
import io
import os
from pathlib import Path
import queue
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve.server as server                                            # noqa: E402
from serve.server import StrataEngine                                    # noqa: E402

UNASKED = "the engine ended without the server asking it to"


def pumped(log, rc=3, **state):
    """An engine whose output has ended, as _pump() sees it; `state`: what the server set before (ended, closing)."""
    item = object.__new__(StrataEngine)
    item.lines, item.slot_q, item.slot_busy = queue.Queue(), [], []
    item.log, item.ended, item.last_err = log, False, None
    item.__dict__.update(state)
    item.proc = SimpleNamespace(stdout=iter(["READY 4096 stop\n"]), wait=lambda timeout=None: rc)
    return item


class UnaskedExitNote(unittest.TestCase):
    def test_an_engine_that_ends_by_itself_is_named_with_its_exit_status(self):
        log = io.StringIO()
        item = pumped(log, rc=3)
        with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            item._pump()
        for text in (log.getvalue(), out.getvalue()):
            self.assertIn(UNASKED + " (exit code 3)", text)
            self.assertRegex(text, r"\[strata\] \d{4}-\d\d-\d\d \d\d:\d\d:\d\d ")
        self.assertTrue(item.ended)
        self.assertEqual([item.lines.get_nowait(), item.lines.get_nowait()], ["READY 4096 stop\n", None])

    def test_a_windows_crash_status_is_named(self):
        log = io.StringIO()
        with mock.patch("sys.stdout", new_callable=io.StringIO):
            pumped(log, rc=0xC0000005)._pump()
        self.assertIn("0xC0000005", log.getvalue())

    def test_an_end_the_server_asked_for_is_not_named(self):
        for state in ({"closing": True}, {"ended": True}):
            with self.subTest(state=state):
                log = io.StringIO()
                with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
                    pumped(log, **state)._pump()
                self.assertNotIn(UNASKED, log.getvalue() + out.getvalue())

    def test_the_note_goes_to_the_ended_engines_log(self):
        original, successor = io.StringIO(), io.StringIO()
        item = pumped(original)

        def output():
            item.log = successor                        # a restart's new log, opened while this pump ran
            yield "READY 4096 stop\n"

        item.proc.stdout = output()
        with mock.patch("sys.stdout", new_callable=io.StringIO):
            item._pump()
        self.assertIn(UNASKED, original.getvalue())
        self.assertEqual(successor.getvalue(), "")

    def test_an_exit_status_that_cannot_be_read_still_ends_the_pump(self):
        def wait(timeout=None):
            raise server.subprocess.TimeoutExpired("strata-poly", timeout)
        log = io.StringIO()
        item = pumped(log)
        item.proc.wait = wait
        with mock.patch("sys.stdout", new_callable=io.StringIO):
            item._pump()
        self.assertIn(UNASKED + " (its exit status could not be read)", log.getvalue())
        self.assertEqual([item.lines.get_nowait(), item.lines.get_nowait()], ["READY 4096 stop\n", None])


READY_THEN_EXIT = r'''
import sys, time
print('READY 4096 stop', flush=True)
time.sleep(0.3)
sys.exit(3)
'''

READY_UNTIL_QUIT = r'''
import sys
print('READY 4096 stop', flush=True)
for line in sys.stdin:
    if line.startswith('QUIT'):
        break
'''


@unittest.skipIf(os.name == "nt", "the .cmd wrapper keeps the pipe open after the kill (only the wrapper dies), so the "
                                  "pump's end comes late; a real engine is one process")
class UnaskedExitProcess(unittest.TestCase):
    """The same with a real process: an engine idle after READY that exits, and one that a close() ends."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.log = self.root / "engine.log"

    def script(self, name, code):
        py = self.root / (name + ".py")
        py.write_text("#!" + sys.executable + "\n" + code)
        py.chmod(0o700)
        return str(py)

    def log_text(self):
        return self.log.read_text(encoding="utf-8", errors="replace") if self.log.is_file() else ""

    def test_an_idle_engine_that_exits_is_named_and_not_loaded(self):
        eng = StrataEngine(self.script("exits", READY_THEN_EXIT), [], log=str(self.log))
        self.addCleanup(eng.close)
        self.assertTrue(eng.alive())
        deadline = time.monotonic() + 15
        while UNASKED not in self.log_text() and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertIn(UNASKED + " (exit code 3)", self.log_text())
        self.assertFalse(eng.alive())                   # /health's "loaded"

    def test_a_close_is_not_named(self):
        eng = StrataEngine(self.script("quits", READY_UNTIL_QUIT), [], log=str(self.log))
        eng.close()
        time.sleep(0.5)
        self.assertNotIn(UNASKED, self.log_text())


if __name__ == "__main__":
    unittest.main()
