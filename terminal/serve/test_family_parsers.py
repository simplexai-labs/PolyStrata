"""Every strata-poly family of setup.py's catalog through the server's output parser, as the server builds it.

The server takes the parser class from the chat template's tool-call form (frontend.parser_for_template: JsonOutputParser for a
template that documents JSON calls, else OutputParser) and the call format from the tokenizer's pre-tokenizer
(Service._read_tokenizer: GLM's, Gemma's, DeepSeek's, LFM2's or the Qwen form), and builds the parser with both on every request
(Service.run).  Inkling 2 is the exception: its replies are typed blocks, so its pre-tokenizer picks the parser class as well.  0.2.16's merge gave JsonOutputParser upstream's arguments and the call site PolyStrata's: Flint and Kolibri, whose
templates read as JSON, failed every request with a TypeError (call_format).

fixtures/family_templates.json.gz holds, for each family, the template and pre-tokenizer the server is given: tokenizer.chat_template
and tokenizer.ggml.pre from the header of the first file setup pins (or setup's pinned chat_template.jinja where the file has none),
with the file and revision they came from (tools/family_templates.py writes it; a new family goes in MATRIX too).  Each
family here renders its own template with a tool, and a reply in its own call form comes back as one call; Flint and Kolibri also
over HTTP, streamed and not.

    python -m unittest serve.test_family_parsers
"""
from __future__ import annotations

import gzip
import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate, InklingOutputParser, JsonOutputParser, OutputParser, parser_for_template
from serve.server import ByteTokenizer, MockEngine, Service, serve
from serve import toolformat

HERE = Path(__file__).resolve().parent
with gzip.open(HERE / "fixtures" / "family_templates.json.gz", "rt", encoding="utf-8") as _f:
    FAMILIES = json.load(_f)

#: family: (the template's tool-call form, the parser class, the call format)
MATRIX = {
    "glm": ("json", JsonOutputParser, "glm"),
    "deepseek": ("dsml", OutputParser, "dsml"),
    "lfm25": ("other", OutputParser, "lfm2"),
    "flint": ("json", JsonOutputParser, "glm"),
    "kolibri": ("json", JsonOutputParser, "function"),
    "qwen36": ("xml", OutputParser, "function"),
    "clover": ("xml", OutputParser, "function"),
    "ornith": ("xml", OutputParser, "function"),
    "gemma4": ("other", OutputParser, "gemma"),
    "tiel": ("xml", OutputParser, "function"),
    "minicpmv": ("xml", OutputParser, "function"),
    "ornith122": ("xml", OutputParser, "function"),
    "groot": ("xml", OutputParser, "function"),
    "inkling": ("other", InklingOutputParser, "inkling"),
}

TOOL = {"type": "function", "function": {"name": "get_weather", "description": "The weather in a city.",
                                         "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                        "required": ["city"]}}}
D, Q = "｜DSML｜", '<|"|>'
#: call format: the call get_weather(city="Paris") in that form, and the token that ends the reply there
CALLS = {
    "function": ("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>", "<|im_end|>"),
    "glm": ("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>", "<|observation|>"),
    "gemma": (f"<|tool_call>call:get_weather{{city:{Q}Paris{Q}}}<tool_call|>", "<|tool_response>"),
    "dsml": (f'<{D}tool_calls>\n<{D}invoke name="get_weather">\n<{D}parameter name="city" string="true">Paris</{D}parameter>\n'
             f"</{D}invoke>\n</{D}tool_calls>", "<｜end▁of▁sentence｜>"),
    "lfm2": ("<|tool_call_start|>[get_weather(city='Paris')]<|tool_call_end|>", "<|im_end|>"),
    "inkling": ('<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Paris"}}'
                "<|end_message|>", "<|content_model_end_sampling|>"),
}
JSON_CALL = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'


class Tokens(ByteTokenizer):
    """A byte tokenizer with the family's control tokens (so the template's turns are single ids, as in the real pack) and its
    pre-tokenizer name, which is all the server reads of it to pick the call format."""

    def __init__(self, call_format: str, pre: str):
        self.pre = pre
        self.SPECIALS, self.special_ids = {
            "function": (["<|im_start|>", "<|im_end|>", "<|endoftext|>"], {}),
            "glm": (["<|endoftext|>", "<|user|>", "<|observation|>", "<|assistant|>", "<|system|>", "<|developer|>",
                     "<|end_of_text|>"], {"tokenizer.ggml.eos_token_id": 256}),
            "gemma": (["<bos>", "<|turn>", "<turn|>", "<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>",
                       "<|tool_response>", "<tool_response|>", Q], {"tokenizer.ggml.bos_token_id": 256,
                                                                    "tokenizer.ggml.eos_token_id": 258}),
            "dsml": (["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>"],
                     {"tokenizer.ggml.bos_token_id": 256, "tokenizer.ggml.eos_token_id": 257}),
            "lfm2": (["<|startoftext|>", "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|tool_call_start|>",
                      "<|tool_call_end|>"], {"tokenizer.ggml.bos_token_id": 256, "tokenizer.ggml.eos_token_id": 259}),
            "inkling": (["<|message_system|>", "<|message_user|>", "<|message_tool|>", "<|message_model|>",
                         "<|content_text|>", "<|content_thinking|>", "<|content_invoke_tool_json|>", "<|end_message|>",
                         "<|content_model_end_sampling|>"], {"tokenizer.ggml.eos_token_id": 264}),
        }[call_format]


def service(family: str, folder: Path) -> tuple[Service, Tokens]:
    row = FAMILIES[family]
    path = folder / f"{family}.jinja"
    path.write_text(row["chat_template"], encoding="utf-8")
    tok = Tokens(MATRIX[family][2], row["pre"])
    return Service(MockEngine(tok, "hi"), tok, ChatTemplate(path)), tok


def reply(svc: Service, ids: list[int], thinking: bool) -> str:
    """The model's reply to a prompt with a tool: a short thought where the model writes one, then the family's call."""
    if svc.parser_cls is JsonOutputParser and svc.call_format == "function":
        call, end = JSON_CALL, "<|im_end|>"
    else:
        call, end = CALLS[svc.call_format]
    if svc.call_format == "inkling":       # blocks: the text block a prompt opened is ended first, a thought is a block
        if svc.text_open_ids and ids[-len(svc.text_open_ids):] == svc.text_open_ids:
            return "<|end_message|>" + call + end
        think = "<|message_model|><|content_thinking|>the user wants the weather<|end_message|>" if thinking else ""
        return think + call + end
    if not thinking:
        return call + end
    opened = bool(svc.think_open_ids) and ids[-len(svc.think_open_ids):] == svc.think_open_ids
    think_open = {"glm": "<think>", "gemma": "<|channel>thought\n"}.get(svc.call_format, "")
    think_end = "<channel|>" if svc.call_format == "gemma" else "</think>"
    return ("" if opened else think_open) + "the user wants the weather" + think_end + call + end


class EveryFamily(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def test_the_fixture_has_every_strata_poly_family_of_setup(self):
        import sys
        sys.path.insert(0, str(HERE.parent))
        import setup
        poly = {f for f, d in setup.FAMILIES.items() if d.get("engine") == "poly"}
        self.assertEqual(set(FAMILIES), poly)
        self.assertEqual(set(MATRIX), poly)

    def test_each_family_gets_its_form_parser_and_call_format(self):
        for family, (form, cls, call_format) in MATRIX.items():
            with self.subTest(family):
                svc, _ = service(family, Path(self.dir.name))
                self.assertEqual(toolformat.tool_format(FAMILIES[family]["chat_template"]), form)
                if call_format != "inkling":      # (its parser is its pre-tokenizer's, not its template's)
                    self.assertIs(parser_for_template(FAMILIES[family]["chat_template"]), cls)
                self.assertIs(svc.parser_cls, cls)
                self.assertEqual(svc.call_format, call_format)

    def test_each_family_renders_a_tool_and_reads_its_call(self):
        for family in MATRIX:
            for effort in (None, "none"):
                with self.subTest(family, reasoning_effort=effort):
                    svc, tok = service(family, Path(self.dir.name))
                    kwargs = {"enable_thinking": False} if effort == "none" else {}
                    ids, thinking, max_new = svc.prepare([{"role": "user", "content": "Weather in Paris?"}],
                                                         [TOOL["function"]], kwargs, 256)
                    self.assertIn("get_weather", tok.decode(ids))           # the template wrote the tool into the prompt
                    svc.engine = MockEngine(tok, reply(svc, ids, thinking))
                    evs = [e for kind, e in svc.run(ids, thinking, [TOOL["function"]], max_new, {}, threading.Event())
                           if kind == "event"]
                    calls = [e.call for e in evs if e.kind == "tool_call"]
                    self.assertEqual([(c.name, c.arguments) for c in calls], [("get_weather", {"city": "Paris"})], evs)
                    self.assertEqual("".join(e.text for e in evs if e.kind == "content").strip(), "", evs)


class OverHttp(unittest.TestCase):
    """Flint and Kolibri, the two families whose every request failed: a chat request with a tool, streamed and not."""

    def post(self, family: str, stream: bool) -> tuple[int, bytes]:
        with tempfile.TemporaryDirectory() as d:
            svc, tok = service(family, Path(d))
            ids, thinking, _ = svc.prepare([{"role": "user", "content": "Weather in Paris?"}], [TOOL["function"]], {}, 256)
            svc.engine = MockEngine(tok, reply(svc, ids, thinking), max_context=8192)
            httpd = serve(svc, port=0)
            conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=30)
            try:
                conn.request("POST", "/v1/chat/completions", json.dumps({
                    "messages": [{"role": "user", "content": "Weather in Paris?"}], "tools": [TOOL], "max_tokens": 256,
                    "stream": stream}), {"Content-Type": "application/json"})
                r = conn.getresponse()
                return r.status, r.read()
            finally:
                conn.close()
                httpd.shutdown()
                httpd.server_close()

    def test_a_tool_call_comes_back(self):
        for family in ("flint", "kolibri"):
            with self.subTest(family):
                status, body = self.post(family, False)
                self.assertEqual(status, 200, body)
                choice = json.loads(body)["choices"][0]
                self.assertEqual(choice["finish_reason"], "tool_calls", body)
                (call,) = choice["message"]["tool_calls"]
                self.assertEqual((call["function"]["name"], json.loads(call["function"]["arguments"])),
                                 ("get_weather", {"city": "Paris"}))

    def test_a_streamed_tool_call_comes_back(self):
        for family in ("flint", "kolibri"):
            with self.subTest(family):
                status, body = self.post(family, True)
                self.assertEqual(status, 200, body)
                chunks = [json.loads(line[6:]) for line in body.decode("utf-8").splitlines()
                          if line.startswith("data: {")]
                self.assertFalse([c for c in chunks if "error" in c], body)
                deltas = [d for c in chunks for ch in c.get("choices", []) for d in (ch.get("delta") or {}).get("tool_calls") or []]
                self.assertEqual({d["function"]["name"] for d in deltas if d.get("function", {}).get("name")}, {"get_weather"})
                args = "".join(d.get("function", {}).get("arguments") or "" for d in deltas)
                self.assertEqual(json.loads(args), {"city": "Paris"})
                self.assertIn("tool_calls", [ch.get("finish_reason") for c in chunks for ch in c.get("choices", [])])


if __name__ == "__main__":
    unittest.main()
