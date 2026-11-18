"""Laguna behind the server: its pre-tokenizer, the GLM tool calls it writes, its stops, its one turn opener, and the
thinking its template opens in the prompt (or closes there when asked not to think).

    python -m unittest serve.test_laguna
"""
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import strata_tokenizer as ST  # noqa: E402
from serve.frontend import ChatTemplate, OutputParser  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine  # noqa: E402


class PreTokenizer(unittest.TestCase):
    def pieces(self, pre, text):
        return ST.Tokenizer(["a"], [], pre=pre)._pieces(text)

    def test_lines_are_cut_first(self):
        # llama.cpp's LLAMA_VOCAB_PRE_TYPE_LAGUNA: [^\n]+|[\n]+ first, then Qwen 2's pattern in each piece.  The
        # spaces before a newline stay with their line; Qwen 2's pattern alone takes them with the newline.
        self.assertEqual(self.pieces("laguna", "a  \nb"), ["a", "  ", "\n", "b"])
        self.assertEqual(self.pieces("kolibri1", "a  \nb"), ["a", "  \n", "b"])
        self.assertEqual(self.pieces("laguna", "x\n\n\n    return y"), ["x", "\n\n\n", "   ", " return", " y"])

    def test_digits_one_at_a_time_and_contractions_apart(self):
        self.assertEqual(self.pieces("laguna", "12345 it's"), ["1", "2", "3", "4", "5", " it", "'s"])


class ToolCalls(unittest.TestCase):
    SCHEMA = [{"name": "get_weather", "parameters": {"properties": {"city": {"type": "string"},
                                                                    "days": {"type": "integer"}}}}]
    # Laguna's template writes a call with no line breaks between its parts (GLM's has them)
    TEXT = ("the user wants a forecast</think>Let me look.<tool_call>get_weather<arg_key>city</arg_key><arg_value>"
            "Paris</arg_value><arg_key>days</arg_key><arg_value>3</arg_value></tool_call>")

    def test_a_call_without_line_breaks_is_parsed_however_it_arrives(self):
        for step in (1, 5, 10_000):
            with self.subTest(step=step):
                p = OutputParser(thinking=True, tools=self.SCHEMA, call_format="glm", opened=True)
                evs = [e for i in range(0, len(self.TEXT), step) for e in p.feed(self.TEXT[i:i + step])] + p.finish()
                calls = [e.call for e in evs if e.kind == "tool_call"]
                self.assertEqual([(c.name, c.arguments) for c in calls], [("get_weather", {"city": "Paris", "days": 3})])
                self.assertEqual("".join(e.text for e in evs if e.kind == "reasoning"), "the user wants a forecast")
                self.assertEqual("".join(e.text for e in evs if e.kind == "content"), "Let me look.")


class LagunaTokens(ByteTokenizer):
    # the ids as their types are in the GGUF: 〈|EOS|〉 (bos and eos) and </assistant> (eot) are control tokens;
    # <assistant>, <think>, </think>, <tool_call> and </tool_call> are user-defined ones, matched in any text;
    # <system>, <user> and <tool_response> are not tokens at all
    SPECIALS = ["〈|EOS|〉", "<assistant>", "</assistant>", "<think>", "</think>", "<tool_call>", "</tool_call>"]
    ALWAYS = ("<assistant>", "<think>", "</think>", "<tool_call>", "</tool_call>")
    pre = "laguna"
    special_ids = {"tokenizer.ggml.bos_token_id": 256, "tokenizer.ggml.eos_token_id": 256,
                   "tokenizer.ggml.eot_token_id": 258, "tokenizer.ggml.seperator_token_id": 300}


# Laguna's template, cut to what the server depends on: the literal BOS it opens with, {% generation %} around an
# assistant turn, and the generation prompt that opens the thinking, or closes it when asked not to think
TEMPLATE = ('{{- "〈|EOS|〉" -}}{%- set enable_thinking = enable_thinking | default(true) -%}'
            '{%- for m in messages -%}{%- if m.role == "user" -%}{{- "<user>" + m.content + "</user>\\n" -}}'
            '{%- elif m.role == "assistant" -%}{%- generation -%}{{- "<assistant><think></think>" + m.content + '
            '"</assistant>\\n" -}}{%- endgeneration -%}{%- endif -%}{%- endfor -%}'
            '{%- if add_generation_prompt -%}{{- "<assistant>" -}}{%- if enable_thinking -%}{{- "<think>" -}}'
            '{%- else -%}{{- "</think>" -}}{%- endif -%}{%- endif -%}')


class ServiceForLaguna(unittest.TestCase):
    def setUp(self):
        self.tok = LagunaTokens()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "chat_template.jinja"
            path.write_text(TEMPLATE, encoding="utf-8")
            self.svc = Service(MockEngine(self.tok, "hi"), self.tok, ChatTemplate(path))
        self.chat = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"},
                     {"role": "user", "content": "q2"}]

    def test_glm_calls_its_stops_and_its_one_turn_opener(self):
        self.assertEqual(self.svc.call_format, "glm")
        self.assertEqual(self.svc.stop_ids, {256, 258})          # llama.cpp's: 〈|EOS|〉 and </assistant>
        self.assertEqual(self.svc.turn_ids, frozenset({257}))    # <assistant>
        self.assertEqual(self.svc.think_open_ids, [259])

    def test_the_prompt_has_one_bos_and_opens_or_closes_the_thinking(self):
        # the template writes the BOS itself and the server adds none, as llama.cpp ends up with
        for kwargs, last in (({}, 259), ({"enable_thinking": False}, 260)):
            with self.subTest(**kwargs):
                ids = self.svc.encode_prompt(self.chat, None, kwargs)
                self.assertEqual(ids[0], 256)
                self.assertEqual(ids.count(256), 1)
                self.assertEqual(ids[-2:], [257, last])
                self.assertNotIn(300, ids)                       # no separator token

    def test_the_last_turn_is_where_the_reply_begins_and_the_next_request_shares_it(self):
        # strata-poly keeps a checkpoint at turn_at, which the chat's next request goes on from
        engine = StrataEngine("missing-executable", [], lazy=True)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "chat_template.jinja"
            path.write_text(TEMPLATE, encoding="utf-8")
            svc = Service(engine, self.tok, ChatTemplate(path))
        ids = svc.encode_prompt(self.chat, None, {})
        at = int(engine.turn_key(ids).split("=")[1])
        self.assertEqual(at, len(ids) - 2)                       # the generation prompt's <assistant>
        nxt = svc.encode_prompt(self.chat + [{"role": "assistant", "content": "Paris"},
                                             {"role": "user", "content": "and?"}], None, {})
        self.assertEqual(nxt[:at + 1], ids[:at + 1])

    def test_the_template_gets_the_tools_as_openai_sends_them(self):
        # Laguna's template prints each tool whole, and vLLM, SGLang and llama.cpp hand it the request's tools as
        # they are: {"type": "function", "function": {...}}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "chat_template.jinja"
            path.write_text("{%- for t in tools -%}{{- t | tojson -}}{%- endfor -%}<assistant><think>", encoding="utf-8")
            svc = Service(MockEngine(self.tok, "hi"), self.tok, ChatTemplate(path))
        text = svc.render_prompt([{"role": "user", "content": "q"}], [{"name": "f", "parameters": {}}], {})
        self.assertEqual(text, '{"type": "function", "function": {"name": "f", "parameters": {}}}<assistant><think>')

    def test_the_reply_goes_on_inside_the_thinking_the_prompt_opened(self):
        for kwargs, reply, reasoning in (({}, "say Paris</think>Paris</assistant>", "say Paris"),
                                         ({"enable_thinking": False}, "Paris</assistant>", "")):
            with self.subTest(**kwargs):
                ids = self.svc.encode_prompt(self.chat, None, kwargs)
                self.svc.engine = MockEngine(self.tok, reply)
                evs = [e for kind, e in self.svc.run(ids, not kwargs, None, 64, {}, threading.Event())
                       if kind == "event"]
                self.assertEqual("".join(e.text for e in evs if e.kind == "reasoning"), reasoning)
                self.assertEqual("".join(e.text for e in evs if e.kind == "content"), "Paris")


if __name__ == "__main__":
    unittest.main()
