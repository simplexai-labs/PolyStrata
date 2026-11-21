"""Inkling 2 Mini behind the server: its pre-tokenizer, its chat template, its blocks of reasoning, text and tool calls,
and its stops, effort and forced calls.

    python -m unittest serve.test_inkling

The template is the model's own (serve/fixtures/inkling2_chat_template.jinja, the GGUF's tokenizer.chat_template and
thinkingmachines/Inkling-2-Mini's chat_template.jinja, SHA-256 53a9035f...).  serve/fixtures/inkling2_golden.json
holds the token ids of 51 strings (53 cases) from Hugging Face's tokenizers on the model's tokenizer.json (the reference;
llama.cpp's where they differ are kept as "llama_cpp_ids", and the cases with control tokens read as text are
llama.cpp's), checked with the part of the vocabulary they use (serve/fixtures/inkling2_vocab_subset.json) and again
with the whole vocabulary from $STRATA_INKLING_TOKENIZER (a pack's tokenizer/ directory) when it is set; and llama.cpp
29b52d398b's rendered prompts and its parse of replies.
"""
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import strata_tokenizer as ST  # noqa: E402
from serve.frontend import (ChatTemplate, InklingParser, anthropic_to_messages, forced_call,  # noqa: E402
                            openai_to_messages, parse_inkling_tool_call)
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TEMPLATE = FIXTURES / "inkling2_chat_template.jinja"
GOLDEN = FIXTURES / "inkling2_golden.json"


class PreTokenizer(unittest.TestCase):
    def pieces(self, text):
        return ST.Tokenizer(["a"], [], pre="inkling2")._pieces(text)

    def test_words_are_cut_where_their_capitals_end(self):
        self.assertEqual(self.pieces("HTTPServer and McDonald's"), ["HTTPServer", " and", " Mc", "Donald's"])
        self.assertEqual(self.pieces("HELLO hello HeLLo"), ["HELLO", " hello", " He", "LLo"])

    def test_contractions_stay_on_the_word_in_either_case(self):
        self.assertEqual(self.pieces("I'm sure they'LL"), ["I'm", " sure", " they'LL"])

    def test_digits_go_in_threes_and_punctuation_keeps_line_breaks_and_slashes(self):
        self.assertEqual(self.pieces("1234567"), ["123", "456", "7"])
        self.assertEqual(self.pieces("a.\n\nb"), ["a", ".\n\n", "b"])
        self.assertEqual(self.pieces("x =//\ny"), ["x", " =//\n", "y"])
        self.assertEqual(self.pieces("end.)/\n/next"), ["end", ".)/\n/", "next"])     # Flint's: ".)/\n", "/next"
        self.assertEqual(self.pieces("a   b\n\n  c"), ["a", "  ", " b", "\n\n", " ", " c"])

    def test_a_letter_past_ascii_is_both_cases(self):
        # llama.cpp matches the pattern on text where every such letter is one placeholder: "helloПривет" is one piece
        self.assertEqual(self.pieces("helloПривет мир"), ["helloПривет", " мир"])
        self.assertEqual(self.pieces("naïve Café"), ["naïve", " Café"])
        self.assertEqual(self.pieces("abc中文def"), ["abc中文def"])             # no run of ideographs of its own

    def test_a_piece_that_is_a_token_is_that_token(self):
        # tokenizer.json's ignore_merges (Hugging Face is the reference; llama.cpp 29b52d398b does not set it)
        tok = ST.Tokenizer(["a", "b", "ab"], [], pre="inkling2")
        self.assertTrue(tok.ignore_merges)
        self.assertEqual(tok.encode("ab"), [2])        # "ab" is a token, though no merge makes it


def subset_tokenizer():
    """The vocabulary's tokens and merges that the golden strings use (serve/fixtures/inkling2_vocab_subset.json), at
    their own ids; the other ids hold placeholders."""
    sub = json.loads((FIXTURES / "inkling2_vocab_subset.json").read_text(encoding="utf-8"))
    tokens = [f"<placeholder {i}>" for i in range(sub["n_tokens"])]
    types = [1] * sub["n_tokens"]
    for i, t in sub["tokens"].items():
        tokens[int(i)] = t
    for i in sub["control"]:
        types[i] = 3
    return ST.Tokenizer(tokens, sub["merges"], types, pre=sub["pre"], special_ids=sub["special_ids"])


class ReferenceIds(unittest.TestCase):
    """The reference ids of the golden strings, with and without parse_special (no download needed)."""

    def test_the_golden_strings(self):
        tok = subset_tokenizer()
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        self.assertGreater(len(golden["tokens"]), 40)
        for case in golden["tokens"]:
            with self.subTest(case["name"]):
                ids = tok.encode(case["text"], parse_special=case["parse_special"])
                self.assertEqual(ids, case["ids"])
                self.assertEqual(tok.decode(ids), case["text"])

    def test_a_prompt_is_cut_at_the_control_tokens(self):
        tok = subset_tokenizer()
        ids = tok.encode("<|message_user|><|content_text|>Hi there<|end_message|><|message_model|>", parse_special=True)
        self.assertEqual((ids[:2], ids[-2:]), ([200000, 200004], [200010, 200001]))
        self.assertNotIn(200010, tok.encode("<|end_message|>"))     # text unless parse_special

    def test_where_llama_cpp_differs(self):
        # "kwargs" is one token that no chain of merges reaches: Hugging Face gives it whole, llama.cpp as three
        tok = subset_tokenizer()
        case = next(c for c in json.loads(GOLDEN.read_text(encoding="utf-8"))["tokens"] if "llama_cpp_ids" in c)
        self.assertEqual(case["text"], "kwargs **kwargs")
        self.assertEqual(tok.encode(case["text"]), [87550, 197033, 87550])
        self.assertEqual(case["llama_cpp_ids"], [86226, 70, 82, 197033, 86226, 70, 82])


def real_tokenizer():
    path = os.environ.get("STRATA_INKLING_TOKENIZER")
    return ST.Tokenizer.from_dir(path) if path and (Path(path) / "vocab.json").is_file() else None


@unittest.skipIf(real_tokenizer() is None, "no Inkling tokenizer here ($STRATA_INKLING_TOKENIZER)")
class RealVocabulary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = real_tokenizer()

    def test_the_vocabulary_and_its_specials(self):
        self.assertEqual(len(self.tok.tokens), 200058)
        self.assertEqual(self.tok.pre, "inkling2")
        self.assertEqual(self.tok.special_ids["tokenizer.ggml.eos_token_id"], 200006)
        self.assertEqual(self.tok.special_ids["tokenizer.ggml.padding_token_id"], 199999)
        self.assertNotIn("tokenizer.ggml.bos_token_id", self.tok.special_ids)
        for text, i in (("<|message_model|>", 200001), ("<|content_model_end_sampling|>", 200006),
                        ("<|end_message|>", 200010), ("<|content_invoke_tool_json|>", 200049)):
            self.assertEqual(self.tok.encode(text, parse_special=True), [i])
            self.assertNotEqual(self.tok.encode(text), [i])      # a control token: text unless parse_special

    def test_round_trips(self):
        for text in ("", "Hello, world!", "  leading and trailing  ", "def f(x):\n\treturn x\n", "你好，世界",
                     "مرحبا", "\U0001f600\U0001f680\U0001f1fa\U0001f1f8", "é́", "\x00\x01\x7f",
                     " nbsp", "1234567890", "\r\n\r\n", "x" * 3000, "<|message_user|> as text"):
            with self.subTest(text=text[:20]):
                self.assertEqual(self.tok.decode(self.tok.encode(text)), text)

    def test_reference_token_ids(self):
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        for case in golden["tokens"]:
            with self.subTest(case["name"]):
                self.assertEqual(self.tok.encode(case["text"], parse_special=case.get("parse_special", True)),
                                 case["ids"])


class Template(unittest.TestCase):
    TOOLS = [{"name": "get_weather", "description": "Get the weather",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                             "required": ["city"]}}]

    @classmethod
    def setUpClass(cls):
        cls.t = ChatTemplate(TEMPLATE)

    def test_a_user_message_and_the_default_effort(self):
        self.assertEqual(self.t.render([{"role": "user", "content": "Hi"}]),
                         "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
                         "<|message_user|><|content_text|>Hi<|end_message|>")

    def test_the_effort_comes_after_the_system_prompt_and_reads_names_and_numbers(self):
        msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]
        for effort, shown in (("none", "0"), ("minimal", "0.1"), ("low", "0.2"), ("medium", "0.7"), ("high", "0.9"),
                              ("xhigh", "0.99"), ("max", "0.99"), (0.5, "0.5")):
            with self.subTest(effort=effort):
                self.assertEqual(self.t.render(msgs, reasoning_effort=effort),
                                 "<|message_system|><|content_text|>Be brief.<|end_message|>"
                                 f"<|message_system|><|content_text|>Thinking effort level: {shown}<|end_message|>"
                                 "<|message_user|><|content_text|>Hi<|end_message|>")

    def test_tools_calls_and_answers(self):
        msgs = [{"role": "user", "content": "Weather in Paris?"},
                {"role": "assistant", "content": "", "reasoning_content": "I need the tool.",
                 "tool_calls": [{"id": "c1", "function": {"name": "get_weather", "arguments": {"days": 2, "city": "Paris"}}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "18 C"},
                {"role": "assistant", "content": "It is 18 C."}]
        self.assertEqual(self.t.render(msgs, tools=self.TOOLS),
                         '<|message_system|>tool_declare<|content_xml|>[{"description":"Get the weather",'
                         '"name":"get_weather","parameters":{"properties":{"city":{"type":"string"},'
                         '"days":{"type":"integer"}},"required":["city"],"type":"object"},"type":"function"}]'
                         "<|end_message|>"
                         "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
                         "<|message_user|><|content_text|>Weather in Paris?<|end_message|>"
                         "<|message_model|><|content_thinking|>I need the tool.<|end_message|>"
                         '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather",'
                         '"args":{"city":"Paris","days":2}}<|end_message|><|content_model_end_sampling|>'
                         "<|message_tool|>get_weather<|content_text|>18 C<|end_message|>"
                         "<|message_model|><|content_text|>It is 18 C.<|end_message|><|content_model_end_sampling|>")

    def test_a_tool_in_openai_shape_renders_the_same(self):
        flat = self.t.render([{"role": "user", "content": "x"}], tools=self.TOOLS)
        wrapped = self.t.render([{"role": "user", "content": "x"}],
                                tools=[{"type": "function", "function": self.TOOLS[0]}])
        self.assertEqual(flat, wrapped)

    def test_the_server_reads_what_the_template_can_do(self):
        self.assertEqual(self.t.caps, {"supports_tools": True, "supports_tool_calls": True,
                                       "supports_system_role": True, "supports_parallel_tool_calls": True,
                                       "supports_preserve_reasoning": True})

    def test_llama_cpp_renders_the_same(self):
        # llama.cpp's prompts for OpenAI requests, read the way the server reads them
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        self.assertGreater(len(golden["templates"]), 10)
        for case in golden["templates"]:
            with self.subTest(case["name"]):
                messages, tools, _ = openai_to_messages(case["request"])
                self.assertEqual(self.t.render(messages, tools=tools, **case["kwargs"]), case["prompt"])


H, THINK, TEXT, JSON_, END = ("<|message_model|>", "<|content_thinking|>", "<|content_text|>",
                              "<|content_invoke_tool_json|>", "<|end_message|>")


def events(text, step, tools=None, stream_tools=True, thinking=True):
    p = InklingParser(thinking=thinking, tools=tools, stream_tools=stream_tools)
    evs = []
    for i in range(0, len(text), step):
        evs += p.feed(text[i:i + step])
    return evs + p.finish()


def joined(evs, kind):
    return "".join(e.text for e in evs if e.kind == kind)


class Blocks(unittest.TestCase):
    TOOLS = [{"name": "get_weather", "parameters": {"properties": {"city": {"type": "string"}}}},
             {"name": "now", "parameters": {"properties": {}}}]
    STEPS = (1, 3, 7, 10_000)

    def test_reasoning_then_the_answer(self):
        text = f"{H}{THINK}The user greets me.{END}{H}{TEXT}Hello! <b>x</b>{END}"
        for step in self.STEPS:
            with self.subTest(step=step):
                evs = events(text, step)
                self.assertEqual(joined(evs, "reasoning"), "The user greets me.")
                self.assertEqual(joined(evs, "content"), "Hello! <b>x</b>")

    def test_an_answer_without_reasoning_and_blocks_one_after_another(self):
        text = f"{H}{TEXT}One.{END}{H}{THINK}more{END}{H}{TEXT} Two.{END}"
        for step in self.STEPS:
            with self.subTest(step=step):
                evs = events(text, step)
                self.assertEqual(joined(evs, "content"), "One. Two.")
                self.assertEqual(joined(evs, "reasoning"), "more")

    def test_thinking_off_still_reads_a_thinking_block_as_reasoning(self):
        evs = events(f"{H}{THINK}brief{END}{H}{TEXT}ok{END}", 2, thinking=False)
        self.assertEqual((joined(evs, "reasoning"), joined(evs, "content")), ("brief", "ok"))

    def test_tool_calls_however_they_arrive(self):
        text = (f"{H}{THINK}Need weather.{END}{H}get_weather{JSON_}"
                '{"name":"get_weather","args":{"city":"Paris <|x|>"}}' f"{END}{H}now{JSON_}" '{"name":"now","args":{}}'
                f"{END}")
        for stream_tools in (False, True):
            for step in self.STEPS:
                with self.subTest(stream_tools=stream_tools, step=step):
                    evs = events(text, step, self.TOOLS, stream_tools)
                    calls = [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]
                    self.assertEqual(calls, [("get_weather", {"city": "Paris <|x|>"}), ("now", {})])
                    self.assertEqual(joined(evs, "reasoning"), "Need weather.")
                    self.assertEqual(joined(evs, "content"), "")

    def test_a_streamed_call_is_announced_and_keeps_its_id(self):
        text = f"{H}get_weather{JSON_}" '{"name":"get_weather","args":{"city":"Oslo"}}' f"{END}"
        evs = [e for e in events(text, 1, self.TOOLS) if e.kind.startswith("tool_")]
        self.assertEqual([e.kind for e in evs], ["tool_start", "tool_args", "tool_call"])
        self.assertEqual(evs[0].call.id, evs[2].call.id)
        self.assertEqual(json.loads(evs[1].text), {"city": "Oslo"})

    def test_a_call_to_a_tool_not_declared_is_text(self):
        text = f"{H}rm_rf{JSON_}" '{"name":"rm_rf","args":{}}' f"{END}"
        for tools in (None, self.TOOLS):
            with self.subTest(tools=bool(tools)):
                evs = events(text, 4, tools)
                self.assertFalse([e for e in evs if e.kind.startswith("tool_")])
                self.assertIn("rm_rf", joined(evs, "content"))

    def test_a_call_whose_end_did_not_come(self):
        whole = f"{H}now{JSON_}" '{"name":"now","args":{}}'
        evs = events(whole, 5, self.TOOLS)                       # its JSON is whole: the call
        self.assertEqual([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"], [("now", {})])
        cut = f"{H}get_weather{JSON_}" '{"name":"get_weather","args":{"ci'
        evs = events(cut, 5, self.TOOLS)                         # announced and cut: no call (as OutputParser, #211)
        self.assertEqual([e.kind for e in evs if e.kind.startswith("tool_")], ["tool_start"])
        evs = events(cut, 5, self.TOOLS, stream_tools=False)     # not announced: the text it is
        self.assertFalse([e for e in evs if e.kind.startswith("tool_")])
        self.assertIn('"args":{"ci', joined(evs, "content"))

    def test_stray_markers_are_not_shown(self):
        evs = events(f"{H}{TEXT}Hi{END}<|content_model_end_sampling|>", 1)
        self.assertEqual(joined(evs, "content"), "Hi")
        evs = events(f"{H}{TEXT}Hi", 1)                          # cut before its end: what came
        self.assertEqual(joined(evs, "content"), "Hi")

    def test_a_block_without_its_content_token_is_text(self):
        evs = events(f"{H}Plain answer.\nSecond line{END}", 3)
        self.assertEqual(joined(evs, "content"), "Plain answer.\nSecond line")

    def test_the_parser_states_the_server_reads(self):
        p = InklingParser(tools=self.TOOLS)
        self.assertEqual((p.state, p.buf), ("content", ""))
        p.feed(H)
        self.assertNotEqual(p.buf, "")                           # the header is held: no forced call written here
        p.feed(THINK + "x")
        self.assertEqual(p.state, "reasoning")
        p.feed(END)
        self.assertEqual((p.state, p.buf), ("content", ""))     # between blocks: a forced call goes here
        p.feed(H + TEXT + "y")
        self.assertEqual(p.state, "text")                        # not inside a text block
        self.assertEqual(p.think_end, END)

    def test_llama_cpp_reads_the_same(self):
        # llama.cpp's parse of the replies in its own test-chat.cpp cases for Inkling 2, and a few more
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        self.assertGreater(len(golden["replies"]), 10)
        for case in golden["replies"]:
            for step in (1, 5, 10_000):
                with self.subTest(case["name"], step=step):
                    evs = events(case["text"], step, case["tools"] or None)
                    self.assertEqual(joined(evs, "content"), case["content"])
                    self.assertEqual(joined(evs, "reasoning"), case["reasoning"])
                    self.assertEqual([{"name": e.call.name, "arguments": e.call.arguments}
                                      for e in evs if e.kind == "tool_call"], case["calls"])

    def test_the_block_closer_inside_a_string_argument_does_not_end_the_call(self):
        text = f"{H}python{JSON_}" '{"name":"python","args":{"code":"print(\'<|end_message|>\')"}}' f"{END}{H}{TEXT}ok{END}"
        tools = [{"name": "python", "parameters": {"properties": {"code": {"type": "string"}}}}]
        for step in (1, 4, 10_000):
            with self.subTest(step=step):
                evs = events(text, step, tools)
                self.assertEqual([e.call.arguments for e in evs if e.kind == "tool_call"],
                                 [{"code": "print('<|end_message|>')"}])
                self.assertEqual(joined(evs, "content"), "ok")
        # JSON that never closes ends where the next block opens
        evs = events(f'{H}python{JSON_}{{"name":"python","args":{{"code":"x{END}{H}{TEXT}after{END}', 3, tools)
        self.assertEqual(joined(evs, "content").count("after"), 1)
        self.assertFalse([e for e in evs if e.kind == "tool_call"])

    def test_parse_inkling_tool_call(self):
        self.assertEqual(parse_inkling_tool_call("f", '{"name":"f","args":{"a":1}}').arguments, {"a": 1})
        self.assertEqual(parse_inkling_tool_call("f", '{"name": "f", "arguments": "{\\"a\\": 1}"}').arguments, {"a": 1})
        self.assertEqual(parse_inkling_tool_call("f", '{"a": 1}').arguments, {"a": 1})
        self.assertEqual(parse_inkling_tool_call("f", '{"name":"f"}').arguments, {})        # no arguments
        self.assertEqual(parse_inkling_tool_call("f", '{"name":"f","a":1}').arguments, {"a": 1})
        self.assertEqual(parse_inkling_tool_call("f", '{"name":"x"}').arguments, {"name": "x"})  # a "name" argument
        for body in ('{"name":"f","args":[1]}', "{", "[]"):
            self.assertIsNone(parse_inkling_tool_call("f", body))


class InklingTokens(ByteTokenizer):
    SPECIALS = ["<|endoftext|>", "<|message_user|>", "<|message_model|>", "<|message_system|>", "<|message_tool|>",
                "<|content_text|>", "<|content_image|>", "<|content_model_end_sampling|>", "<|content_thinking|>",
                "<|end_message|>", "<|content_xml|>", "<|content_invoke_tool_json|>"]
    pre = "inkling2"
    special_ids = {"tokenizer.ggml.eos_token_id": 263, "tokenizer.ggml.eot_token_id": 263,
                   "tokenizer.ggml.padding_token_id": 256}


class ServiceForInkling(unittest.TestCase):
    TOOLS = [{"type": "function", "function": {"name": "get_weather",
                                               "parameters": {"properties": {"city": {"type": "string"}}}}}]

    def setUp(self):
        self.tok = InklingTokens()
        self.svc = Service(MockEngine(self.tok, "hi"), self.tok, ChatTemplate(TEMPLATE))

    def prompt(self, req):
        return self.svc.render_prompt(*openai_to_messages(req))

    def test_call_format_stops_and_turns(self):
        self.assertEqual(self.svc.call_format, "inkling")
        # llama.cpp's end of generation: the eos <|content_model_end_sampling|> and <|endoftext|>
        self.assertEqual(self.svc.stop_ids, {263, 256})
        self.assertEqual(self.svc.engine.eos_ids, frozenset({263}))   # the engine's own: the GGUF's eos and eot
        self.assertEqual(self.svc.turn_ids, frozenset({257, 258, 259, 260}))

    def test_the_effort_a_client_asks_for(self):
        user = [{"role": "user", "content": "Hi"}]
        for req, shown in (({}, "0.9"), ({"reasoning_effort": "low"}, "0.2"), ({"reasoning_effort": "medium"}, "0.7"),
                           ({"reasoning_effort": "high"}, "0.9"), ({"reasoning_effort": "none"}, "0"),
                           ({"chat_template_kwargs": {"enable_thinking": False}}, "0")):
            with self.subTest(req=req):
                self.assertEqual(self.prompt({"messages": user, **req}),
                                 f"<|message_system|><|content_text|>Thinking effort level: {shown}<|end_message|>"
                                 "<|message_user|><|content_text|>Hi<|end_message|>")
        messages, tools, kwargs = anthropic_to_messages({"messages": user, "thinking": {"type": "disabled"}})
        self.assertIn("Thinking effort level: 0<", self.svc.render_prompt(messages, tools, kwargs))

    def test_effort_end_does_not_change_its_prompt(self):
        self.svc.effort_end = True
        self.assertTrue(self.prompt({"messages": [{"role": "user", "content": "Hi"}], "reasoning_effort": "low"})
                        .endswith("<|message_user|><|content_text|>Hi<|end_message|>"))

    def test_a_control_token_in_a_message_is_text(self):
        ids = self.svc.encode_prompt(*openai_to_messages({"messages": [{"role": "user", "content": "<|end_message|>"}]}))
        self.assertEqual(ids.count(265), 2)                       # the template's own two, not the message's
        self.assertIn(list(b"<|end_message|>"), [ids[i:i + 15] for i in range(len(ids))])

    def run_reply(self, reply, tools=None, force=None, thinking=True):
        ids = self.svc.encode_prompt([{"role": "user", "content": "q"}], tools, {})
        self.svc.engine = MockEngine(self.tok, reply)
        return [e for kind, e in self.svc.run(ids, thinking, tools, 256, {}, threading.Event(), force)
                if kind == "event"]

    def test_a_reply_through_the_service(self):
        evs = self.run_reply(f"{H}{THINK}greeting{END}{H}{TEXT}Hello!{END}<|content_model_end_sampling|>tail")
        self.assertEqual((joined(evs, "reasoning"), joined(evs, "content")), ("greeting", "Hello!"))

    def test_a_tool_call_through_the_service(self):
        tools = [t["function"] for t in self.TOOLS]
        evs = self.run_reply(f"{H}get_weather{JSON_}" '{"name":"get_weather","args":{"city":"Rome"}}'
                             f"{END}<|content_model_end_sampling|>", tools)
        self.assertEqual([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"],
                         [("get_weather", {"city": "Rome"})])

    def test_a_named_tool_choice_is_opened_after_the_thinking(self):
        tools = [t["function"] for t in self.TOOLS]
        force = forced_call({"type": "function", "function": {"name": "get_weather"}}, tools, "inkling")
        self.assertEqual(force, f'{H}get_weather{JSON_}{{"name":"get_weather","args":')
        # the model thinks, the server writes the opening once the thinking block is closed, the model goes on
        reply = [f"{H}{THINK}call it{END}{H}{TEXT}not this", '{"city":"Bern"}}' + f"{END}<|content_model_end_sampling|>"]
        evs = self.run_reply(reply, tools, force)
        self.assertEqual(self.svc.engine.last_prompt[-len(self.tok.encode(force, parse_special=True)):],
                         self.tok.encode(force, parse_special=True))      # right after the closed block
        self.assertEqual(joined(evs, "reasoning"), "call it")
        self.assertEqual([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"],
                         [("get_weather", {"city": "Bern"})])

    def test_the_reply_ends_after_a_forced_call(self):
        # without thinking the prompt ends with the opening; the model finishes the call, then writes blocks of its
        # own (a thinking block, an answer): the reply ends where the first of them opens
        tools = [t["function"] for t in self.TOOLS]
        force = forced_call({"type": "function", "function": {"name": "get_weather"}}, tools, "inkling")
        evs = self.run_reply('{"city":"Bern"}}' f"{END}{H}{THINK}why{END}{H}{TEXT}a joke{END}", tools, force,
                             thinking=False)
        self.assertEqual([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"],
                         [("get_weather", {"city": "Bern"})])
        self.assertEqual((joined(evs, "reasoning"), joined(evs, "content")), ("", ""))

    def test_required_among_several_tools_is_left_to_the_model(self):
        tools = [{"name": "a"}, {"name": "b"}]
        self.assertIsNone(forced_call("required", tools, "inkling"))
        self.assertTrue(forced_call("required", [{"name": "a"}], "inkling").startswith(H + "a" + JSON_))
        self.assertEqual(forced_call("required", tools), "<tool_call>\n<function=")   # the others' as before


class Pictures(unittest.TestCase):
    """A picture in a user turn: the template writes <|content_image|><|unused_200054|> in a block of its own, and
    the server puts one <|unused_200054|> a row of the picture there, as llama.cpp 29b52d398b's mtmd feeds it
    (<|message_user|><|content_image|>[rows]<|end_message|>; its img_beg is <|content_image|>, it writes no end)."""

    class FakeVision:
        def __init__(self, d, n=(3, 2)):
            self.n, self.seen = list(n), {}
            self.dir = Path(d)
            self.rows = self.dir / "img.sve"
            self.rows.write_bytes(b"rows")

        def encode(self, source):     # a picture encoded once is the cache's (a tool's is encoded before)
            if source not in self.seen:
                self.seen[source] = self.n.pop(0)
            return self.rows, self.seen[source]

    def service(self, d, n=(3, 2)):
        class Asked(MockEngine):
            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                self.asked = (StrataEngine.sampling_keys(sampling), embeddings)
                yield from super().generate(ids, max_new, sampling, cancel, embeddings)

        tok = subset_tokenizer()
        return Service(Asked(tok, "a cat"), tok, ChatTemplate(TEMPLATE), vision=self.FakeVision(d, n))

    def test_the_rows_go_where_the_template_put_the_picture(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self.service(d)
            self.assertEqual(svc.image_marks, ("<|unused_200054|>", "<|content_image|>", None))
            msgs = [{"role": "user", "content": [{"type": "image", "source": "a.png"},
                                                 {"type": "text", "text": "Hi there"},
                                                 {"type": "image", "source": "b.png"}]}]
            ids, _, _ = svc.prepare(msgs, None, {})
            at = ids.index(200005)
            self.assertEqual(ids[at - 1:at + 7], [200000, 200005, 200054, 200054, 200054, 200010, 200000, 200004])
            at = ids.index(200005, at + 1)
            self.assertEqual(ids[at - 1:at + 4], [200000, 200005, 200054, 200054, 200010])
            self.assertEqual((ids.count(200054), svc.embeddings.pad), (5, 200054))
            # the engine is told which token stands for a picture's rows
            rows = svc.embeddings.path
            list(svc.run(ids, False, None, 8, {}, threading.Event()))
            self.assertEqual(svc.engine.asked, (" image_pad=200054", rows))

    def test_two_pictures_back_to_back(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self.service(d, (2, 3))
            msgs = [{"role": "user", "content": [{"type": "image", "source": "a.png"},
                                                 {"type": "image", "source": "b.png"}]}]
            ids, _, _ = svc.prepare(msgs, None, {})
            at = ids.index(200005)
            self.assertEqual(ids[at - 1:at + 10], [200000, 200005, 200054, 200054, 200010,
                                                   200000, 200005, 200054, 200054, 200054, 200010])

    def test_a_picture_in_a_tool_message(self):
        # the tool's name follows <|message_tool|> in its header; the rows still follow <|content_image|>
        with tempfile.TemporaryDirectory() as d:
            svc = self.service(d, (3,))
            msgs = [{"role": "user", "content": "Hi there"},
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "shot", "arguments": {}}}]},
                    {"role": "tool", "tool_call_id": "c1",
                     "content": [{"type": "image", "source": "a.png"}, {"type": "text", "text": "Hi there"}]}]
            ids, _, _ = svc.prepare(msgs, None, {})
            tool = svc.tok.encode("<|message_tool|>shot", parse_special=True)
            at = next(i for i in range(len(ids)) if ids[i:i + len(tool)] == tool) + len(tool)
            self.assertEqual(ids[at:at + 6], [200005, 200054, 200054, 200054, 200010] + tool[:1])
            self.assertEqual((ids.count(200054), svc.embeddings.pad), (3, 200054))

    def test_the_marks_in_a_message_are_text(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self.service(d, (2,))
            msgs = [{"role": "user", "content": [{"type": "text", "text": "<|content_image|><|unused_200054|>"},
                                                 {"type": "image", "source": "a.png"}]}]
            ids, _, _ = svc.prepare(msgs, None, {})
            self.assertEqual((ids.count(200005), ids.count(200054)), (1, 2))
            self.assertIn(svc.tok.encode("<|unused_200054|>"), [ids[i:i + 15] for i in range(len(ids))])

    def test_a_picture_from_the_model_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self.service(d, (2,))
            msgs = [{"role": "user", "content": "Hi there"},
                    {"role": "assistant", "content": [{"type": "image", "source": "a.png"}]},
                    {"role": "user", "content": "Hi there"}]
            with self.assertRaises(Exception) as e:
                svc.prepare(msgs, None, {})
            self.assertIn("only supported for user and tool messages", str(e.exception))


if __name__ == "__main__":
    unittest.main()
