# Inkling 2 Mini

Back to [README.md](../../README.md)

**Inkling 2 Mini** by Thinking Machines, released on 2026-11-20: a mixture of 128 experts, 8 used a token, in 24
layers. Layers 6 to 17 run three times for each token, so a token goes through 48 layers with the weights of 24. Five
layers in six look back only 1,024 tokens. PolyStrata runs [ggml-org's GGUF files](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF)
and [Unsloth's UD-Q2_K_XL](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) on the strata-poly engine.

**It is experimental and new.** Support was added on 2026-11-21. It has run on one PC so far (below). Its logits were
checked against llama.cpp's on the same files (below): the same most likely next token after each of the prompts, and
the same at 29 to 32 of 32 steps with llama.cpp's own answer fed in.

## How fast is it?

> **Measured on:** RTX 2070 Super 8 GB, Ryzen 5 3600XT, 62 GB DDR4, Ubuntu 22.04, one request, warm file cache, the
> server's flags (128K context, q8_0 caches, the model's expert profile), 128 tokens written, two runs each, the
> 302- and 1,500-token rows on the same day.

| | IQ4_NL prompt | IQ4_NL writing | UD-Q2_K_XL prompt | UD-Q2_K_XL writing |
| --- | ---: | ---: | ---: | ---: |
| 302-token prompt | 273-275 tokens/s | 37.3-38.1 tokens/s | 503-511 tokens/s | 50.7-53.0 tokens/s |
| 1,500-token prompt | 452-455 tokens/s | 36.1-36.5 tokens/s | 586-611 tokens/s | 46.7-48.1 tokens/s |
| 34,800-token prompt (IQ4_NL, through the server, an earlier day) | 232 tokens/s | 28.1 tokens/s | | |

Since these rows the engine takes a visit's projections in one launch and its cached experts a warp a row. In
back-to-back runs of strata-poly alone (the same flags, 2026-11-22, two runs each) writing after the 302-token prompt
went from 38.2-38.7 to 39.0-39.5 tokens/s with IQ4_NL (+2%, with a copy of its expert profile) and from 51.2-51.9 to
54.6-55.2 with UD-Q2_K_XL (+6%, without one).

A prompt batch's attention runs on the tensor cores (cards of Turing or newer). Next to the build before it
(2026-11-22, the same card; `strata-poly --serve`, a 128K context, q8_0 caches, one engine a run, each from a fresh
copy of the model's expert profile, the prompt read in a fresh conversation; text from the parity prompts repeated;
the 302- and 1,500-token rows two runs each, old and new taken in turn, 256 tokens written; the longer ones one run,
64 tokens written):

| run | prompt | before | writing | before |
| --- | ---: | ---: | ---: | ---: |
| 302 tokens, IQ4_NL | 282-290 tokens/s | 271-272 tokens/s | 38.8-40.0 tokens/s | 39.3-40.6 tokens/s |
| 1,500 tokens, IQ4_NL | 521-523 tokens/s | 458-461 tokens/s | 38.5-38.7 tokens/s | 39.0-39.2 tokens/s |
| 302 tokens, UD-Q2_K_XL | 557-561 tokens/s | 504-508 tokens/s | 51.1-52.3 tokens/s | 52.3-52.7 tokens/s |
| 1,500 tokens, UD-Q2_K_XL | 881-888 tokens/s | 609-611 tokens/s | 49.0-49.5 tokens/s | 48.6-48.8 tokens/s |
| 2,048 tokens, IQ4_NL | 658 tokens/s | 504 tokens/s | 34.0 tokens/s | 34.1 tokens/s |
| 32,768 tokens, IQ4_NL | 820 tokens/s | 263 tokens/s | 28.7 tokens/s | 28.5 tokens/s |
| 131,008 tokens, IQ4_NL | 659 tokens/s (199 s) | 92 tokens/s (1,421 s) | 20.6 tokens/s | 20.8 tokens/s |

Writing does not run on the tensor cores (a token at a time takes the split path); single runs differ by up to
4.5% both ways, the means of two runs by at most 1.5%.

IQ4_NL's speed depends on its expert profile (`--expert-profile`, which the server keeps up to date): it chooses the
experts the card holds. On 100 questions (50 GSM8K, 50 MMLU-Pro, greedy, medium reasoning) IQ4_NL got 49 and 44
right, UD-Q2_K_XL 48 and 41; the two gave the same answer to 92 of the 100.

About 4.6 GiB of experts fit on the 8 GB card. At most 7.5 GiB of VRAM was in use (nvidia-smi, IQ4_NL, a
31,600-token prompt). The engine's RAM was 16.5 GiB with IQ4_NL.

## What you need

| | |
| --- | --- |
| **Graphics card** | One **NVIDIA** card. About 3.1 GiB of VRAM before any expert at an 8K context (measured on an 8 GB card), about 4 GiB at 32K. |
| **RAM** | About **22 GB** (IQ4_NL), **14 GB** (UD-Q2_K_XL) or **32 GB** (Q8_0) beside an 8 GB card, setup's estimate (the engine itself used 16.5 GiB with IQ4_NL, 9.1 GiB with UD-Q2_K_XL). The experts the card does not hold stay in RAM. Beside a 24 GB card the IQ4_NL file's experts fit in VRAM by the same estimate. |
| **Disk** | 17.6, 9.1 or 27.8 GB for the model file. |
| **System** | Linux and a current NVIDIA driver. Setup asks before it installs it on Windows: it was not built or run there. |

It is not offered on AMD or Intel cards or on a Mac: the engine code for it was written for NVIDIA cards only. For now
it also runs on one card (no layer split); setup leaves that option out for it. It can answer several requests at once
(`--parallel`, see [Several requests at once](#several-requests-at-once)). Its attention caches are fp16, or q8_0 with
`--kv q8_0`, which halves them: a 128K context then takes about 2.3 GiB of VRAM when it is full, in fp16 4.3 GiB. On an
8 GB card, use q8_0 for contexts past 32K. The file allows 1,048,576 tokens and the engine sets no limit of its own, but
the cache stays in VRAM (about 17 KiB a token in q8_0; the engine's KV streaming is not wired up for this model), and it
was tried up to a 128K context, with prompts of up to 35K tokens. Pictures are optional: see [Images](#images). It has
no draft model, so it writes one token at a time unless you let it guess from the conversation (`--window`, see
[Guesses from the conversation](#guesses-from-the-conversation)).

## Install

```bash
./setup.sh --family inkling
```

```bash
./setup.sh --family inkling --model UD-Q2_K_XL
```

```bash
./setup.sh --family inkling --model Q8_0
```

## Which model should I pick?

| Size | What it is | Download | RAM beside an 8 GB card (estimate) |
| --- | --- | ---: | ---: |
| **IQ4_NL** | 4-bit experts in layers 0 to 19 (with an importance matrix), 8-bit everywhere else | 17.6 GB | about 22 GB |
| **UD-Q2_K_XL** | Unsloth's: 2-bit experts (Q2_K, IQ1_S and Q2_0), higher where it matters. The fastest, a little less accurate | 9.1 GB | about 14 GB |
| **Q8_0** | 8-bit throughout: the closest to the model as trained | 27.8 GB | about 32 GB |

Setup takes IQ4_NL unless you name another size. UD-Q2_K_XL comes from Unsloth's repository in the architecture's
native form (`inkling2`, the loop not unrolled); the engine reads that form as it is. Q8_0 was not run. ggml-org's model card gives the KL divergence against the BF16 file on
wikitext-2: 0.0030 for Q8_0 and 0.0095 for IQ4_NL. PolyStrata did not measure that.

## Several requests at once

With `"parallel": N` in the run config (or `--parallel N` when you run setup) the engine answers up to N requests
together, as for the other models of strata-poly ([BATCHING.md](../BATCHING.md)). Each request then has its own
attention caches and convolution state. A token of every request goes through the model in one pass, so the dense
weights, and each expert that two requests both use, are read once for all of them. With the expert cache held still
(`--check-batch`, [BATCHING.md](../BATCHING.md)), every sequence got the tokens it gets alone: IQ4_NL and UD-Q2_K_XL,
fp16 and q8_0 caches, 2, 3 and 4 sequences after a 302-token prompt and 4 after a 3,000-token one, and 4 sequences
that start, join and stop at different times (`--check-stagger`: prompts of 16, 46, 375 and 3,000 tokens, read while
the others write, answers of 64 to 16 tokens, so that a step has 1 to 4 of them).

> **Measured on:** RTX 2070 Super 8 GB, Ryzen 5 3600XT, 62 GB DDR4, Ubuntu 22.04: the server's config (128K context,
> q8_0 caches, the expert profile), chats of 300 tokens started together (greedy, thinking off), two runs each; the
> server was started again for each row. Measured before the change of 2026-11-22 above.

| Chats at once | `"parallel"` | IQ4_NL, all together | IQ4_NL, each chat | UD-Q2_K_XL, all together | UD-Q2_K_XL, each chat | Experts in VRAM (IQ4_NL / UD-Q2_K_XL) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 37.1-37.7 tokens/s | 39.6-40.2 | 46.2-47.8 | 49.6-51.2 | 1,034 / 2,005 |
| 1 | 4 | 34.5-35.0 | 36.9-37.6 | 44.3-44.9 | 46.7-47.4 | 782 / 1,547 |
| 2 | 2 | 44.6-45.5 | 23.5-24.9 | 57.7-59.7 | 30.1-32.1 | 911 / 1,777 |
| 3 | 3 | 48.9-49.0 | 17.0-18.2 | 63.2-64.5 | 21.9-23.2 | 849 / 1,663 |
| 4 | 4 | 51.2-51.9 | 13.3-14.4 | 64.1-66.5 | 16.6-18.0 | 786 / 1,547 |

"All together" is the tokens of all the chats over the time from the first request to the last answer, prompts
included. "Each chat" is a chat's writing speed as the server reports it (`timings`). On this card most experts run
on the CPU, and requests that route to different experts each need their own, so four chats together wrote 1.34 to
1.44 times as much as one alone. One chat with `"parallel": 4` was 3 to 9% slower than with 1: the slots' VRAM leaves
fewer experts on the card. A card that holds more of the experts may gain more; that was not measured.

Each slot takes VRAM from the expert cache at load: its 40 window layers' rows and its convolution state. With q8_0
caches `"parallel": 2` took about 0.5 GiB (it adds two sequences beside the first) and each slot after it about
0.27 GiB (the experts the cache held less, at the average size of a cached expert). Its full-attention rows (17 KiB a
token in q8_0) come out of the cache as its conversation grows, as the first request's do.

## Guesses from the conversation

With `--window 4` in the run config's `"args"` (`"--window", "4"`) the engine guesses what comes next where the
answer repeats the conversation: after an earlier repeat of the last three tokens or more (`--lookup N`, default 3) it
takes what followed as up to three guesses, and puts the known token and the guesses through the model together. A
guess counts when the model would have written that token itself, so the answer is the one it writes without
guesses. It is off by default: without `--window` the engine writes one token at a time, identical to `--window 1`.
The file has no draft block, so the conversation is the only source of guesses.

It pays where the answer copies text from the prompt, such as code given back with a change. Elsewhere it guesses
rarely and costs about nothing. On a card that holds few of the experts it gains less: most experts of the guessed
tokens run on the CPU, which takes about as long for a guessed token as for one alone.

> **Measured on:** RTX 2070 Super 8 GB (Ryzen 5 3600XT, 62 GB DDR4), 128K context, q8_0 caches, the model's expert
> profile; an RTX 3090 24 GB (rented, every expert of both sizes in VRAM), the same context and caches. Greedy answers
> of 300 tokens, two runs each. "Code": a 373-token prompt that asks for a 25-line Python file back with a function
> renamed; "text": a 37-token question on hash tables. The RTX 3090 rows are from an earlier build: the same window
> code on an earlier revision of the batch slots.

| | `--window 1` | `--window 4` | Guesses right |
| --- | ---: | ---: | ---: |
| RTX 2070 Super, IQ4_NL, code | 35.1-35.4 tokens/s | 38.8-40.0 | 102 of 119, 115 of 131 |
| RTX 2070 Super, IQ4_NL, text | 36.7-37.8 | 36.7-37.2 | 2 of 12, 4 of 14 |
| RTX 2070 Super, UD-Q2_K_XL, code | 47.1-47.4 | 47.9-48.1 | 40 of 60, 40 of 60 |
| RTX 2070 Super, UD-Q2_K_XL, text | 48.2-48.4 | 48.8-48.8 | 14 of 19, 13 of 19 |
| RTX 3090, IQ4_NL, code | 59.4-59.8 | 64.3-64.4 | 56 of 99 |
| RTX 3090, IQ4_NL, text | 58.0-59.8 | 59.9-60.1 | 7 of 16 |
| RTX 3090, UD-Q2_K_XL, code | 68.1-68.2 | 74.6-75.2 | 64 of 128 |
| RTX 3090, UD-Q2_K_XL, text | 68.2-68.5 | 70.1-70.2 | 16 of 25 |

With every expert on the CPU (`--no-gpu-experts`), `--window 4` wrote the same 200 tokens as `--window 1`, with the
same five likeliest tokens and their probabilities at every step: IQ4_NL and UD-Q2_K_XL, three prompts each, greedy
(RTX 2070 Super, q8_0 caches). `tools/window_agree.py` runs that check on a model and a prompt of your choice; on
the code prompt (both sizes) and on its own repeated prompt (UD-Q2_K_XL) it found the same 200 tokens and five
likeliest at every step, with 35 to 128 guesses right. With
the card's expert cache holding the same experts (`STRATA_FREEZE_EXPERTS=1`, and for `--window 1` a larger
`--vram-reserve-mib`, because a window's rows take VRAM from the cache), UD-Q2_K_XL's answers were the same too, code
and text. Where the two caches held one expert apart (IQ4_NL: 1,006 and 1,007 of 3,072), they parted at a token the
model rated about as likely as another.

Without `--window`, 200 greedy tokens were compared with inkling2 791be9e's (the same tokens and twenty likeliest
tokens with their probabilities): IQ4_NL and UD-Q2_K_XL, fp16 and q8_0 caches, code and text prompts, every expert on
the CPU and with the cache holding the same experts. 15 of the 16 runs were identical. In the 16th (UD-Q2_K_XL, fp16,
code, experts cached) the two caches never held the same experts (1,864 and 1,869 at load), and the answers first
differ at a token the model rated about as likely as another (ln p1/p2 0.15).

## Images

Setup asks whether you want pictures; the answer is no unless you say yes, so `--yes` installs text only.
`--vision cpu` (or `gpu`, or `yes` for setup's choice by VRAM) adds them:

```bash
./setup.sh --family inkling --vision cpu
```

That downloads ggml-org's image encoder, `mmproj-Inkling-2-Mini-Q8_0.gguf` (0.35 GB), and compiles strata-vision
from llama.cpp 29b52d398b, the commit that added the model and its encoder, into `engine-poly/vision-inkling/`. The
other models' encoder (`engine-poly/strata-vision`, an older llama.cpp) is not changed. The encoder is llama.cpp's
own (its `mtmd` library); PolyStrata places its rows where the template writes the picture
(`<|content_image|>` and one `<|unused_200054|>` for each of the picture's rows after it, in a block of its own, as
llama.cpp does) and does not norm them, as llama.cpp does not. A picture in a tool's message goes in the same way.

A picture larger than the token limit (setup's 1,024 on the GPU, 300 on the CPU) is made smaller first, as
llama.cpp does with the same limit. A picture costs its rows in the context: 640 x 640 pixels is 256 tokens.

On an 8 GB card setup puts the encoder on the CPU. Measured on an RTX 2070 Super 8 GB with a Ryzen 5 3600XT
(6 encoder threads), IQ4_NL, flash attention on in the encoder (its default):

| | 256 rows (640 x 640) | 990 rows (1000 x 2000, at the 1,024 limit) | VRAM |
| --- | ---: | ---: | ---: |
| encoder on the CPU | 2.1 s | 12.3 s | none |
| encoder on the GPU (`--vision gpu`) | 0.06 s | 0.31 s | 576 MiB after its warm-up |

The photo below (208 rows) took 1.7 s on the CPU (setup's `--vision cpu` build, 6 threads) and 55 ms on the GPU.

With the encoder on the GPU the engine held 908 routed experts in VRAM instead of 1,038. A prompt with a picture
was read at the speed of a text prompt: 282 tokens with a 256-row picture in 1.2 to 1.35 s (209 to 239 tokens/s in
5 server runs, the encoder on the CPU in 3 of them, on the GPU in 2), 323 tokens of text in 1.4 to 1.5 s (216 and 226
tokens/s, 2 runs). A whole request (the encode, the prompt and an answer of 90 to 170 tokens) took 6.1 to 8.4 s with
the encoder on the CPU and 3.7 to 6.0 s with it on the GPU.

**UD-Q2_K_XL** reads pictures too (the same encoder; `--model UD-Q2_K_XL --vision gpu`). With the encoder on the GPU
it read the 282-token prompt at 354 tokens/s and answered at 50 tokens/s; a whole request took 3.3 to 3.8 s.

**How it was checked.** Against llama.cpp 29b52d398b's `mtmd` on the CPU, for 6 cases (7 drawn pictures from 1 to
990 rows, two of them in one prompt) and a photo (llama.cpp's `tools/mtmd/test-1.jpeg`, a newspaper page, 640 x 488
pixels, 208 rows), with the Q8_0 encoder and IQ4_NL unless named:

- **Rows.** strata-vision's were bit-identical to llama.cpp's with flash attention off (the drawn pictures). With
  flash attention on (the default) their cosine similarity was 0.98 or more, 0.999 on average (the photo: 0.970 or
  more); with the encoder on the GPU 0.965 or more, 0.998 on average, and for the photo 0.957 or more, 0.999 on
  average.
- **Logits.** The engine, given llama.cpp's rows, against llama.cpp: cosine similarity of the prompt's logits 0.99986
  or more, and greedy decoding chose the same token in all 33 positions (the prompt's and 32 steps) for every
  case and the photo. With `--kv q8_0`: 32 of 33 for the 640 x 640 picture (the miss where llama.cpp's top two were
  0.08 apart; a second run with the fp16 cache missed there too), 33 of 33 for the 990-row picture and the photo
  (0.99980). With the photo split across calls (`--prompt-batch 64`): 0.99975, 33 of 33. With the rows normed like
  a token's, as the engine did first, it was 0.98485 or more and 29 to 32 of 33. Rows from strata-vision with flash
  attention on, from the GPU encoder, or from the BF16 encoder gave the same agreement (0.99989 or more, 33 of 33,
  on 2 pictures).
- **UD-Q2_K_XL.** The engine on Unsloth's file against llama.cpp on the same weights (converted to the form
  llama.cpp loads, as for the text checks): the photo 0.99990 (33 of 33; `--kv q8_0` 0.99985, 33 of 33), the 640 x
  640 picture 0.99993 (32 of 33, at step 11), the two pictures 0.99984 (33 of 33).
- **Answers.** Asked "Describe the picture in one sentence." at temperature 0 about a 640 x 640 test picture, the
  server and `llama-mtmd-cli` both answered: *A light blue background shows a red circle in the upper left, the word
  "HELLO" in bold black letters across the middle, and a green square in the lower right.* Two server runs with the
  encoder on the CPU gave that answer, one of them installed by setup with `--vision cpu`; two others and two with
  it on the GPU worded the same picture a little differently. The engine's runs are not bit-for-bit repeatable, and
  the reasoning before the answer has a near tie. About the photo, the `--vision cpu` install answered word for
  word as `llama-mtmd-cli`: *A front page of The New York Times from July 20, 1969, with the bold headline "MEN WALK
  ON MOON" and a black-and-white photo of the lunar surface and spacecraft.* The `--vision gpu` install read the
  same headline, subhead and date and answered *A vintage New York Times front page from July 20, 1969, with the
  bold headline "MEN WALK ON MOON" and a black-and-white photo of the lunar surface.* With UD-Q2_K_XL (encoder on
  the GPU) the server and `llama-mtmd-cli` both answered *A vintage New York Times front page headlined "MEN WALK ON
  MOON" with a black-and-white photo of the lunar surface and spacecraft.*; about the 640 x 640 picture both named
  the red circle, "HELLO" and the green square, in other words.

Why these settings: the Q8_0 encoder is half the BF16 one's download, uses 576 instead of 904 MiB of VRAM and is
faster (256 rows: 2.1 against 3.3 s on the CPU), and the engine's logits were as close to llama.cpp's with either.
Flash attention off made the encoder slower (256 rows 2.8 s, 990 rows 22.7 s on the CPU) and used 1,584 MiB of VRAM
on the GPU, for no change in the logits that could be measured (on the drawn pictures).

## How it talks

The server reads these from the tokenizer that setup writes from the GGUF (`pre` = `inkling2`):

- **Tokens.** Byte-level BPE with 200,058 tokens, llama.cpp's `inkling2` split (its GPT-4o pattern) and, as in the
  model's `tokenizer.json`, `ignore_merges`: a piece that is a token as a whole is that token. No token starts a
  prompt. A reply ends at `<|content_model_end_sampling|>` (the GGUF's end of text) or at `<|endoftext|>`.
- **Reply.** A reply is a list of blocks. Each starts with `<|message_model|>` and ends with `<|end_message|>`. A
  `<|content_thinking|>` block is the reasoning, a `<|content_text|>` block is the answer, and a tool call is
  `<|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>`. This is the form
  llama.cpp reads (its Inkling parser, ggml-org/llama.cpp PR 32381). The model writes the first header itself.
- **Reasoning effort.** The template writes "Thinking effort level" in a system block. `reasoning_effort` low, medium
  and high give 0.2, 0.7 and 0.9. High (0.9) is the default. Thinking turned off gives 0. The model can still think a
  little at 0: its model card says the lowest effort does not turn reasoning off. "xhigh" and "max" are 0.9 too,
  and "minimal" is thinking off (0), as for every model here: the template's 0.1 and 0.99 are not offered.
- **Sampling.** Setup writes the GGUF's own defaults into the run config: temperature 1.0, top_k 50, top_p 1.0. A
  `"sampling"` block you wrote yourself is kept.
- **Forced calls.** `tool_choice` naming one tool is written for the model after its thinking, or at the start of
  the reply with thinking off. The reply ends with that call: a block the model opens after it is not sent.
  `"required"` with several tools is left to the model.

## How it was checked

Token ids against Hugging Face's `tokenizers` library on the model's `tokenizer.json`, the reference. Prompts and
replies against llama.cpp 29b52d398b, the commit that added the model, built on the CPU and run without the model's
weights. 51 of the strings (53 cases), the 14 prompts and the 11 replies are kept in `serve/fixtures/` and checked by
`serve/test_inkling.py`.

- **Tokens.** 1,006 strings, about 587,000 tokens: English, Japanese, Chinese, Korean, Arabic, Hindi and Russian text,
  code, emoji, whitespace runs, control tokens, PolyStrata's docs and source in pieces, and 300 random strings. Every
  id was the same as Hugging Face's. llama.cpp's differed in 11 of the strings (see below).
- **Prompts.** 14 requests rendered by llama.cpp's chat code and by the server were the same text: effort levels, a
  system prompt, tools, two tool calls and their answers, earlier reasoning, Unicode and quotes.
- **Replies.** 11 replies (llama.cpp's own Inkling tests and a few more) gave the same reasoning, answer and tool calls,
  fed whole, in 5-character pieces and one character at a time.

The engine's output was checked against llama.cpp's on the same files, IQ4_NL and UD-Q2_K_XL, for a short prompt and prompts
of 302 and 1,500 tokens, each run twice: the logits' cosine similarity was 0.9989 to 0.9998, and 16 to 20 of the 20
most likely tokens were the same. With llama.cpp's own 32-token answer fed in one token at a time, the most likely
token was the same at 29 to 32 of the 32 steps. Where it differed, llama.cpp's two best tokens were within 0.14 of each
other in logit, except at one step (UD-Q2_K_XL, the 1,500-token prompt, in every run), where they were 1.0 apart. Two
runs of the same input do not give identical logits here (cosine 0.9992 between two runs of the 302-token prompt).

## Known differences

- llama.cpp 29b52d398b does not set `ignore_merges` for `inkling2`, so a word that is one token but that no chain of
  merges reaches comes out as several there: `kwargs` is one token here and with Hugging Face, three in llama.cpp.
  That changed 11 of the 1,006 strings above. PolyStrata follows Hugging Face. `serve/fixtures/inkling2_golden.json`
  keeps llama.cpp's ids for one of them (`kwargs`).
- A message whose content is a list of text parts is one block here and one block a part in llama.cpp. The server
  joins the parts for every model.
- llama.cpp checks a tool call against the declared tools with a grammar while the model writes it. PolyStrata has no
  grammar: it reads the call when its block is closed. A call to a tool that was not declared is shown as text.

## How does it work?

The engine side is in `src/poly/inkling.cpp`. The server side is `InklingParser` in `serve/frontend.py`, and the tests
are `serve/test_inkling.py` and `tools/test_setup_inkling.py`.
