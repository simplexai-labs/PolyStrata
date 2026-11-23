# Inkling 2 Mini

Back to [README.md](../../README.md)

**Inkling 2 Mini** by Thinking Machines: a file of 9 to 18 GB. It is experimental here. It has run on Linux with an
RTX 5090, on Windows with an 8 GB laptop card and on a Mac with an M5 Ultra (a beta there), one request at a time. It
reads pictures too. Its 24 layers of weights run as 48: the twelve in the middle run three
times for every token. Forty of the 48 look back 1,024 tokens only, so a long conversation takes little VRAM:
32,768 bytes a token.

## How fast is it?

Measured on an RTX 5090 (32 GB) with a Xeon Platinum 8470Q (25 vCPU) and 90 GB of RAM, on Linux. The card holds
every expert of all three sizes. Each run writes 256 tokens, the most likely one each time, at a 32K context, and each
number is the range of three runs.

| Size | Writes after a 19-token prompt | Writes after a 10,081-token prompt | Reads that prompt |
| --- | ---: | ---: | ---: |
| **UD-Q4_K_XL** | 111-112 tokens/s | 109 tokens/s | 3,290-3,300 tokens/s |
| **UD-Q3_K_XL** | 114 tokens/s | 110-112 tokens/s | 3,360-3,450 tokens/s |

llama.cpp on the same card, run in turns with the above on the same prompts (commit 29b52d3 with the fix of its
pull request 32704, `llama-server -ngl 99`):

| Size | Writes after a 19-token prompt | Writes after a 10,081-token prompt | Reads that prompt |
| --- | ---: | ---: | ---: |
| **UD-Q4_K_XL** | 82-83 tokens/s | 76-77 tokens/s | 1,540-1,550 tokens/s |
| **UD-Q3_K_XL** | 84-86 tokens/s | 77-78 tokens/s | 1,510-1,530 tokens/s |

Both runs print the three likeliest tokens of every step, which the comparison below reads; that costs a little.
Without it, UD-Q4_K_XL writes 127-129 tokens/s after the short prompt and 124 after the long one, and reads the long
one at 3,280-3,320 tokens/s (six runs each); the README's benchmark measures 128.

The two start with the same tokens. For UD-Q4_K_XL the first 13 after the short prompt and the first 14 after the
long one are the same, and they part at a step whose two leading candidates are nearly as likely (0.35 against 0.34,
and 0.028 against 0.028). For UD-Q3_K_XL the first 13 and the first 40 are the same.

On an RTX 50 card the engine multiplies the experts kept in NVFP4 (most of them, in both sizes) on the FP4 tensor
cores while it reads a prompt, with the prompt's numbers rounded to about 8 bits instead of 11. That reads a prompt
1.5 to 1.9 times as fast as multiplying them as halves (`STRATA_PREFILL_NVFP4=fp16`). It moves the answers a little:
written after the halves' greedy answers to eight prompts of 19 to 10,124 tokens, 256 tokens each, it gives the same
most likely token at 98.5% of the steps (95.3% to 100% a prompt); the halves run again give 100%. A card that holds
only part of the experts was not measured. Three runs each, 32 tokens written after the prompt:

| Size | Prompt | Halves (`fp16`) | FP4 (the default on RTX 50) |
| --- | ---: | ---: | ---: |
| **UD-Q4_K_XL** | 2,664 tokens | 2,450-2,490 tokens/s | 3,800-3,840 tokens/s |
| | 10,081 tokens | 3,200-3,220 tokens/s | 5,390-5,410 tokens/s |
| | 20,162 tokens | 3,250-3,260 tokens/s | 5,500-5,510 tokens/s |
| **UD-Q3_K_XL** | 2,664 tokens | 2,570-2,600 tokens/s | 3,750-3,880 tokens/s |
| | 10,081 tokens | 3,340-3,360 tokens/s | 5,360-5,400 tokens/s |
| | 20,162 tokens | 3,390-3,400 tokens/s | 5,480-5,510 tokens/s |

On a laptop with an RTX 4070 Laptop (8 GB), a Core i7-14700HX and 32 GB of RAM, on Windows, UD-Q3_K_XL has 1,216 of
its 3,072 experts on the card and the rest are computed on the CPU. On a card that does not hold every expert, setup
asks whether to use three small-card settings that read fewer bytes for each token and lower answer quality:
`--attn-requant q5_0` (the attention's matrices in 5 bits instead of 8), `--loop-bf16` (the loop controllers in BF16
instead of F32) and `--tail-skip 4` (a token's experts ranked fifth to eighth that are not on the card are not
computed). The default is no, and `--yes` takes no. `--inkling-small-card on` or `off` answers it without asking, and
a later setup run keeps the answer (`"small_card"` in the model's run config). The first two round more coarsely and
the third skips computation, 9% to 27% of a token's expert picks on the 8 GB cards measured; with them the same most
likely token comes at 94.3-95.0% of the steps, against 98.2-99.0% for a second run without them
([one setting at a time](#setups-four-settings-one-at-a-time)). A card that holds the whole model is not asked and
gets none of them. Expert swaps stay on: with `--expert-swaps 0` the experts the
card holds stay the ones it loaded, fewer of a token's experts are found there, and the CPU computes more of them
(16-23% slower writing after a 302-token prompt on the 8 and 16 GB cards measured, about the same after 8,192).
The fastest of three runs of the README's benchmark, measured when setup also gave `--expert-swaps 0`:

| UD-Q3_K_XL | Writes in a short chat | Reads a 30,902-token prompt | Writes after it |
| --- | ---: | ---: | ---: |
| Setup's settings | 56 tokens/s | 887 tokens/s | 42 tokens/s |

Written after the greedy answers without them to five prompts of 19 to 5,761 tokens, 256 tokens each, setup's
settings give the same most likely token at 96.7% of the steps; a second run without them gives 98.3%.

A prompt is read a group of calls at a time through each layer run, so the experts that are not on the card cross
the bus once for the group instead of once for every call and every pass of the looped layers.

`--kv fp8` keeps the attention caches in 8 bits instead of 16, which leaves about 0.7 GB more VRAM for the experts at
a 32K context.

On a Mac it runs on strata-poly's Metal engine, as a beta. Measured on an Apple M5 Ultra with 256 GB of unified memory,
UD-Q3_K_XL, the README's benchmark on setup's settings with a 40,960-token context: 256 tokens written, the most likely
one each time. Other programs were using the GPU at the same time. Two runs of the same prompts given to the engine
alone wrote 73-74 and 60-61 tokens/s and read 2,411 tokens/s.

| UD-Q3_K_XL | Writes in a short chat | Reads a 32,787-token prompt | Writes after it |
| --- | ---: | ---: | ---: |
| M5 Ultra | 74 tokens/s | 2,440 tokens/s | 61 tokens/s |

Given the 128 tokens the CUDA engine wrote after a 4,526-token prompt (UD-Q3_K_XL), it picks the same most likely token
at 126 of the 128 steps. UD-Q2_K_XL has run there too (73 tokens/s in the short chat). UD-Q4_K_XL has not been measured
on a Mac. Pictures are read on the Mac's CPU, because the picture tower has no Metal kernels: a screenshot read as 984
rows took 3.4 s.

## What you need

| | |
| --- | --- |
| **Graphics card** | One **NVIDIA** card with **8 GB of VRAM or more**. It has run on an RTX 5090 (32 GB) and on an RTX 4070 Laptop (8 GB). About 3.7 GB are taken before any expert at a 32K context. Or an **Apple Silicon Mac** (beta): it has run on an M5 Ultra. |
| **RAM** | About **21, 17 or 13 GB** beside an 8 GB card for UD-Q4_K_XL, UD-Q3_K_XL and UD-Q2_K_XL, the installer's estimate. A 32 GB card holds any of them whole. On a Mac, setup takes UD-Q3_K_XL from 32 GB of unified memory and UD-Q2_K_XL below that. |
| **Disk** | 9.1 to 17.6 GB for the model file. |
| **System** | Linux or Windows 10 / 11, and a current NVIDIA driver; or macOS 15 or newer. |

Not run yet: an AMD card, two cards (`--expert-gpu`, built but not run), UD-Q4_K_XL on the 8 GB card and UD-Q4_K_XL
on a Mac. Pictures go through Unsloth's mmproj-F16.gguf, read on the CPU (`--vision cpu`) or the card; on a Mac, on
the CPU. The draft file is an
option (`--draft-model`). The engine takes one request at a time
for this model, and the installer writes `"parallel": 1`. Its license is Apache 2.0.

## Install

```sh
./setup.sh --family inkling
```

Add `--model UD-Q3_K_XL` or `--model UD-Q2_K_XL` for a smaller size. On a card that does not hold every expert,
setup asks about the faster small-card settings, which lower answer quality (off unless you say yes;
`--inkling-small-card on|off`, see [How fast is it?](#how-fast-is-it)).

## Which model should I pick?

| Size | What it is | Download | RAM beside an 8 GB card |
| --- | --- | ---: | ---: |
| **UD-Q4_K_XL** | 4-bit experts, 8-bit where they write ([Unsloth](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF)) | 17.6 GB | about 21 GB |
| **UD-Q3_K_XL** | 3- and 4-bit experts mixed | 13.6 GB | about 17 GB |
| **UD-Q2_K_XL** | 1- to 2-bit experts: the smallest | 9.1 GB | about 13 GB |

The installer downloads the files under the repository's `Mainline_Support` folder, the layout llama.cpp runs
(`inkling2_loop`). The files at the top of the repository are another layout, which the engine refuses by name.

## How does it work?

A reply is a list of typed blocks: thinking, text, or a tool call with its arguments as JSON. The server turns
them into the same reasoning, content and tool calls as for the other models, so the OpenAI and Anthropic APIs work
as they do elsewhere. A request that asks for no thinking has its prompt end inside the text block, because the
model otherwise opens a thinking block now and then. More in
[BACKENDS.md](../BACKENDS.md#inkling-2-mini-one-card).
