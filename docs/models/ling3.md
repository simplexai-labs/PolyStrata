# Ling 3.0 Flash

Back to [README.md](../../README.md)

**Ling 3.0 Flash** by inclusionAI: 124B parameters, 5.1B of them active a token, llama.cpp's architecture
`bailingmoe3`, read from bloomer010's GGUF files. A file of 43 to 78 GB, most of which stays in RAM. Seven of its 42
layers keep an attention cache, and that cache is compressed (a 512-value latent and 64 rotary values a token), so
a 128K context takes 0.9 GB more VRAM than 32K. The others are KDA layers, GLM 5.3 Flash's gated delta rule, with a
state of fixed size. Its license is MIT.

It is new here and experimental. It has run on one Linux PC only: an RTX 4080 SUPER (16 GB) and an RTX 3080 (10 GB),
a Ryzen 7 5800X3D and 62 GB of DDR4. Not on Windows, AMD or a Mac.

## The sizes

| Size | From | Download | Experts | RAM beside a 16 GB card (setup's estimate) |
| --- | --- | ---: | ---: | ---: |
| **Q4_K_M** | [bloomer010](https://huggingface.co/bloomer010/Ling-3.0-flash-GGUF): 4-bit experts, 6- and 8-bit where they write | 78.3 GB | 75.8 GB | about 72 GB |
| **Q3_K_M** | bloomer010: 3- and 4-bit experts | 62.6 GB | 60.5 GB | about 57 GB |
| **UD-Q2_K_XL** | bloomer010: 2- and 3-bit i-quant experts | 43.4 GB | 40.5 GB | about 37 GB |

Only Q4_K_M has been run; the other two are the same model at less RAM. UD-Q2_K_XL's experts are i-quants, whose
CPU kernels are slower per weight than the K-quants' (a token's experts the card does not hold are computed on the
CPU). Setup takes the first size whose estimate the PC meets: on the PC above (62 GB) that is Q3_K_M, and Q4_K_M,
named with `--model Q4_K_M`, ran there with part of its file read from the SSD. Every file is pinned to the
repository's commit of 2026-11-17 and checked against its SHA-256.

The file carries a draft block (`blk.42.nextn`). Setup's configs leave it out (`--no-draft`): on this PC it made
answers no faster and writing deep in a long context slower, 14.9 against 20.0 tokens/s after an 88K prompt.

## How fast is it?

### One card: the RTX 4080 SUPER (16 GB)

PolyStrata's own benchmark, `polystrata.py benchmark --prompt-tokens 24000` (a short chat, then a prompt built from
this repository's documents, 256 tokens written after each), Q4_K_M, a 32K context, `--no-draft`, three runs; each
starts the server, so the short chat is the first answer it writes.

| Run | Writes a short chat | Reads the prompt (32,318 to 32,368 tokens) | Writes after it | Experts in VRAM |
| --- | ---: | ---: | ---: | ---: |
| 1 (`--vram-reserve-mib 1100`) | 11.9 tokens/s | 948 tokens/s | 6.6 tokens/s | 2,691 of 20,480 |
| 2 (`--vram-reserve-mib 1900`) | 11.6 tokens/s | 835 tokens/s | 11.8 tokens/s | 2,517 |
| 3 (`--vram-reserve-mib 1900`) | 8.6 tokens/s | 1,048 tokens/s | 13.0 tokens/s | 2,517 |

The card drives the display. During the long prompt the staging buffer took VRAM past `--vram-reserve-mib`: the card
had 0.2 GB free with a reserve of 1,100 MiB and 0.8 GB with 1,900.

### Two cards: RTX 4080 SUPER + RTX 3080, a layer split

A 128K context, `--prompt-batch 512 --no-draft`, every other setting the engine's default; 4,030 of 20,480 experts in
VRAM. Each answer is 500 tokens with thinking off; "new prompts" are two the server had not seen, asked after it had
answered two others twice (a prompt asked again is written faster, from experts the cache has just taken for it).
The long prompt is 88,621 tokens of this repository's documents with one planted sentence, 200 tokens written after
it. llama.cpp (PR #32517 at 0a054d0, `--fit`, 4,096-token batches) on the same PC and file for comparison:

| | Writes new prompts | Reads the long prompt | Writes after it | Found the sentence |
| --- | ---: | ---: | ---: | --- |
| PolyStrata | 30.9 (code), 35.5 (prose) tokens/s | 733 tokens/s | 20.0 tokens/s | yes |
| llama.cpp, no draft | 22.7 tokens/s (three prompts) | 463 tokens/s | 22.4 tokens/s | yes |
| llama.cpp, its draft block 2 ahead | 24.8 tokens/s | | | |

`--prompt-batch 512` leaves room for experts on a layer split; with `--prompt-batch 2048` (and the draft block on)
the long prompt was read at 812 tokens/s, but 3,134 experts were in VRAM and writing after it fell to 11.0. Changing `--stage-mib` from its
default turns off the staging buffer's growth for a long read (`--stage-read-mib`), and the long prompt was then read
at 110 tokens/s.

## Ling 3.0 Tiny

The same engine runs Ling 3.0 Tiny (24 layers, 128 experts, a low-rank query; [inclusionAI's Q4_K_M](https://huggingface.co/inclusionAI/Ling-3.0-tiny-GGUF),
4.8 GB), with `strata-poly --model Ling-3.0-tiny-Q4_K_M.gguf` or a run config written by hand. Setup does not offer it: its
experts (3.9 GiB) fit in most cards' VRAM, where llama.cpp does as well. On the 4080 SUPER it wrote 246 tokens/s.

## How it was checked

Against llama.cpp on the same files, five chat prompts of 35 to 8,036 tokens, 64 tokens each: see
[BACKENDS.md](../BACKENDS.md#how-each-backend-was-checked).
