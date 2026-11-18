# Ornith 1.5 122B at 256K and 512K context: stock, KV-resident q8_0, KV-resident q4_0

Three builds of `strata-poly` on two PCs, with `polystrata.py benchmark --model ornith-1` (prompts of 21, 2,952
and 10,420 tokens, 256 tokens out at most). One run of each build at each context size. Every number in the
tables below is copied from the benchmark's JSON files (`bench-ornith122-<pc>-<build>-<ctx>.json`) by
`gen_unified_tables.py`; none is typed by hand.

- **Stock**: the branch's base, `--kv q8_0`, no `--kv-resident`.
- **KV-resident, q8_0**: this PR, `--kv-resident N --kv q8_0`. The newest N rows stay in VRAM, every row is also
  kept in pinned host RAM, and attention over older rows streams them through VRAM in pieces.
- **KV-resident, q4_0**: the same with `--kv q4_0`. Rows in host RAM are ggml `q4_0` blocks.

## The two PCs

| | PC1 | PC2 |
| --- | --- | --- |
| OS | Windows 11 Pro 64-bit | Zorin OS 18.1 (Ubuntu 24.04 base), Linux 7.0.0-43-generic |
| CPU | AMD Ryzen 9 9950X3D, 16 cores / 32 threads | Intel Core i7-10700K, 8 cores / 16 threads |
| RAM | 64 GB DDR5-6000, 2 × 32 GB (TeamGroup TMD532GB6000URGB36), dual channel | 64 GB DDR4 at 2666 MT/s, 4 × 16 GB on the CPU's two channels |
| GPU | NVIDIA GeForce RTX 4060 Ti 16 GB, PCIe 4.0 x8, driver 610.88 | NVIDIA GeForce RTX 3050 8 GB, x8 link, 130 W limit, driver 595.99.02 |
| GPU mod | none | [Zagitta `gpu-unlock`](https://github.com/Zagitta/gpu-unlock) v2.0.0: the full tensor-core rate the same silicon has as an RTX A-series card (Linux only) |
| Model file | Ornith-1.5-122B-Uncensored-AD-IQ3_XXS (2 parts) | Ornith-1.5-122B-UD-IQ3_XXS |
| Engine flags | `--kv-resident 32768 --tail-skip 6 --prompt-batch 2048 --cache-bias 0.5 --vram-reserve-mib 1408 --conversation-cache-mib auto` | `--kv-resident 16384 --vram-reserve-mib 1024 --conversation-cache-mib auto` |

The two PCs ran different files and different flags, so compare builds on one PC, not the PCs with each other.

## Row sizes (Ornith 122B: 2 KV heads of 256)

| KV format | Bytes per token per attention layer | Against q8_0 |
| --- | ---: | ---: |
| f16 | 2,048 | +88% |
| q8_0 | 1,088 | — |
| q4_0 | 576 | −47.1% |

The engine prints the row size it uses at load (`q4_0 KV cache: 576 bytes a row (288 K + 288 V)`).

## PC1: RTX 4060 Ti 16 GB, DDR5-6000

### 256K context

| Build | Prompt | Tokens in | TTFT s | Prefill tok/s | Decode tok/s | Expert cache hits | MTP accepted | VRAM GiB | RAM GiB | GPU % | CPU % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stock (no `--kv-resident`, q8_0 KV) | short | 21 | 2.11 | 10.0 | 24.3 | 62.2% | 77.9% | 14.96 | 56.05 | 36.7 | 35.5 |
| Stock (no `--kv-resident`, q8_0 KV) | long-2000 | 2,952 | 6.22 | 492.6 | 35.0 | 60.5% | 74.6% | 15.64 | 54.88 | 77.3 | 32.8 |
| Stock (no `--kv-resident`, q8_0 KV) | long-8000 | 10,420 | 11.86 | 713.8 | 36.0 | 67.5% | 66.5% | 15.67 | 53.71 | 83.8 | 21.0 |
| KV-resident, q8_0 KV | short | 21 | 2.21 | 9.6 | 20.9 | 61.2% | 73.2% | 14.88 | 46.67 | 48.8 | 38.0 |
| KV-resident, q8_0 KV | long-2000 | 2,952 | 6.43 | 479.0 | 30.9 | 60.5% | 65.3% | 15.67 | 57.90 | 73.6 | 30.7 |
| KV-resident, q8_0 KV | long-8000 | 10,420 | 12.06 | 704.1 | 32.7 | 67.3% | 61.4% | 15.63 | 57.21 | 82.9 | 21.9 |
| KV-resident, q4_0 KV | short | 21 | 2.87 | 7.4 | 17.7 | 63.1% | 78.6% | 14.89 | 47.72 | 38.8 | 33.4 |
| KV-resident, q4_0 KV | long-2000 | 2,952 | 6.64 | 447.7 | 30.6 | 59.4% | 73.7% | 15.68 | 58.48 | 68.4 | 35.2 |
| KV-resident, q4_0 KV | long-8000 | 10,420 | 12.05 | 703.6 | 30.1 | 67.5% | 59.1% | 15.59 | 57.99 | 81.9 | 25.4 |

Long prompts against stock:

| Prompt | Decode stock | Decode q8_0 resident | Decode q4_0 resident | Prefill stock | Prefill q8_0 resident | Prefill q4_0 resident |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| long-2000 | 35.0 | 30.9 (-11.7%) | 30.6 (-12.6%) | 492.6 | 479.0 (-2.8%) | 447.7 (-9.1%) |
| long-8000 | 36.0 | 32.7 (-9.2%) | 30.1 (-16.4%) | 713.8 | 704.1 (-1.4%) | 703.6 (-1.4%) |

### 512K context

| Build | Prompt | Tokens in | TTFT s | Prefill tok/s | Decode tok/s | Expert cache hits | MTP accepted | VRAM GiB | RAM GiB | GPU % | CPU % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stock (no `--kv-resident`, q8_0 KV) | short | 21 | 1.33 | 16.1 | 28.7 | 64.2% | 74.5% | 14.95 | 54.71 | 52.9 | 50.0 |
| Stock (no `--kv-resident`, q8_0 KV) | long-2000 | 2,952 | 6.10 | 504.7 | 31.8 | 60.6% | 67.4% | 15.56 | 54.12 | 75.7 | 31.9 |
| Stock (no `--kv-resident`, q8_0 KV) | long-8000 | 10,420 | 11.77 | 718.4 | 35.3 | 67.8% | 64.8% | 15.57 | 53.83 | 85.6 | 21.8 |
| KV-resident, q8_0 KV | short | 21 | 7.56 | 2.8 | 20.2 | 64.2% | 72.1% | 14.85 | 56.63 | 33.5 | 22.2 |
| KV-resident, q8_0 KV | long-2000 | 2,952 | 6.69 | 443.2 | 26.8 | 61.7% | 61.2% | 15.64 | 57.82 | 73.2 | 33.7 |
| KV-resident, q8_0 KV | long-8000 | 10,420 | 12.38 | 677.5 | 30.8 | 68.8% | 65.8% | 15.63 | 58.81 | 80.3 | 26.1 |
| KV-resident, q4_0 KV | short | 21 | 5.47 | 3.9 | 22.0 | 63.7% | 71.6% | 14.93 | 56.58 | 34.7 | 27.1 |
| KV-resident, q4_0 KV | long-2000 | 2,952 | 6.56 | 451.5 | 29.4 | 59.7% | 73.4% | 15.67 | 57.45 | 69.5 | 35.4 |
| KV-resident, q4_0 KV | long-8000 | 10,420 | 12.38 | 677.8 | 29.6 | 66.7% | 66.8% | 15.69 | 58.46 | 78.7 | 27.5 |

Long prompts against stock:

| Prompt | Decode stock | Decode q8_0 resident | Decode q4_0 resident | Prefill stock | Prefill q8_0 resident | Prefill q4_0 resident |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| long-2000 | 31.8 | 26.8 (-15.7%) | 29.4 (-7.5%) | 504.7 | 443.2 (-12.2%) | 451.5 (-10.5%) |
| long-8000 | 35.3 | 30.8 (-12.7%) | 29.6 (-16.1%) | 718.4 | 677.5 (-5.7%) | 677.8 (-5.7%) |

## PC2: RTX 3050 8 GB with gpu-unlock, DDR4-2666

### 256K context

| Build | Prompt | Tokens in | TTFT s | Prefill tok/s | Decode tok/s | Expert cache hits | MTP accepted | VRAM GiB | RAM GiB | GPU % | CPU % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stock (no `--kv-resident`, q8_0 KV) | short | 21 | 1.48 | 14.2 | 13.1 | 8.0% | 74.3% | 7.18 | 3.98 | 33.6 | 61.5 |
| Stock (no `--kv-resident`, q8_0 KV) | long-2000 | 2,952 | 14.05 | 210.2 | 13.1 | 4.2% | 78.8% | 7.72 | 4.10 | 65.2 | 50.0 |
| Stock (no `--kv-resident`, q8_0 KV) | long-8000 | 10,420 | 26.05 | 321.6 | 13.4 | 5.6% | 81.4% | 7.89 | 5.04 | 71.8 | 38.0 |
| KV-resident, q8_0 KV | short | 21 | 1.45 | 14.5 | 13.7 | 7.5% | 84.6% | 7.18 | 7.45 | 35.8 | 62.0 |
| KV-resident, q8_0 KV | long-2000 | 2,952 | 14.03 | 210.6 | 12.7 | 3.4% | 78.3% | 7.78 | 7.59 | 62.9 | 50.4 |
| KV-resident, q8_0 KV | long-8000 | 10,420 | 25.95 | 322.9 | 13.5 | 4.9% | 82.7% | 7.90 | 8.47 | 75.6 | 38.1 |
| KV-resident, q4_0 KV | short | 21 | 1.45 | 14.5 | 13.4 | 7.3% | 79.1% | 7.18 | 5.85 | 34.6 | 62.6 |
| KV-resident, q4_0 KV | long-2000 | 2,952 | 14.02 | 210.7 | 13.0 | 2.3% | 81.8% | 7.84 | 6.01 | 62.7 | 50.0 |
| KV-resident, q4_0 KV | long-8000 | 10,420 | 25.98 | 322.5 | 12.7 | 2.2% | 79.1% | 7.99 | 6.84 | 73.4 | 38.8 |

Long prompts against stock:

| Prompt | Decode stock | Decode q8_0 resident | Decode q4_0 resident | Prefill stock | Prefill q8_0 resident | Prefill q4_0 resident |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| long-2000 | 13.1 | 12.7 (-3.1%) | 13.0 (-0.8%) | 210.2 | 210.6 (+0.2%) | 210.7 (+0.2%) |
| long-8000 | 13.4 | 13.5 (+0.7%) | 12.7 (-5.2%) | 321.6 | 322.9 (+0.4%) | 322.5 (+0.3%) |

### 512K context

| Build | Prompt | Tokens in | TTFT s | Prefill tok/s | Decode tok/s | Expert cache hits | MTP accepted | VRAM GiB | RAM GiB | GPU % | CPU % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stock (no `--kv-resident`, q8_0 KV) | short | 21 | 1.46 | 14.5 | 14.2 | 8.3% | 89.7% | 7.22 | 4.00 | 31.5 | 63.3 |
| Stock (no `--kv-resident`, q8_0 KV) | long-2000 | 2,952 | 13.99 | 211.2 | 12.8 | 4.3% | 77.0% | 7.76 | 4.20 | 64.6 | 50.0 |
| Stock (no `--kv-resident`, q8_0 KV) | long-8000 | 10,420 | 25.96 | 322.7 | 13.7 | 5.9% | 83.1% | 7.94 | 5.09 | 75.8 | 37.9 |
| KV-resident, q8_0 KV | short | 21 | 1.48 | 14.3 | 14.0 | 7.8% | 87.5% | 7.20 | 10.95 | 32.2 | 63.0 |
| KV-resident, q8_0 KV | long-2000 | 2,952 | 13.97 | 211.4 | 13.0 | 3.8% | 81.3% | 7.80 | 11.14 | 64.5 | 49.9 |
| KV-resident, q8_0 KV | long-8000 | 10,420 | 26.01 | 322.1 | 13.2 | 5.0% | 78.1% | 7.91 | 12.01 | 74.4 | 39.3 |
| KV-resident, q4_0 KV | short | 21 | 1.46 | 14.4 | 13.4 | 7.0% | 81.6% | 7.18 | 7.80 | 29.9 | 61.6 |
| KV-resident, q4_0 KV | long-2000 | 2,952 | 14.03 | 210.6 | 12.6 | 2.8% | 77.9% | 7.84 | 7.88 | 63.6 | 49.9 |
| KV-resident, q4_0 KV | long-8000 | 10,420 | 25.98 | 322.5 | 12.6 | 2.2% | 77.8% | 8.00 | 8.76 | 72.6 | 38.9 |

Long prompts against stock:

| Prompt | Decode stock | Decode q8_0 resident | Decode q4_0 resident | Prefill stock | Prefill q8_0 resident | Prefill q4_0 resident |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| long-2000 | 12.8 | 13.0 (+1.6%) | 12.6 (-1.6%) | 211.2 | 211.4 (+0.1%) | 210.6 (-0.3%) |
| long-8000 | 13.7 | 13.2 (-3.6%) | 12.6 (-8.0%) | 322.7 | 322.1 (-0.2%) | 322.5 (-0.1%) |

VRAM and RAM are the machine's use at its peak while the request ran, not the engine's alone. The short prompt's
prefill (21 tokens) is mostly the time to the first token, not a rate; read the long prompts.

## What the numbers say

**Host RAM.** On PC2 the KV-resident builds hold the whole context's rows in pinned host RAM from the start. At
512K the machine's peak went from 5.09 GiB (stock) to 12.01 GiB with q8_0 rows and 8.76 GiB with q4_0 rows on the
10,420-token prompt: q4_0 took 3.25 GiB less than q8_0.

**PC2 speed.** Prefill is the same in all three builds within 0.4%. Decode is within −8.0% to +1.6% of stock. On
this PC almost no expert fits in VRAM (107 to 129 of 12,544 at load, 2% to 8% cache hits), and a token's time goes
to the CPU's experts.

**PC1 speed: slower than stock.** On the long prompts decode is 7.5% to 16.4% below stock in both KV-resident
builds, and prefill 1.4% to 12.2% below. All these prompts fit in the resident window (10,420 < 32,768 rows), so
attention never streams from host RAM here; the loss comes from somewhere else. What the engine's log shows at
load:

| PC1, 512K | Experts in VRAM | RAM budget for the experts outside VRAM (38.8 GiB of them) |
| --- | ---: | --- |
| Stock | 1,601 (5.6 GiB) | all of them (no budget line printed) |
| KV-resident, q8_0 | 1,579 (5.5 GiB) | 35,968 MiB; the rest from the file cache |
| KV-resident, q4_0 | 1,560 (5.5 GiB) | 37,687 MiB; the rest from the file cache |

The pinned host rows are taken at load for the whole `--max-context`, before the engine looks at free RAM for the
experts. With less free, it falls under the "experts plus 4 GiB" line and keeps only a budget of them in RAM
(docs/BACKENDS.md, "When the experts do not fit in RAM"); the others come back a page fault at a time. That fits
the slower decode, but it is not proven: the run that would show it (KV-resident with `--ram-experts-mib -1`, or
with the host rows allocated as the context grows) has not been done. With one run per build, the run-to-run noise
is not known either.

**Experts in VRAM.** The KV-resident builds load 22 (q8_0) and 41 (q4_0) fewer experts than stock on PC1, and
the q8_0 build 18 (256K) and 20 (512K) fewer on PC2. In the q4_0 build the VRAM holds both a q4_0 copy and a q8_0 copy of the resident rows
(the q4_0 copy is where new rows are written before they are unpacked to q8_0 and copied to host RAM). Attention
reads only the q8_0 copy.

## Not measured

- A prompt longer than the resident window (over 16,384 rows on PC2, 32,768 on PC1). Every prompt above fits in
  it, so none of these runs streams rows from host RAM during attention. The 512K numbers show what the larger
  host buffer costs, not what attending over 512K tokens costs.
- Whether stock fails or slows at long prompts on these PCs.
- Answer quality of q4_0 rows (perplexity, KL against q8_0, a needle test). The q4_0 to q8_0 unpack is exact, but
  quantizing a row to q4_0 loses more than q8_0 does. The token-for-token check done for the q8_0 KV-resident build
  was not repeated for q4_0.
- More than one run per build.
