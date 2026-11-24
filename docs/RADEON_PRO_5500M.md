# Tiel Coder 35B-A3B on a Radeon Pro 5500M (gfx1012)

Back to [AMD_HIP.md](AMD_HIP.md)

This page is for a PC with a Navi 14 card (RDNA1, `gfx1012`: Radeon Pro 5500M, RX 5500 XT). It says how to install
strata-poly for it, what was changed in the engine for it, and what was measured.

> **On this page:** [The machine](#the-machine) · [Install](#install) · [What was changed](#what-was-changed) ·
> [Measured](#measured) · [Context](#context) · [Not done](#not-done)

## The machine

- MacBook Pro 16 (2019) on Windows 11: Core i9-9980HK (8 cores, 16 threads, AVX2, no AVX-512), 32 GB DDR4.
- AMD Radeon Pro 5500M, 8 GB (Navi 14, 24 CUs, wave32). Windows budgets 7,383 of its 8,176 MiB for one process.
  A device-to-device copy runs at 162 GB/s. The Intel UHD 630 drives the built-in panel, so the Radeon's VRAM is free.
- ROCm 10.2.0a20260930 from AMD's TheRock Windows wheels (`rocm[libraries,devel,device-gfx1012]`), Visual Studio 2026's
  C++ tools, ROCm's clang for host and device code.

## Install

```bat
set STRATA_EXPERIMENTAL_GFX1012=1
START-HERE.bat --yes --family tiel --backend hip --vision no --model UD-IQ3_XXS --port 8081 --no-start
run-tiel-ud-iq3_xxs.bat
```

`STRATA_EXPERIMENTAL_GFX1012=1` makes setup know the card (PCI id 0x7340, or a name with RX 5500 / Pro 5300M /
Pro 5500M / W5500) and compile strata-poly for `gfx1012`. Without it setup is as it was. For that card it writes, in
place of the usual choices:

| Setting | Why |
| --- | --- |
| `--max-context 65536` | `--kv q8_0` holds 64K in the VRAM that 32K takes in fp16; `--context` sets another |
| `--kv q8_0` | at 32K it left room for 3,994 experts in VRAM instead of 3,655 |
| `--threads 8` (the physical cores) | the engine's default is every logical CPU, 16 here: slower and twice the CPU (below) |
| `--kv-whole` | as for every AMD card ([AMD_HIP.md](AMD_HIP.md#strata-poly-on-amd-beta)) |

`--vision no`: the image encoder is a CUDA program, so pictures stay off on AMD.

## What was changed

Three changes, each one the same products as before (checked as described), so the answers do not move:

1. **The signed dot4 instruction on gfx1011 / gfx1012** (`include/strata/hip_compat/intrinsics.hpp`). Navi 12 and
   Navi 14 have `v_dot4_i32_i8` (LLVM's `dot1-insts`); Navi 10 (gfx1010) does not. The engine gave gfx1012 the
   six-instruction SDWA sequence llama.cpp uses for RDNA1. `__builtin_amdgcn_sdot4` compiles to `v_dot4c_i32_i8`
   there, matched the byte-wise form on 1,048,576 random operand triples, and ran 3.47 times as fast in a loop of
   dot products. `-DSTRATA_GFX1012_SDWA_DOT=ON` keeps the old sequence.
2. **The same in llama.cpp's `ggml_cuda_dp4a`**, which the prompt path's MMQ kernels use. The checkout in
   `third_party/llama.cpp` is downloaded, so CMake patches its one line in `ggml/src/ggml-cuda/common.cuh` when the
   HIP architectures include gfx1011 or gfx1012 (`CMakeLists.txt`, before `strata_poly_mmq`).

3. **A Q6_K kernel for gfx1011 / gfx1012** (`q6k_rdna1_kernel`, `src/kernels/cuda/native_mmvq.cu`). Tiel Coder's
   attention, GDN and shared-expert matrices and its output head are Q6_K, and with the guesses on every token
   that is checked multiplies them by two to eight columns at once. The generic kernels were bound there by their
   instruction count, not by the weights' reads: four 64-bit address calculations for every column, the activations
   reloaded for every row, and a final sum whose lane exchanges each waited on `ds_bpermute`. The new kernel keeps
   every thread's blocks, every expression and every order of addition (so the outputs are the same bits), and
   gets there with buffer loads (a lane's 32-bit offset plus a column's or row's scalar one), one activation load
   per block for 2 to 8 rows, the three-operand `v_dot4_i32_i8`, and the final sum spread over the four warps as
   DPP / `permlanex16` moves. Checked bit for bit against `native_mmvq` on Tiel's real matrices at 1 to 8 columns
   (a standalone harness that is not in this tree, every output) and by `mmvq_multi_parity` (148,480 outputs, 0 differ, its control still finds the
   other layout's differences), and deterministic (406 million outputs over 300 launches a shape, half of them beside
   a busy second stream, 0 differ from the first). `STRATA_Q6K_RDNA1=0` turns it off; `=1` takes it on any gfx10
   card (unmeasured there).

   The engine's greedy answers: with the experts frozen (`STRATA_FREEZE_EXPERTS=1`) the same tokens as the build
   before at window 1, and at window 8 in most runs; some window-8 runs take another token at a near tie (token 44
   or 51 of 256). The build before does the same when only the window changes (window 2 against 1, 4 and 8: token
   44), so it is how the engine groups the checked tokens, which follows the machine's timing, not the kernel.

   | Matrix (real weights), µs | 1 column | 2 | 3 | 4 | 8 |
   | --- | ---: | ---: | ---: | ---: | ---: |
   | output head, 248,320 x 2,048: before | 3,448 | 4,537 | 5,557 | 6,686 | 12,329 |
   | after | 2,707 | 3,040 | 3,250 | 3,794 | 5,837 |
   | attn_qkv, 8,192 x 2,048: before | 100 | 147 | 180 | 219 | 410 |
   | after | 91 | 101 | 115 | 128 | 186 |
   | ssm_out, 2,048 x 4,096: before | 49 | 68 | 85 | 102 | 190 |
   | after | 47 | 51 | 53 | 61 | 83 |

   A no-load variant of the new kernel takes two thirds of its time at eight columns, so it is now bound by its
   arithmetic: nine instructions a column for every 256 weights a warp, which is what `q6_q8_dot_impl`'s
   expression costs when it must stay the same bits.

   In the engine (`strata-poly --tokens`, a 34-token prompt, 256 greedy tokens, 64K, window 8, the experts swapping
   as in use, with `-mno-gather`; three pairs, the build before and then this one): 39.2 / 36.1 / 35.1 -> 43.2 / 38.7 / 38.1 tokens/s,
   and the wait for the card's routing 10.9 / 10.5 / 9.9 -> 10.0 / 9.1 / 8.3 ms a token. Through the server (six
   coding prompts of 256 tokens, `STRATA_Q6K_RDNA1=0` and then on, twice): the
   median 39.2 -> 40.0 and 34.9 -> 36.1 tokens/s; the 12K prompt's read and the answer after it moved with the
   warming (27.9, 25.4, 21.8, 20.8 tokens/s in run order), not with the kernel.

`STRATA_HIP_SPLIT_ATTEND=2`'s int8 scores ([AMD_HIP.md](AMD_HIP.md#opt-in-split-key-decode-attention-in-strata-poly-wave32-hip))
take the same instruction on gfx1012 (`src/poly/hip_split_attend.cuh`); `poly_hip_split_attend_test` passes its 79
checks on the card.

The rows below marked *with `-mno-gather`* were measured with ggml-cpu also compiled with `-mno-gather`
(`STRATA_CPU_NO_GATHER`, pull request 171), which makes the experts outside VRAM faster on this Intel CPU; it is
not part of this page's change.

The card's matrix-vector kernels, one launch, before and after change 1 (`native_mmvq`, random weights, the mean of 200
launches):

| Matrix | Tokens | Before | After |
| --- | ---: | ---: | ---: |
| Q6_K 12,288 x 2,048 | 1 / 2 / 4 | 156 / 243 / 390 µs | 151 / 211 / 321 µs |
| Q6_K 248,320 x 2,048 (the output head) | 1 / 4 | 3.74 / 7.84 ms | 3.49 / 6.49 ms |
| IQ3_XXS 12,288 x 2,048 | 1 / 4 | 104 / 253 µs | 90 / 171 µs |
| IQ3_XXS 1,024 x 2,048 (an expert's gate and up) | 1 / 4 | 12.4 / 20.9 µs | 12.0 / 14.1 µs |

For one token Q6_K reads at 137 of the card's 162 GB/s; the cost of each further token is what is left (below).

The engine alone (`strata-poly --tokens`, a 34-token prompt, 256 greedy tokens, 32K, `--kv q8_0 --threads 8`):

| Build | Window 1 | Window 8 (the default) | CPU experts, window 8 |
| --- | ---: | ---: | ---: |
| as setup compiled it | 25.5 tokens/s | 27.0 tokens/s | 20.8 ms a token |
| changes 1 and 2 | 26.3 | 32.7 | 16.4 |
| with `-mno-gather` | 31.8 | 38.8 | 8.4 |

## Measured

> **Measured on:** the machine above, 2026-11-06, Tiel Coder 35B-A3B UD-IQ3_XXS, through the server, greedy,
> thinking off. Nothing else heavy ran.

`tools/strata_bench.py` (medians of three runs; the suite's own prompts):

| | Setup's install before these changes (32K, fp16, 16 threads) | This page's install (64K, q8_0, 8 threads), with `-mno-gather` |
| --- | ---: | ---: |
| Chat, writes | 22.0 tokens/s | 31.0 tokens/s |
| Chat, first token | 3.39 s | 1.63 s |
| 4K prompt, reads | 193 tokens/s | 382 tokens/s |
| 16K prompt, reads / first token | 154 tokens/s / 105 s | 320 tokens/s / 50 s |
| Agent session, writes / a turn's first token | 15.1 tokens/s / 20.9 s | 23.3 tokens/s / 12.6 s |
| Agent session, prompt tokens taken from the cache | 70% | 70% |

Settings one at a time (each run a server start of its own: a warm-up, six coding prompts of 256 tokens, then a
12,000-token prompt of this repository's `setup.py` and 192 tokens after it; one run each, so about 2 tokens/s either
way is noise):

| Engine | Settings | Writes, chat | After 12K | Reads 12K | CPU while writing |
| --- | --- | ---: | ---: | ---: | ---: |
| as setup compiled it | setup's (16 threads, fp16, 32K) | 19.9 to 21.7 | 9.3 to 9.8 | 172 | 83% |
| | `--threads 8` | 23.1 to 23.5 | 10.0 to 12.3 | 175 | 48% |
| | `--threads 8 --kv q8_0` | 23.0 | 11.7 | 176 | 51% |
| | `--window 1` (no guesses) | 18.2 | 9.7 | 173 | 80% |
| with `-mno-gather` | setup's (16 threads, fp16, 32K) | 26.4 | 13.3 | 312 | 81% |
| | `--threads 8 --kv q8_0`, 32K | 38.3 | 24.9 | 282 | 45% |
| | `--threads 8 --kv q8_0`, 64K | 34.8 to 37.1 | 18.4 to 22.9 | 318 to 320 | 48% |
| | `--threads 6 --kv q8_0`, 64K | 32.1 | 18.7 | 322 | 38% |
| | `--threads 4 --kv q8_0`, 64K | 28.8 | 15.2 | 322 | 27% |
| | `--threads 8 --kv q8_0 --tail-skip 6`, 64K | 33.7 | 23.3 | 320 | 49% |
| | UD-Q3_K_XL, `--threads 8 --kv q8_0`, 64K | 26.6 | 20.1 | 193 | 39% |

With 64K and q8_0 the card held 3,548 to 3,658 of the 10,496 routed experts; the guessed tokens were right 82 to 88%
of the time. The RAM left available while it wrote was 16.0 to 16.8 GB (13.2 GB with UD-Q3_K_XL), out of the 24 GB
that Windows and the usual apps leave with no model loaded.

## Context

| `--max-context` | Loads | Routed experts in VRAM |
| ---: | --- | ---: |
| 32,768 | yes | 3,994 |
| 65,536 (setup's choice here) | yes | 3,548 |
| 131,072 | yes | 2,783 |
| 262,144 (the model's own) | yes | 1,337 (not run with a long prompt) |

With `-mno-gather`: a prompt that nearly fills the context, then 512 tokens after it from the conversation cache, and a 1,024-token
answer at short context; the machine sampled about twice a second (RAM available, the engine's private bytes, CPU
across all 16 logical CPUs, every disk's reads, the engine's VRAM from Windows' GPU counters):

| | 64K: 46,312-token prompt | 128K: 88,067-token prompt |
| --- | ---: | ---: |
| Prompt read | 193 tokens/s (4.0 min) | 114 tokens/s (12.9 min) |
| CPU while reading: mean / highest | 12% / 22% | 13% / 22% |
| Writing after it | 14.6 tokens/s | 11.0 tokens/s |
| CPU while writing: mean / highest | 49% / 57% | 44% / 52% |
| A 1,024-token answer at short context | 42.1 tokens/s, CPU 44% / 53% | 33.9 tokens/s, CPU 48% / 58% |
| RAM available, lowest in any phase | 14.9 GB | 14.9 GB |
| Engine: resident / private, highest | 11.2 / 1.2 GB | 11.2 / 1.5 GB |
| Disk reads: mean / highest | 0.3 / 66 MB/s (while reading the prompt) | 0.8 / 85 MB/s |
| VRAM: dedicated / shared, while reading | 7,473 / 700 MiB | 7,471 / 828 MiB |

The resident figure is mostly the model file's pages that the engine maps (the experts outside VRAM); its own
memory is the private one. The file was in the file cache from earlier runs, so the disk was idle apart from the
start of a read; the first start after a reboot reads the 13.6 GB file once.

## Not done

- The Q6_K kernel (change 3) is bound by its arithmetic from about four columns on (4 take 1.4 times one's time,
  8 take 2.1 times); fewer instructions would need another expression, and so other bits. With one token (window 1)
  the card's share of a token is mostly its other kernels, and no change was measurable.
- The CPU slows as the laptop warms (the CPU experts went from 9 to 13 ms a token over six back-to-back runs, the
  card heating the shared cooler), so single runs here differ by up to 10%; compare paired runs.
- `--kv q8_0`'s answers part from fp16's at near-ties ([BACKENDS.md](BACKENDS.md)); no quality benchmark was run here.
- Setup gives these settings only to gfx1012 cards; the thread count was measured only on this CPU.
