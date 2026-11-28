# Community benchmark on AMD Radeon 6600XT

Measured on 2026-11-28 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon 6600XT, 8 GiB VRAM
- CPU: AMD Ryzen 7 5700X 8-Core Processor (8 cores, 16 threads); installed RAM: 62.7 GiB; storage: sda     WDC WD40EFAX-68JH4N1         sata; sdb     WDC WD40EFAX-68JH4N1         sata; nvme0n1 Samsung SSD 970 EVO Plus 1TB nvme
- OS: Linux 7.3.6-arch1-1; GPU driver: `amdgpu`; CUDA (driver supports up to): not read
- PolyStrata: project 0.1.66, commit not a git checkout; engine 0.1.66
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: inkling-2-mini-ud-q4_k_xl
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata-poly --model Inkling-2-Mini-UD-Q4_K_XL.gguf --max-context 65536 --expert-profile expert-profile.txt --conversation-cache-mib auto --attn-requant q5_k --loop-bf16 --kv-whole --vram-reserve-mib 1024
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a PolyStrata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 44 (43-50) | 0 | 100 (100-100) | 9 | not measured | 21.9 (21.1-23.1) | 0.93 (0.91-0.99) |
| 4K prompt, fresh | 4075 (4075-4075) | 0 (0-0) | 24 cap | 3 | 355.0 (352.1-356.2) | not measured | 11.48 (11.44-11.58) |
| 16K prompt, fresh | 16128 (16128-16128) | 0 (0-0) | 24 cap | 3 | 327.5 (323.1-328.4) | not measured | 49.26 (49.12-49.94) |
| agent session, follow-up turns | 15683 (15679-15694) at the last turn | see below | up to 64 | 3 | see below | 21.5 (16.1-22.7) | 12.55 (6.27-16.71) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.8-68.8); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 14899 (14899-14899) new tokens of the files are read once); first turn (nothing cached) first token: 2.67 s (2.63-4.00). Corpus: `repo_docs` (53e7037ff848) [README.md, docs/HOW_IT_WORKS.md, docs/DETAILS.md, docs/INSTALL.md, docs/MODELS.md].

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 555 | 0 | 555 | 138.8 | 4.002 | 16.6 |
| 1 | 4173 | 584 | 3589 | 214.9 | 16.711 | 16.8 |
| 2 | 5955 | 4216 | 1739 | 244.5 | 7.117 | 16.1 |
| 3 | 9434 | 6011 | 3423 | 251.1 | 13.642 | 19.8 |
| 4 | 12488 | 9480 | 3008 | 253.4 | 11.878 | 22.6 |
| 5 | 15683 | 12538 | 3145 | 241.0 | 13.061 | 20.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 477 s.
