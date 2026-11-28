# Community benchmark on AMD Radeon 6600XT

Measured on 2026-11-28 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon 6600XT, 8 GiB VRAM
- CPU: AMD Ryzen 7 5700X 8-Core Processor (8 cores, 16 threads); installed RAM: 62.7 GiB; storage: sda     WDC WD40EFAX-68JH4N1         sata; sdb     WDC WD40EFAX-68JH4N1         sata; nvme0n1 Samsung SSD 970 EVO Plus 1TB nvme
- OS: Linux 7.3.6-arch1-1; GPU driver: `amdgpu`; CUDA (driver supports up to): not read
- PolyStrata: project 0.1.66, commit not a git checkout; engine 0.1.66
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: flint-40b-a2.5b-ud-q4_k_xl
- Context: 128000 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata-poly --model flint-UD-Q4_K_XL.gguf --max-context 128000 --expert-profile expert-profile.txt --conversation-cache-mib auto --prompt-batch 2048 --kv q8_0 --vram-reserve-mib 1024 --kv-whole
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
| short chat | 30 (27-34) | 0 | 100 (100-100) | 9 | not measured | 41.4 (34.8-46.7) | 0.39 (0.34-0.51) |
| 4K prompt, fresh | 4100 (4100-4100) | 0 (0-0) | 24 cap | 3 | 835.9 (834.6-843.5) | not measured | 4.94 (4.89-4.95) |
| 16K prompt, fresh | 16388 (16388-16388) | 0 (0-0) | 24 cap | 3 | 665.9 (665.5-667.0) | not measured | 24.66 (24.62-24.68) |
| agent session, follow-up turns | 13511 (13511-13511) at the last turn | see below | up to 64 | 3 | see below | 39.8 (36.0-44.0) | 5.06 (3.41-6.16) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.2 % (68.2-68.2); earlier-turn tokens read again per session: 325 (325-325) (earlier turns that the cache did not hold; the 12733 (12733-12733) new tokens of the files are read once); first turn (nothing cached) first token: 1.72 s (1.68-2.03). Corpus: `repo_docs` (53e7037ff848) [README.md, docs/HOW_IT_WORKS.md, docs/DETAILS.md, docs/INSTALL.md, docs/MODELS.md].

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 458 | 0 | 458 | 229.0 | 2.029 | 39.5 |
| 1 | 3590 | 457 | 3133 | 572.9 | 5.504 | 38.2 |
| 2 | 5125 | 3589 | 1536 | 425.0 | 3.651 | 43.7 |
| 3 | 8149 | 5124 | 3025 | 602.9 | 5.057 | 41.1 |
| 4 | 10717 | 8148 | 2569 | 419.9 | 6.157 | 39.0 |
| 5 | 13511 | 10716 | 2795 | 471.8 | 5.965 | 36.0 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 368 s.
