# Community benchmark on AMD Radeon 6600XT

Measured on 2026-11-28 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon 6600XT, 8 GiB VRAM
- CPU: AMD Ryzen 7 5700X 8-Core Processor (8 cores, 16 threads); installed RAM: 62.7 GiB; storage: sda     WDC WD40EFAX-68JH4N1         sata; sdb     WDC WD40EFAX-68JH4N1         sata; nvme0n1 Samsung SSD 970 EVO Plus 1TB nvme
- OS: Linux 7.3.6-arch1-1; GPU driver: `amdgpu`; CUDA (driver supports up to): not read
- PolyStrata: project 0.1.66, commit not a git checkout; engine 0.1.66
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.6-35b-a3b-ud-q4_k_m
- Context: 128000 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata-poly --model Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --max-context 128000 --expert-profile expert-profile.txt --conversation-cache-mib auto --vram-reserve-mib 1024 --lock-experts --tail-skip 6 --kv-whole
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 58.1 (48.6-70.3) | 0.55 (0.54-0.63) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 856.9 (856.5-860.8) | not measured | 4.70 (4.67-4.70) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 919.9 (918.1-921.7) | not measured | 17.50 (17.44-17.50) |
| agent session, follow-up turns | 14899 (14898-14942) at the last turn | see below | up to 64 | 3 | see below | 50.9 (44.4-60.8) | 4.95 (3.17-7.10) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.6-68.7); earlier-turn tokens read again per session: 247 (246-290) (earlier turns that the cache did not hold; the 14078 (14078-14078) new tokens of the files are read once); first turn (nothing cached) first token: 1.81 s (1.81-2.22). Corpus: `repo_docs` (53e7037ff848) [README.md, docs/HOW_IT_WORKS.md, docs/DETAILS.md, docs/INSTALL.md, docs/MODELS.md].

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 275.4 | 2.216 | 64.3 |
| 1 | 4071 | 602 | 3469 | 489.3 | 7.099 | 50.2 |
| 2 | 5709 | 4064 | 1645 | 411.2 | 4.007 | 60.8 |
| 3 | 9062 | 5702 | 3360 | 601.1 | 5.602 | 49.0 |
| 4 | 11907 | 9055 | 2852 | 577.3 | 4.949 | 49.5 |
| 5 | 14899 | 11900 | 2999 | 496.9 | 6.046 | 44.4 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 308 s.
