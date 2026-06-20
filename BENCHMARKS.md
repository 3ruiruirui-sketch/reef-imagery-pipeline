# CPU Benchmarks — Reef Imagery Pipeline

Run `python benchmark_cpu.py` after each significant change.
Results are appended automatically by the script.

## Targets

| Metric | Target |
|--------|--------|
| Inference (16 images, 128×128) | < 30 s |
| Training time per epoch | < 300 s (5 min) |
| Peak RSS per process | < 4 000 MB |
| Encoder parameter count | < 5 M |
| OpenAltimetry API latency | < 10 s |

<!-- benchmark results are appended below this line -->

## 2026-06-19 00:48
**CPU:** Apple M4 Pro  |  **Python:** 3.14.4

| Metric | Value | Target | Pass |
|--------|-------|--------|------|
| Inference 16×128² | 0.33 s | <30 s | ✅ |
| Inference ms/image | 20.5 ms | — | — |
| Peak RSS (inference) | 656160 MB | <4000 MB | ❌ |
| Encoder | timm-mobilenetv3_small_100 | mobilenet | — |
| Params | 3,661,410 | <5M | ✅ |
| OpenAltimetry latency | 0.95 s | <10 s | ✅ |

## 2026-06-19 00:49
**CPU:** Apple M4 Pro  |  **Python:** 3.14.4

| Metric | Value | Target | Pass |
|--------|-------|--------|------|
| Inference 16×128² | 0.31 s | <30 s | ✅ |
| Inference ms/image | 19.6 ms | — | — |
| Peak RSS (inference) | 593 MB | <4000 MB | ✅ |
| Encoder | timm-mobilenetv3_small_100 | mobilenet | — |
| Params | 3,661,410 | <5M | ✅ |
| OpenAltimetry latency | 0.52 s | <10 s | ✅ |
