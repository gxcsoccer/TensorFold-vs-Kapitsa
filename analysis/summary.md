# 机器可读汇总（由 `analysis/aggregate.py` 生成）

> `image` 列反映各引擎的**默认** server flags：TensorFold 需要 `--vision` 才接受图像，Kapitsa 默认开启；TF 0.6.6 在本机因缺 torchvision 无法启用。详见 README §5。

## Performance

| engine | load_s | mem_idle_gib | mem_peak_gib | decode_greedy | decode_sampled | ttft_ms | prefill_4k | prefill_16k | cached_ttft_ms | conc2 | conc4 | conc8 | conc8_ttft_ms | longctx_ttft_s | longctx_ok | quality | fidelity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kapitsa · 27B | 3.1 | 15.0 | 22.0 | 14.3 | 15.8 | 366.9 | 386.3 | 335.4 | 295.4 | 15.1 | 15.3 | 15.5 | 11949.6 | 71.7 | True | 10/12 | 7e79897e0f4ecf4726ed37f1 |
| TF 1.2.2 · 27B | 2.4 | 21.0 | 41.0 | 14.5 | 14.6 | 442.2 | 108.9 | 104.2 | 413.3 | 23.9 | 37.6 | 60.0 | 1406.5 | 224.9 | True | 10/12 | 7e79897e0f4ecf4726ed37f1 |
| TF 0.6.6 · 27B | 133.2 | 18.0 | 31.0 | 61.4 | 59.0 | 509.4 | 129.9 | 125.4 | 910.6 | 41.2 | 53.4 | 57.0 | 1487.1 | 183.8 | True | 10/12 | f1dcdb9e083b8a80c8222f60 |
| Kapitsa · 35B | 4.1 | 18.0 | 22.0 | 109.7 | 107.5 | 64.2 | 1189.2 | 975.5 | 88.6 | 96.4 | 100.4 | 103.9 | 1519.0 | 25.8 | True | 11/12 | 5633e6ae9f1f0c49d6c3f9ff |
| TF 0.6.6 · 35B | 22.9 | 20.0 | 29.0 | 88.3 | 87.4 | 101.6 | 930.7 | 817.2 | 323.1 | 95.7 | 138.3 | 174.4 | 430.6 | 27.7 | True | 11/12 | 5633e6ae9f1f0c49d6c3f9ff |

## Feature checks

| engine | models | metrics | completions | responses | messages | image | tools | tools_stream | json_schema | json_object | thinking_off_tokens | thinking_on_tokens | cancel_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kapitsa · 27B | 200 | 404 | 404 | 200 | 200 | 200 | True | True | True | True | 6 | 116 | True |
| TF 1.2.2 · 27B | 200 | 200 | 200 | 200 | 200 | 400 | True | True | True | True | 6 | 116 | True |
| TF 0.6.6 · 27B | 200 | 200 | 200 | 200 | 200 | 400 | True | True | False | False | 6 | 116 | True |
| Kapitsa · 35B | 200 | 404 | 404 | 200 | 200 | 200 | True | True | True | True | 6 | 471 | True |
| TF 0.6.6 · 35B | 200 | 200 | 200 | 200 | 200 | 400 | True | True | False | False | 6 | 512 | True |

## Probes

| label | load_s | prefill_4k | decode_tps | mem_peak_gib | cmd |
|---|---|---|---|---|---|
| kapitsa-27b-ctx8k | 3.4 | 406.7 | 16.1 | 17.0 | serve --model /Users/lang/.cache/huggingface/hub/models--TensorFold--Qwen3.8-27B-MLX-4bit/snapshots/22d8d53815 |
| kapitsa-27b-dflash | 3.9 | 398.4 | 93.1 | 19.0 | serve --model /Users/lang/.cache/huggingface/hub/models--TensorFold--Qwen3.8-27B-MLX-4bit/snapshots/22d8d53815 |
| kapitsa-27b-plain-s1 | 1.6 | 387.2 | 15.8 | 17.0 | serve --model /Users/lang/.cache/huggingface/hub/models--TensorFold--Qwen3.8-27B-MLX-4bit/snapshots/22d8d53815 |
| kapitsa-4b-dflash | 1.6 | 2034.1 | 404.6 | 5.0 | serve --model /Users/lang/.cache/kapitsa/models/qwen3.5-4b-4bit --private --port 8329 --context 8192 --model-i |
| kapitsa-4b-mtp | 1.1 | 1882.7 | 225.4 | 4.0 | serve --model /Users/lang/.cache/kapitsa/models/qwen3.5-4b-4bit --private --port 8358 --context 8192 --model-i |
| kapitsa-4b-plain | 1.1 | 2035.1 | 88.9 | 3.8 | serve --model /Users/lang/.cache/kapitsa/models/qwen3.5-4b-4bit --private --port 8329 --context 8192 --model-i |
| tf06-27b-nodrafts | 133.9 | 125.5 | 15.7 | 24.0 | serve /Users/lang/.cache/huggingface/hub/models--TensorFold--Qwen3.8-27B-MLX-4bit/snapshots/22d8d538154e0e5b6f |
| tf06-35b-nodrafts | 25.0 | 965.8 | 89.9 | 24.0 | serve /Users/lang/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f8 |
| tf12-27b-ctx8k | 2.3 | 107.0 | 14.7 | 21.0 | serve /Users/lang/.cache/huggingface/hub/models--TensorFold--Qwen3.8-27B-MLX-4bit/snapshots/22d8d538154e0e5b6f |
| tf12-27b-drafter | 22.1 | 109.6 | 72.9 | 25.0 | serve /Users/lang/.cache/huggingface/hub/models--TensorFold--Qwen3.8-27B-MLX-4bit/snapshots/22d8d538154e0e5b6f |
| tf12-27b-rows2048-dflash | 5.7 | 105.9 | 71.6 | 34.0 | serve /Users/lang/.cache/huggingface/hub/models--TensorFold--Qwen3.8-27B-MLX-4bit/snapshots/22d8d538154e0e5b6f |
| tf12-27b-rows2048 | 4.6 | 107.7 | 14.8 | 32.0 | serve /Users/lang/.cache/huggingface/hub/models--TensorFold--Qwen3.8-27B-MLX-4bit/snapshots/22d8d538154e0e5b6f |

## Quality per task

| engine | arith | speed | count_r | instr | json | french | syllogism | units | summarize | code | date | multi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kapitsa · 27B | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| TF 1.2.2 · 27B | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| TF 0.6.6 · 27B | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| Kapitsa · 35B | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| TF 0.6.6 · 35B | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
