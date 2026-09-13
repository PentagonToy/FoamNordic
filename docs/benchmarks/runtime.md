# Runtime and model execution

These measurements cover work outside the CFD equations: field transforms, model-input preparation, inference, artefact handling and resident memory.

## Measurement index

| Measurement | Isolates | Excludes |
| --- | --- | --- |
| Field transform | Native in-place field arithmetic | Solver equations and model transport |
| Model-input packing | Cast and interleave into a packed tensor | Inference and transport |
| Equinox execution | Cold JAX activation and warm model calls | Python and JAX import |
| Joblib and ONNX | Backend load and prediction | Fjord and process startup |
| Compiled estimator | Compilation, cache load and prediction | Solver integration |
| ModelHost memory | Production SHM exchange and workspace retention | OpenFOAM equations |
| Generation turnover | Immutable runtime publication and collection | Native compilation |

Throughput is reported as processed rows or source bytes per measured second. Unless a section states otherwise, central values and deviations follow the [benchmark reporting contract](README.md).

## Field transform

Two hundred in-place fused transforms were applied to one million three-component float64 cells. Seven warm runs were reduced by their median.

| Platform | Elapsed | Field throughput |
|---|---:|---:|
| MacBook | 0.42972 ± 0.02107 s | 10,652 ± 497 MiB/s |
| Roihu CPU | 1.22934 ± 0.01652 s | 3,724 ± 50 MiB/s |

The final field range was `[1, 7.31602]` in every run.

## Model-input packing

Two hundred packs converted three one-million-cell float64 scalar fields into one interleaved float32 tensor. The comparison is an idealised two-pass cast and interleave.

| Platform | Two-pass | FoamNordic pack | Speed ratio | Source throughput |
|---|---:|---:|---:|---:|
| MacBook | 0.12257 s | 0.16095 s | 0.762× | 28,442 MiB/s |
| Roihu CPU | 0.24974 s | 0.25064 s | 0.9996× | 18,264 MiB/s |

The MacBook used seven runs. Roihu job `1090004` used one core from a `small` allocation and measured 21 runs after one warm-up. Checksums were identical.

## Equinox model execution

An Equinox MLP with ten inputs, three 128-wide hidden layers and one output ran on 262,144 rows per call. Each platform completed one unmeasured compilation and warm-up call followed by 20 measured inference calls.

| Platform | Cold activation | Median inference | Maximum deviation | Throughput | FNOM size |
|---|---:|---:|---:|---:|---:|
| MacBook CPU | 1.736e-1 ± 4.42e-2 s | 2.466e-2 s | ± 1.04e-3 s | 10,630,665 rows/s | 148.1 kB |
| Roihu GH200 | 1.208 ± 7.580e-2 s | 2.785e-3 s | ± 6.726e-4 s | 94,114,030 rows/s | 148.5 kB |

Cold activation includes FNOM loading, evaluator construction, JIT compilation and first inference, but excludes Python and JAX import. Warm inference includes NumPy-to-JAX input and contiguous NumPy output. The MacBook used JAX 0.11.0; Roihu job `1093649` used JAX 0.11.1 on one GH200. The script is `tools/benchmarks/equinox_runtime.py`.

## Joblib and ONNX model execution

Roihu job `1091323` used one thread on a `test` node. Both backends used the same scikit-learn MLP with ten inputs, three 128-wide hidden layers and one output.

| Backend | FNOM size | Median cold load | Maximum deviation |
|---|---:|---:|---:|
| Joblib | 561.8 kB | 1.288e-3 s | ± 1.142e-2 s |
| ONNX | 139.5 kB | 6.546e-4 s | ± 6.146e-2 s |

Cold load used 11 repetitions.

| Batch rows | Joblib median | ONNX median | ONNX speedup | Joblib throughput | ONNX throughput |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.614e-5 s | 5.720e-6 s | 9.81× | 17,813 rows/s | 174,825 rows/s |
| 256 | 1.941e-4 s | 9.725e-5 s | 2.00× | 1,319,098 rows/s | 2,632,391 rows/s |
| 4,096 | 2.217e-3 s | 1.500e-3 s | 1.48× | 1,847,573 rows/s | 2,730,996 rows/s |
| 262,144 | 1.845e-1 s | 1.538e-1 s | 1.20× | 1,420,869 rows/s | 1,704,270 rows/s |

Warm inference used 21 repetitions per batch. ONNX used single-threaded ONNX Runtime 1.29.0 in Python; the result excludes Fjord transport and process startup.

## Compiled estimator crossover

Roihu job `1091731` compared compiled and Joblib FNOM on one thread using an ExtraTrees regressor with 32 trees, maximum depth 10 and ten inputs.

| Backend | FNOM size | Median initial load | Maximum deviation | Median cached load |
|---|---:|---:|---:|---:|
| Compiled | 1.244 MB | 7.433e-1 s | ± 7.077e-1 s | 2.437e-3 s |
| Joblib | 1.651 MB | 3.531e-3 s | ± 9.489e-3 s | — |

The compiled initial load includes native compilation: its median compiler time was 7.399e-1 s over seven fresh caches. Eleven subsequent cache hits loaded in 2.437e-3 ± 1.233e-3 s. Joblib load also used 11 repetitions.

| Batch rows | Compiled median | Joblib median | Compiled speedup | Compiled throughput | Joblib throughput |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.751e-6 s | 9.277e-4 s | 161.31× | 173,882 rows/s | 1,078 rows/s |
| 256 | 4.497e-4 s | 1.611e-3 s | 3.58× | 569,263 rows/s | 158,934 rows/s |
| 4,096 | 7.763e-3 s | 7.416e-3 s | 0.96× | 527,628 rows/s | 552,337 rows/s |
| 262,144 | 3.641e-1 s | 3.552e-1 s | 0.98× | 720,033 rows/s | 737,962 rows/s |

Warm inference used 21 repetitions per batch. Compiled execution is faster for the two smaller batches; the paths are similar at 4,096 and 262,144 rows.

## Resident ModelHost memory

The resident-memory benchmark drives the production ModelHost over SHM with a deterministic affine kernel. Each run performs 20 exchanges of 1,024 rows, three exchanges of 1,048,576 rows, then 20 more small exchanges. The large input and output contain 80 MiB and 8 MiB of float64 data. Each platform used seven independently started ModelHost processes; the table reports the median and maximum observed deviation.

| Platform | Connected RSS | Initial small RSS | Large RSS | Returned-small RSS | Peak RSS |
|---|---:|---:|---:|---:|---:|
| MacBook | 1.125 MiB | 3.281 ± 0.015 MiB | 209.641 ± 0.016 MiB | 209.641 ± 0.015 MiB | 209.641 ± 0.015 MiB |
| Roihu CPU | 1.828 ± 0.035 MiB | 3.734 ± 0.031 MiB | 130.164 ± 0.031 MiB | 50.160 ± 0.031 MiB | 206.211 ± 0.031 MiB |

| Platform | Publication peak | Receive peak | Working peak | Retained workspace |
|---|---:|---:|---:|---:|
| MacBook | 16 MiB | 80 MiB | 96 MiB | 80 KiB |
| Roihu CPU | 16 MiB | 80 MiB | 96 MiB | 80 KiB |

Internal accounting was identical on both platforms. MacBook RSS remained at its high-water mark; Roihu returned to 50.160 MiB after the final small exchanges. Both retained only the 1,024-row active workspace. Roihu job `1093451` used one `test` node.

## Runtime generation turnover

The MacBook generation stress published 256 immutable runtime generations with a 64 KiB payload in each of seven independent series. Every publication atomically replaced the current selector and invoked garbage collection while one generation remained pinned for the entire series and another pin moved every 16 publications. After releasing both pins, exactly the current and most recent generation survived as required by the retention policy.

| Platform | Series time | Publish latency | GC latency | Collected generations |
|---|---:|---:|---:|---:|
| MacBook | 3.471 ± 0.703 s | 3.089 ms | 5.498 ms | 254 of 256 |

The 1,792 publications preserved the current generation and both pinned generations. Released pins were collected. The script is `tools/benchmarks/generation_turnover.py`.
