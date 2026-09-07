# Runtime and model execution

These measurements cover work outside the CFD equations: field transforms, model-input preparation, inference, artefact handling and resident memory.

## Field transform

Two hundred in-place fused transforms were applied to one million three-component float64 cells. Seven warm runs were reduced by their median.

| Platform | Elapsed | Field throughput |
|---|---:|---:|
| MacBook | 0.42972 ± 0.02107 s | 10,652 ± 497 MiB/s |
| Roihu CPU | 1.22934 ± 0.01652 s | 3,724 ± 50 MiB/s |

The final field range was `[1, 7.31602]` in every run.

## Model-input packing

Two hundred packs converted and interleaved three one-million-cell float64 scalar fields into one float32 model tensor. The comparison loop performs an idealised two-pass cast and interleave; FoamNordic additionally validates the contract, shape and active range and updates reusable workspace metadata.

| Platform | Two-pass | FoamNordic pack | Speed ratio | Source throughput |
|---|---:|---:|---:|---:|
| MacBook | 0.12257 s | 0.16095 s | 0.762x | 28,442 MiB/s |
| Roihu CPU | 0.24974 s | 0.25064 s | 0.9996x | 18,264 MiB/s |

The MacBook two-pass and FoamNordic results were 0.12257 ± 0.00305 s and 0.16095 ± 0.00288 s over seven runs. Roihu job `1090004` requested one `small` node, 8 CPUs and 32 GiB, bound the measurement to one core, discarded one warm-up and measured 21 runs. Its two-pass and FoamNordic results were 0.24974 ± 0.02773 s and 0.25064 ± 0.01486 s; their coefficients of variation were 3.94% and 2.38%. Every checksum was identical.

## Equinox model execution

An Equinox MLP with ten inputs, three 128-wide hidden layers and one output ran on 262,144 rows per call. Each platform completed one unmeasured compilation and warm-up call followed by 20 measured inference calls.

| Platform | Cold activation | Median inference | Maximum deviation | Throughput | FNOM size |
|---|---:|---:|---:|---:|---:|
| MacBook CPU | 1.736e-1 ± 4.42e-2 s | 2.466e-2 s | ± 1.04e-3 s | 10,630,665 rows/s | 148.1 kB |
| Roihu GH200 | 1.208 ± 7.580e-2 s | 2.785e-3 s | ± 6.726e-4 s | 94,114,030 rows/s | 148.5 kB |

The MacBook used JAX 0.11.0 and Equinox 0.13.8 on `CpuDevice(id=0)`. Roihu job `1093649` requested one GH200 and 16 CPUs on `gputest`; JAX 0.11.1 and Equinox 0.13.8 reported `CudaDevice(id=0)`, and the complete allocation finished in 24 seconds. Cold activation uses seven fresh evaluator processes and includes FNOM payload loading, evaluator construction, JIT compilation and the first inference, but excludes Python and JAX import. Warm inference uses 20 calls in one evaluator and includes conversion from the NumPy input buffer to JAX and return of the result as a contiguous NumPy array. On both platforms, FoamNordic exported the model to FNOM, reloaded it through the resident Equinox evaluator and reproduced the same deterministic result. The small artefact-size difference reflects the platform environments. The runnable benchmark is `tools/benchmarks/equinox_runtime.py`. This is a model-execution microbenchmark, not solver throughput.

## Joblib and ONNX model execution

Roihu job `1091323` ran on one `test` node with 8 allocated CPUs and 32 GiB for 27 seconds. Both backends used one thread and the same fitted scikit-learn MLP with ten inputs, three 128-wide hidden layers and one output. FoamNordic embedded the original estimator as Joblib and the converted graph as ONNX; both reproduced the scikit-learn result within their declared tolerances.

| Backend | FNOM size | Median cold load | Maximum deviation |
|---|---:|---:|---:|
| Joblib | 561.8 kB | 1.288e-3 s | ± 1.142e-2 s |
| ONNX | 139.5 kB | 6.546e-4 s | ± 6.146e-2 s |

Cold load used 11 repetitions. The long upper tails are retained and the median is the comparison statistic.

| Batch rows | Joblib median | ONNX median | ONNX speedup | Joblib throughput | ONNX throughput |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.614e-5 s | 5.720e-6 s | 9.81x | 17,813 rows/s | 174,825 rows/s |
| 256 | 1.941e-4 s | 9.725e-5 s | 2.00x | 1,319,098 rows/s | 2,632,391 rows/s |
| 4,096 | 2.217e-3 s | 1.500e-3 s | 1.48x | 1,847,573 rows/s | 2,730,996 rows/s |
| 262,144 | 1.845e-1 s | 1.538e-1 s | 1.20x | 1,420,869 rows/s | 1,704,270 rows/s |

Warm inference used 21 repetitions per batch size. Joblib was measured through FoamNordic's resident evaluator and ONNX through single-threaded ONNX Runtime 1.29.0 in Python. FoamNordic's native ModelHost pins ONNX Runtime 1.28.0, so this result characterises the model engines and FNOM artefacts rather than native ModelHost overhead. It excludes Fjord transport and process startup.

## Compiled estimator crossover

Roihu job `1091731` ran on one `test` node with 8 allocated CPUs and 32 GiB for 33 seconds. A single-threaded ExtraTrees regressor with 32 trees, maximum depth 10 and ten inputs was exported as both compiled and Joblib FNOM. Both paths reproduced the original scikit-learn prediction to a relative and absolute tolerance of `1e-14`.

| Backend | FNOM size | Median initial load | Maximum deviation | Median cached load |
|---|---:|---:|---:|---:|
| Compiled | 1.244 MB | 7.433e-1 s | ± 7.077e-1 s | 2.437e-3 s |
| Joblib | 1.651 MB | 3.531e-3 s | ± 9.489e-3 s | — |

The compiled initial load includes native compilation: its median compiler time was 7.399e-1 s over seven fresh caches. Eleven subsequent cache hits loaded in 2.437e-3 ± 1.233e-3 s. Joblib load also used 11 repetitions.

| Batch rows | Compiled median | Joblib median | Compiled speedup | Compiled throughput | Joblib throughput |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.751e-6 s | 9.277e-4 s | 161.31x | 173,882 rows/s | 1,078 rows/s |
| 256 | 4.497e-4 s | 1.611e-3 s | 3.58x | 569,263 rows/s | 158,934 rows/s |
| 4,096 | 7.763e-3 s | 7.416e-3 s | 0.96x | 527,628 rows/s | 552,337 rows/s |
| 262,144 | 3.641e-1 s | 3.552e-1 s | 0.98x | 720,033 rows/s | 737,962 rows/s |

Warm inference used 21 repetitions per batch size. Compilation is a one-time cost that can be cached, while the compiled path's main benefit is low latency for small exchanges. The crossover occurs below 4,096 rows for this estimator and single-threaded configuration; large batches are effectively at parity with Joblib.

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

FoamNordic's internal accounting was identical on both platforms and in every run. MacBook process RSS remained near its large-exchange high-water mark, whereas Roihu returned from 130.164 MiB after the large exchanges to 50.160 MiB after the final small exchanges. Allocator-resident pages are not an ownership contract: on both systems the retained-workspace counter shows that the active execution context kept only the small 1,024-row workspace. All 301 exchanges per platform passed. Roihu job `1093451` ran on one `test` node for 9 seconds, including configuration and compilation.

## Runtime generation turnover

The MacBook generation stress published 256 immutable runtime generations with a 64 KiB payload in each of seven independent series. Every publication atomically replaced the current selector and invoked garbage collection while one generation remained pinned for the entire series and another pin moved every 16 publications. After releasing both pins, exactly the current and most recent generation survived as required by the retention policy.

| Platform | Series time | Publish latency | GC latency | Collected generations |
|---|---:|---:|---:|---:|
| MacBook | 3.471 ± 0.703 s | 3.089 ms | 5.498 ms | 254 of 256 |

All 1,792 publications preserved a complete current generation. Long-lived and rotating pins were never collected, released pins were collected, incomplete selectors were never observed and the temporary benchmark roots were removed after completion. These timings characterise the local filesystem rather than runtime execution performance; the purpose of the stress is lifecycle correctness. The runnable benchmark is `tools/benchmarks/generation_turnover.py`.
