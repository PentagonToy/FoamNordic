# Runtime and model execution

These measurements cover work outside the CFD equations: field transforms, model-input preparation, inference, artefact handling and resident memory.

## Field transform

Two hundred in-place fused transforms were applied to one million three-component float64 cells. Seven warm runs were reduced by their median.

| Platform | Elapsed | Field throughput |
|---|---:|---:|
| MacBook | 0.42972 s (0.42830 ~ 0.45079) | 10,652 MiB/s (10,155 ~ 10,688) |
| Roihu CPU | 1.22934 s (1.21282 ~ 1.23170) | 3,724 MiB/s (3,717 ~ 3,774) |

The final field range was `[1, 7.31602]` in every run.

## Model-input packing

Two hundred packs converted and interleaved three one-million-cell float64 scalar fields into one float32 model tensor. The comparison loop performs an idealised two-pass cast and interleave; FoamNordic additionally validates the contract, shape and active range and updates reusable workspace metadata.

| Platform | Two-pass | FoamNordic pack | Speed ratio | Source throughput |
|---|---:|---:|---:|---:|
| MacBook | 0.12257 s | 0.16095 s | 0.762x | 28,442 MiB/s |
| Roihu CPU | 0.24974 s | 0.25064 s | 0.9996x | 18,264 MiB/s |

The MacBook ranges were 0.12139 ~ 0.12562 s and 0.15987 ~ 0.16383 s over seven runs. Roihu job `1090004` requested one `small` node, 8 CPUs and 32 GiB, bound the measurement to one core, discarded one warm-up and measured 21 runs. Its two-pass and FoamNordic ranges were 0.24400 ~ 0.27747 s and 0.24613 ~ 0.26550 s; their coefficients of variation were 3.94% and 2.38%. Every checksum was identical.

## Equinox model execution

An Equinox MLP with ten inputs, three 128-wide hidden layers and one output ran on 262,144 rows per call. Each platform completed one unmeasured compilation and warm-up call followed by 20 measured inference calls.

| Platform | Median inference | Observed range | Throughput | FNOM size |
|---|---:|---:|---:|---:|
| MacBook CPU | 2.931e-2 s | 2.882e-2 ~ 4.837e-2 s | 8,943,149 rows/s | 148.1 kB |
| Roihu GH200 | 5.706e-4 s | 5.642e-4 ~ 6.077e-4 s | 459,400,000 rows/s | 148.4 kB |

The MacBook used JAX 0.11.0 on `CpuDevice(id=0)` and had a 13.96% timing coefficient of variation. Roihu job `1089161` requested one GH200 and 16 CPUs on `gputest`; JAX 0.11.1 reported `CudaDevice(id=0)`. On both platforms, FoamNordic exported the model to FNOM, reloaded it through the resident Equinox evaluator and reproduced the direct JAX result. The small artefact-size difference reflects the platform environments. This is a model-execution microbenchmark, not solver throughput.

## Joblib and ONNX model execution

Roihu job `1091323` ran on one `test` node with 8 allocated CPUs and 32 GiB for 27 seconds. Both backends used one thread and the same fitted scikit-learn MLP with ten inputs, three 128-wide hidden layers and one output. FoamNordic embedded the original estimator as Joblib and the converted graph as ONNX; both reproduced the scikit-learn result within their declared tolerances.

| Backend | FNOM size | Median cold load | Observed range |
|---|---:|---:|---:|
| Joblib | 561.8 kB | 1.288e-3 s | 1.255e-3 ~ 1.271e-2 s |
| ONNX | 139.5 kB | 6.546e-4 s | 6.338e-4 ~ 6.211e-2 s |

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

| Backend | FNOM size | Median initial load | Observed range | Median cached load |
|---|---:|---:|---:|---:|
| Compiled | 1.244 MB | 7.433e-1 s | 7.232e-1 ~ 1.451 s | 2.437e-3 s |
| Joblib | 1.651 MB | 3.531e-3 s | 3.239e-3 ~ 1.302e-2 s | — |

The compiled initial load includes native compilation: its median compiler time was 7.399e-1 s over seven fresh caches. Eleven subsequent cache hits loaded in 2.351e-3 ~ 3.670e-3 s. Joblib load also used 11 repetitions.

| Batch rows | Compiled median | Joblib median | Compiled speedup | Compiled throughput | Joblib throughput |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.751e-6 s | 9.277e-4 s | 161.31x | 173,882 rows/s | 1,078 rows/s |
| 256 | 4.497e-4 s | 1.611e-3 s | 3.58x | 569,263 rows/s | 158,934 rows/s |
| 4,096 | 7.763e-3 s | 7.416e-3 s | 0.96x | 527,628 rows/s | 552,337 rows/s |
| 262,144 | 3.641e-1 s | 3.552e-1 s | 0.98x | 720,033 rows/s | 737,962 rows/s |

Warm inference used 21 repetitions per batch size. Compilation is a one-time cost that can be cached, while the compiled path's main benefit is low latency for small exchanges. The crossover occurs below 4,096 rows for this estimator and single-threaded configuration; large batches are effectively at parity with Joblib.

## Pending

- ModelHost peak RSS, retained buffers and generation turnover;
- Equinox cold load and artefact size.
