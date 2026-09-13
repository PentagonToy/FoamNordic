# Transport

The atomic-exchange benchmark sends three 400-cell float64 inputs and returns one 400-cell float64 output. Each result is the median and maximum observed deviation over seven warm runs of 100,000 blocking exchanges.

## Measurement contract

One exchange includes request announcement, every input tensor, request completion, every output tensor and response completion. The reported payload throughput is

$$
B=R\,S,
$$

where $R$ is the completed exchange rate and $S$ is the stated bidirectional payload per exchange. Protocol metadata is not counted in $S$.

| Variable held constant | Value |
| --- | --- |
| Input contract | Three float64 tensors of 400 scalar rows |
| Output contract | One float64 tensor of 400 scalar rows |
| Repetitions | Seven warm runs |
| Exchanges per normal run | 100,000 |

## Same-node channels

| Platform | Channel | Exchanges/s | Payload MiB/s |
|---|---|---:|---:|
| MacBook | UDS | 65,352 ± 2,654 | 797.7 ± 32.3 |
| MacBook | SHM, same process | 257,117 ± 15,236 | 3,138.6 ± 186.0 |
| MacBook | SHM, separate processes | 114,634 ± 3,000 | 1,399.3 ± 36.6 |
| Roihu CPU | UDS | 77,607 ± 3,511 | 947.4 ± 42.9 |
| Roihu CPU | SHM, same process | 260,248 ± 12,690 | 3,176.9 ± 154.9 |
| Roihu CPU | SHM, separate processes | 219,305 ± 29,860 | 2,677.1 ± 364.5 |

The separate-process path uses the production POSIX shared-memory, wake and completion paths. Both measurements used a Release build. Roihu job `1089143` requested one `small` node, 8 CPUs and 32 GiB.

## Cross-node channels

| Platform | Channel | Payload/exchange | Exchanges | Exchanges/s | Payload MiB/s |
|---|---|---:|---:|---:|---:|
| Roihu CPU, two nodes | TCP | 3,200 B | 100,000 | 4,264 ± 475 | 26.03 ± 2.90 |
| Roihu CPU, two nodes | TCP | 1 MiB | 100 | 55.113 ± 0.209 | 110.23 ± 0.42 |
| Roihu CPU, two nodes | UCX | 3,200 B | 100,000 | 80,520 ± 884 | 491.46 ± 5.39 |
| Roihu CPU, two nodes | UCX | 1 MiB | 100 | 4,334 ± 104 | 8,668.83 ± 207.72 |

Both measurements used two `medium` nodes, `rc5171` and `rc5172`. TCP job `1092472` took 3 minutes 24 seconds; UCX job `1093015` took 39 seconds, including configuration and compilation. UCX 1.20.0 used `ib0` with `UCX_TLS=rc,ud,sm,self`. UCX delivered 18.9 times the small-exchange rate and 78.6 times the large-payload throughput of TCP.

These are FoamNordic data-plane measurements rather than standalone `ucx_perftest` results.
