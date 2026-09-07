# Transport

The atomic-exchange benchmark sends three 400-cell float64 inputs and returns one 400-cell float64 output. Each result is the median of seven warm runs of 100,000 blocking exchanges; parentheses contain the observed range.

## Same-node channels

| Platform | Channel | Exchanges/s | Payload MiB/s |
|---|---|---:|---:|
| MacBook | UDS | 65,352 (62,698 ~ 67,739) | 797.7 (765.4 ~ 826.9) |
| MacBook | SHM, same process | 257,117 (249,300 ~ 272,353) | 3,138.6 (3,043.2 ~ 3,324.6) |
| MacBook | SHM, separate processes | 114,634 (111,634 ~ 116,805) | 1,399.3 (1,362.7 ~ 1,425.8) |
| Roihu CPU | UDS | 77,607 (74,096 ~ 79,188) | 947.4 (904.5 ~ 966.7) |
| Roihu CPU | SHM, same process | 260,248 (251,199 ~ 272,938) | 3,176.9 (3,066.4 ~ 3,331.8) |
| Roihu CPU | SHM, separate processes | 219,305 (189,445 ~ 228,056) | 2,677.1 (2,312.6 ~ 2,783.9) |

The separate-process path exercises the production POSIX shared-memory mapping, wake path, atomic completion frame and receive-side ownership lifetime.

Both measurements used runtime commit `89cf240` and a Release build. Roihu job `1089143` requested one `small` node, 8 CPUs and 32 GiB. All integrity checks passed; the wider separate-process range is retained in the table.

## Cross-node channels

Post-refactor TCP and UCX results are pending. They will be measured in one controlled two-node `medium` allocation. Passing requires the reported data plane, payload integrity, exchange identity and clean shutdown to agree; a loopback or `ucx_perftest` result is not a FoamNordic transport benchmark.
