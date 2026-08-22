# Postprocess API

`fno.Postprocess` reads durable OpenFOAM results after a local or scheduled
run. It is deliberately separate from `run.observe()`: observations are
bounded live summaries, while postprocessing reads full stored fields without
affecting solver progress.

## Open a result or case

```python
post = fno.Postprocess.Case(result)

# Paths, fno.OpenFOAM.Case declarations, and Result objects are accepted.
same = fno.Postprocess.Case(result.case)
```

`result.postprocess` is the equivalent compact shortcut. `result.case`,
`result.logs`, and `result.artifacts` expose stable generated locations
without copying files into Python.

## Select one stored time

```python
latest_u = post.field("U")                 # time_idx=-1 by default
first_u = post.field("U", time_idx=0)
at_one = post.field("U", physical_time=1.0)
```

`time_idx` addresses the numerically sorted stored times and accepts negative
indices. `physical_time` selects the matching OpenFOAM physical time. They are
mutually exclusive; passing both is an error. `post.times` returns every
available physical time.

Reconstructed and `processor*` cases are supported. Uniform fields are
expanded to the mesh cell count, while scalar, vector, symmetric-tensor, and
tensor component shapes are retained. `foamlib` handles OpenFOAM decoding and
an ASCII internal-field fallback covers simple stored fields.

## Statistics

```python
statistics = post.statistics(
    ["U", "p", "nut"],
    time_idx=-1,
    verbose=True,
)
```

The result contains `min`, `max`, `mean`, `std`, and `rms`. Vector and tensor
fields use the Euclidean or Frobenius magnitude of each cell. The calculation
is always returned as ordinary dictionaries. `verbose=True` additionally
displays a compact Onsaemiro table; the default is quiet.

## Compare pure OpenFOAM and ML results

```python
metrics = fno.Postprocess.compare(
    reference=baseline_result,
    candidate=ml_result,
    fields=["U", "p", "nut"],
    physical_time=1.0,
    mesh="strict",
    verbose=True,
)
```

Each field reports `mae`, `rmse`, `max_abs`, and `relative_l2`. The default
`mesh="strict"` verifies reconstructed mesh files or matching decomposed mesh
and `cellProcAddressing` files before cell-wise comparison. `mesh="shape"` is
an explicit diagnostic mode for cases where only field shapes are known to
correspond; it cannot detect reordered cells.

Plotting remains ordinary NumPy, Matplotlib, PyVista, or another user-selected
tool. Postprocess owns reliable field access and numerical comparison rather
than a second plotting framework.
