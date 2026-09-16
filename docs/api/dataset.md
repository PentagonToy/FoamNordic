# Dataset API

`Dataset` converts stored OpenFOAM internal fields into one lossless, indexed `dataset.fnod` container for machine-learning and analysis. It reads reconstructed cases directly and processor cases through `cellProcAddressing` without writing reconstructed time directories.

## `Dataset.parse`

```python
fno.Dataset.parse(
    *,
    case_dir,
    fields,
    times=None,
    coordinates=True,
    derive_missing=True,
    output_dir=None,
    note=None,
    n_jobs=None,
    progress=False,
    overwrite=False,
)
```

```python
dataset = fno.Dataset.parse(
    case_dir=CASE_DIR,
    fields=["U", "gradU", "delta", "nut"],
    times=None,
    coordinates=True,
    derive_missing=True,
    output_dir=None,
    note=None,
    n_jobs=None,
    progress=False,
    overwrite=False,
)
```

| Parameter | Default | Accepted values | Meaning |
| --- | --- | --- | --- |
| `case_dir` | required | `str`, `os.PathLike` | Reconstructed or decomposed OpenFOAM case. |
| `fields` | required | Non-empty field-name sequence | Internal volume fields to store. |
| `times` | `None` | `None`, time sequence, two-value inclusive range | Stored physical times to retain. `None` retains every available time. |
| `coordinates` | `True` | `bool` | Store cell-centre `x`, `y`, and `z` columns. Set `False` only when spatial coordinates are intentionally unnecessary. |
| `derive_missing` | `True` | `bool` | Build missing `grad*` fields and `C` in an isolated temporary case when OpenFOAM can derive them. |
| `output_dir` | `None` | `str`, `os.PathLike`, `None` | Destination directory. `None` writes `case_dir / "dataset" / "dataset.fnod"`. |
| `note` | `None` | `str`, `None` | Short user note retained in the container metadata. |
| `n_jobs` | `None` | `-1`, positive `int`, `None` | Worker count for bounded FNOD compression. `None` uses one worker; `-1` selects a conservative automatic bound. |
| `progress` | `False` | `bool` | Render an Onsaemiro progress bar while fields are stored. |
| `overwrite` | `False` | `bool` | Replace an existing `dataset.fnod`. |

The public result is one logical table with `time`, `cell`, cell-centre `x`, `y`, `z`, and the requested field columns. Vector and tensor fields retain their component shape rather than being silently flattened. FNOD preserves OpenFOAM's stored floating-point dtype; it never silently converts `float64` fields to `float32`.

## `Dataset.read` and `Dataset.load`

```python
dataset = fno.Dataset.read(path=DATASET_DIR / "dataset.fnod")

table = dataset.load(
    fields=["gradU", "delta", "nut"],
    times=(0.1, 0.5),
)

X = np.column_stack([table["gradU"], table["delta"]])
y = table["nut"]
```

`Dataset.read(path=...)` opens the FNOD index. `Dataset.load(fields=None, times=None)` inflates selected field-time chunks and returns a structured NumPy array. `table["time"]`, `table["cell"]`, and `table["x"]`, `table["y"]`, `table["z"]` identify each row; field columns retain native component shapes.

The `Dataset` object exposes `path`, `fields`, `times`, and `note` for lightweight inspection.

## Storage contract

`dataset.fnod` is a FoamNordic Data container, not an executable FNOM artefact. It contains losslessly compressed column chunks and a footer index. Metadata contains the schema version, creation time, optional note, selected times, and field storage contracts.

With `derive_missing=True`, FoamNordic derives missing `gradX` fields through OpenFOAM `grad(X)`, missing `C` through `writeCellCentres`, and missing `delta` through a configured `controlDict` function object named `delta`. When no such function is configured, LES cases using `cubeRootVol` derive `delta = deltaCoeff cbrt(V)` through `writeCellVolumes`. Deriving a decomposed field requires one Slurm task per `processor*` directory; `n_jobs` only controls bounded FNOD compression. `n_jobs=-1` resolves to at most four workers for a reconstructed case and two for a decomposed case, further bounded by available field-time work and local CPU count. Root fields define global ordering when they are readable; otherwise processor fields are reconstructed with `processor*/constant/polyMesh/cellProcAddressing` and validated for one value per global cell. Temporary staging directories and incomplete FNOD files are removed on normal completion and handled failures.
