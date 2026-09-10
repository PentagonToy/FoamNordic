# Out-of-tree OpenFOAM adapter

This is a complete equation-level adapter built against an installed FoamNordic SDK. It does not include or link files from the FoamNordic source tree. `externalNutFjord` is intentionally small: OpenFOAM owns the LES model and `nut` field, while the adapter performs one blocking `ClosureHook` call at `correctNut()`.

## Prerequisites

Install FoamNordic, activate the target OpenFOAM environment, and build its ABI-specific SDK once:

```console
python -m pip install foamnordic
module load openfoam/2512
foamnordic build
foamnordic doctor
```

On macOS, activate OpenFOAM with the site's normal shell command instead of the module command.

## Build outside FoamNordic

Copy this directory into the solver project and run:

```console
export FOAM_USER_LIBBIN="$PWD/lib"
mkdir -p "$FOAM_USER_LIBBIN"
wmake libso
```

`Make/options` resolves the exact headers and integration library through:

```console
foamnordic dir --include
foamnordic dir --openfoam-library
```

There is no `FOAMNORDIC_SOURCE` or build-tree path. The resulting `libexternalNutFjord` is specific to the currently loaded OpenFOAM ABI.

## Use

Longship loads the ABI-matched FoamNordic integration. Add the external library to `system/controlDict` without a platform-specific suffix:

```text
libs
(
    "libexternalNutFjord"
);
```

Select `externalNutFjord` and provide the same thin field contract used by an algebraic eddy-viscosity model:

```text
LESModel externalNutFjord;

externalNutFjordCoeffs
{
    foamNordicClosure
    {
        address          "unix:///tmp/external-nut.sock";
        sessionId        1;
        sharedMemory     true;

        inputs           (velocity_grad filter_width);
        inputExpressions ("grad(U)" delta);
        outputs          (nut);
        outputKeys       (eddy_viscosity);
    }
}
```

The external project declares the corresponding FNOM contract. Production projects let Longship start the compatible ModelHost and render the integration library, address, and other lifecycle-owned dictionary values.

The example duplicates only the thin equation call site so it can demonstrate the extension boundary. The installed FoamNordic library retains transport, packing, failure handling, and lifecycle, and invokes BindNordic for model loading, scaling, and execution.
