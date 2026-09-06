#!/usr/bin/env bash

set -u

example_dir=$(cd "$(dirname "$0")" && pwd)
work_dir=$(mktemp -d "${TMPDIR:-/tmp}/foamnordic-external-adapter.XXXXXX")

cleanup()
{
    rm -rf -- "$work_dir"
}

trap cleanup EXIT INT TERM

command -v foamnordic >/dev/null || {
    echo "foamnordic is not available" >&2
    exit 1
}
command -v wmake >/dev/null || {
    echo "wmake is not available; activate OpenFOAM" >&2
    exit 1
}

cp -R "$example_dir" "$work_dir/adapter"
export FOAM_USER_LIBBIN="$work_dir/lib"
mkdir -p "$FOAM_USER_LIBBIN"

if grep -R -E 'FOAMNORDIC_(SOURCE|BUILD)' "$work_dir/adapter/Make" >/dev/null; then
    echo "external adapter unexpectedly references a FoamNordic source tree" >&2
    exit 1
fi

cd "$work_dir/adapter"
if ! wmake libso; then
    echo "external adapter compilation failed" >&2
    exit 1
fi

library=$(find "$FOAM_USER_LIBBIN" -maxdepth 1 -type f \
    \( -name 'libexternalNutFjord.so' -o -name 'libexternalNutFjord.dylib' \) \
    -print | head -1)

if [ -z "$library" ]; then
    echo "external adapter library was not produced" >&2
    exit 1
fi
if ! strings "$library" | grep -q externalNutFjord; then
    echo "external adapter registration was not linked" >&2
    exit 1
fi

echo "[FoamNord] Installed-SDK external adapter: PASS"
echo "[FoamNord] Library: $library"
