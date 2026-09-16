#!/usr/bin/env bash

set -euo pipefail

usage()
{
    echo "Usage: $0 --check|--apply" >&2
}

if [[ $# -ne 1 || ( "$1" != "--check" && "$1" != "--apply" ) ]]
then
    usage
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository="$(cd "${script_dir}/../.." && pwd)"
style="${script_dir}/openfoam.clang-format"

if command -v clang-format >/dev/null 2>&1
then
    formatter="$(command -v clang-format)"
elif [[ -x /Library/Developer/CommandLineTools/usr/bin/clang-format ]]
then
    formatter=/Library/Developer/CommandLineTools/usr/bin/clang-format
else
    echo "clang-format is required to format OpenFOAM-facing sources." >&2
    exit 1
fi

files=()
while IFS= read -r entry
do
    files+=("${entry}")
done < <(
    find \
        "${repository}/src/foamnordic/openfoam" \
        "${repository}/examples/openfoam" \
        "${repository}/tools/openfoam" \
        -type f \( -name '*.C' -o -name '*.H' \) \
        -print | LC_ALL=C sort
)

if [[ ${#files[@]} -eq 0 ]]
then
    echo "No OpenFOAM-facing sources found." >&2
    exit 1
fi

if [[ "$1" == "--apply" ]]
then
    "${formatter}" -i -style="file:${style}" "${files[@]}"
else
    "${formatter}" --dry-run --Werror -style="file:${style}" "${files[@]}"
fi
