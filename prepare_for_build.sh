#!/bin/bash
# Apply Astral's pyg-lib build patches from the checked-out build repository.

set -euxo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <pyg-lib-version>"
    echo "Example: $0 0.8.0"
    exit 1
fi

PYG_LIB_VERSION=$1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${SCRIPT_DIR}/patches/${PYG_LIB_VERSION}"

if [ ! -d "${PATCH_DIR}" ]; then
    echo "Error: patches/${PYG_LIB_VERSION} directory does not exist"
    exit 1
fi

for patch_file in "${PATCH_DIR}"/*.patch; do
    patch -p1 -d "$(pwd)" -i "${patch_file}"
done
