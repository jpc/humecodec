#!/bin/bash
# Custom wheel repair script that fixes RPATHs for bundled libraries
set -e

WHEEL="$1"
DEST_DIR="$2"

if [ -z "$WHEEL" ] || [ -z "$DEST_DIR" ]; then
    echo "Usage: $0 <wheel> <dest_dir>"
    exit 1
fi

# First, run auditwheel repair with excludes
auditwheel repair -w "$DEST_DIR" "$WHEEL" \
    --exclude 'libcuda*' \
    --exclude 'libcublas*' \
    --exclude 'libcufft*' \
    --exclude 'libcusparse*' \
    --exclude 'libcudnn*' \
    --exclude 'libcupti*' \
    --exclude 'libnv*' \
    --exclude 'libnccl*' \
    --exclude 'libgomp*'

# Find the repaired wheel
REPAIRED_WHEEL=$(ls -1 "$DEST_DIR"/*.whl | head -1)
echo "Repaired wheel: $REPAIRED_WHEEL"

# Extract the wheel to patch bundled library RPATHs
TEMP_DIR=$(mktemp -d)
unzip -q "$REPAIRED_WHEEL" -d "$TEMP_DIR"

# Find the .libs directory (e.g., humecodec.libs)
LIBS_DIR=$(find "$TEMP_DIR" -name "*.libs" -type d | head -1)
if [ -n "$LIBS_DIR" ]; then
    echo "Patching RPATHs in: $LIBS_DIR"

    # Set RPATH for all .so files in the libs directory to $ORIGIN
    for lib in "$LIBS_DIR"/*.so*; do
        if [ -f "$lib" ]; then
            echo "Setting RPATH for: $(basename "$lib")"
            patchelf --set-rpath '$ORIGIN' "$lib" 2>/dev/null || true
        fi
    done
fi

# Repack the wheel using Python (zip may not be available in all containers)
rm "$REPAIRED_WHEEL"
python3 -c "
import os
import sys
import zipfile
from pathlib import Path

temp_dir = Path('$TEMP_DIR')
wheel_path = Path('$REPAIRED_WHEEL')

# Create wheel with proper structure
with zipfile.ZipFile(wheel_path, 'w', zipfile.ZIP_DEFLATED) as whl:
    for root, dirs, files in os.walk(temp_dir):
        for f in files:
            full_path = Path(root) / f
            rel_path = full_path.relative_to(temp_dir)
            whl.write(full_path, rel_path)

print(f'Done: {wheel_path}')
"

# Cleanup
rm -rf "$TEMP_DIR"

echo "Done: $REPAIRED_WHEEL"
