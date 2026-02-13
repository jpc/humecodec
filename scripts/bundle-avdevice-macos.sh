#!/bin/bash
# Bundle libavdevice into a macOS wheel as a dlopen-only fallback.
# libavdevice is not linked at compile time (to avoid ObjC class conflicts
# when the user already has FFmpeg installed), so delocate won't bundle it
# automatically.  This script injects it into the .dylibs/ directory of
# every wheel found in DEST_DIR.
set -e

DEST_DIR="$1"
FFMPEG_ROOT="${HUMECODEC_FFMPEG_ROOT:-/tmp/ffmpeg}"

if [ -z "$DEST_DIR" ]; then
    echo "Usage: $0 <dest_dir>"
    exit 1
fi

for whl in "$DEST_DIR"/*.whl; do
    [ -f "$whl" ] || continue
    echo "Injecting libavdevice into: $(basename "$whl")"

    TEMP_DIR=$(mktemp -d)
    unzip -q "$whl" -d "$TEMP_DIR"

    # Find or create the .dylibs directory
    DYLIBS_DIR=$(find "$TEMP_DIR" -name ".dylibs" -type d | head -1)
    if [ -z "$DYLIBS_DIR" ]; then
        # Create it inside the package directory
        PKG_DIR=$(find "$TEMP_DIR" -name "humecodec" -type d | head -1)
        DYLIBS_DIR="$PKG_DIR/.dylibs"
        mkdir -p "$DYLIBS_DIR"
    fi

    # Copy libavdevice dylibs
    for avdev in "$FFMPEG_ROOT"/lib/libavdevice*.dylib; do
        if [ -f "$avdev" ]; then
            echo "  Copying: $(basename "$avdev")"
            cp "$avdev" "$DYLIBS_DIR/"
        fi
    done

    # Repack
    rm "$whl"
    python3 -c "
import os, zipfile
from pathlib import Path
temp_dir = Path('$TEMP_DIR')
wheel_path = Path('$whl')
with zipfile.ZipFile(wheel_path, 'w', zipfile.ZIP_DEFLATED) as whl:
    for root, dirs, files in os.walk(temp_dir):
        for f in files:
            full_path = Path(root) / f
            rel_path = full_path.relative_to(temp_dir)
            whl.write(full_path, rel_path)
print(f'Done: {wheel_path}')
"
    rm -rf "$TEMP_DIR"
done
