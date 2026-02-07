#!/usr/bin/env python3
"""
Fetch pre-built FFmpeg libraries for the current platform.

This script downloads pre-built FFmpeg binaries from PyAV's releases,
which provide well-tested builds for all major platforms.

Usage:
    python scripts/fetch-ffmpeg.py /path/to/destination

The script will:
1. Detect the current platform (Linux, macOS, Windows)
2. Download the appropriate FFmpeg tarball
3. Extract it to the destination directory
"""

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path


def get_platform():
    """Detect the current platform for FFmpeg binary selection."""
    system = platform.system()
    machine = platform.machine().lower()
    is_arm64 = machine in {"arm64", "aarch64"}

    if system == "Linux":
        # Check if glibc or musl
        libc = platform.libc_ver()[0]
        prefix = "manylinux-" if libc == "glibc" else "musllinux-"
        return prefix + ("aarch64" if is_arm64 else "x86_64")
    elif system == "Darwin":
        return "macos-arm64" if is_arm64 else "macos-x86_64"
    elif system == "Windows":
        return "windows-aarch64" if is_arm64 else "windows-x86_64"
    else:
        raise RuntimeError(f"Unsupported platform: {system} {machine}")


def download_file(url: str, dest: Path) -> None:
    """Download a file with progress indication."""
    logging.info(f"Downloading {url}")

    # Try curl first (handles redirects better)
    try:
        subprocess.check_call(
            ["curl", "--location", "--output", str(dest), "--silent", "--show-error", url],
            stdout=subprocess.DEVNULL if logging.root.level > logging.DEBUG else None
        )
        return
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Fall back to urllib
    def report_progress(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(100, block_num * block_size * 100 // total_size)
            sys.stdout.write(f"\r  Progress: {percent}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=report_progress)
    print()  # newline after progress


def extract_tarball(tarball: Path, dest: Path) -> None:
    """Extract a tarball to the destination directory."""
    logging.info(f"Extracting {tarball.name} to {dest}")

    # Try system tar first (faster, handles more formats)
    try:
        subprocess.check_call(
            ["tar", "-C", str(dest), "-xf", str(tarball)],
            stdout=subprocess.DEVNULL if logging.root.level > logging.DEBUG else None
        )
        return
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Fall back to Python tarfile
    with tarfile.open(tarball) as tf:
        tf.extractall(dest)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch pre-built FFmpeg libraries for wheel building"
    )
    parser.add_argument(
        "destination",
        help="Directory to extract FFmpeg to"
    )
    parser.add_argument(
        "--config-file",
        default=Path(__file__).parent / "ffmpeg.json",
        help="JSON config file with FFmpeg URL template"
    )
    parser.add_argument(
        "--cache-dir",
        default=Path(__file__).parent.parent / ".ffmpeg-cache",
        help="Directory to cache downloaded tarballs"
    )
    parser.add_argument(
        "--platform",
        default=None,
        help="Override platform detection (e.g., manylinux-x86_64)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    # Load config
    config_path = Path(args.config_file)
    if not config_path.exists():
        logging.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    # Determine platform
    plat = args.platform or get_platform()
    logging.info(f"Platform: {plat}")

    # Build URL
    url = config["url"].replace("{platform}", plat)

    # Ensure directories exist
    dest_dir = Path(args.destination)
    cache_dir = Path(args.cache_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Download if not cached
    tarball_name = url.split("/")[-1]
    tarball_path = cache_dir / tarball_name

    if not tarball_path.exists():
        download_file(url, tarball_path)
    else:
        logging.info(f"Using cached {tarball_name}")

    # Extract
    extract_tarball(tarball_path, dest_dir)

    # Verify extraction
    lib_dir = dest_dir / "lib"
    include_dir = dest_dir / "include"

    if not lib_dir.exists():
        logging.error(f"Extraction failed: {lib_dir} not found")
        sys.exit(1)

    logging.info(f"FFmpeg libraries installed to {dest_dir}")

    # List what we got
    if logging.root.level <= logging.DEBUG:
        for f in sorted(lib_dir.glob("*")):
            logging.debug(f"  {f.name}")


if __name__ == "__main__":
    main()
