import subprocess
import os
import sys
from pathlib import Path

from setuptools import setup, Extension

# ---------------------------------------------------------------------------
# Discover FFmpeg
# ---------------------------------------------------------------------------

FFMPEG_LIBS = [
    "libavcodec",
    "libavformat",
    "libavfilter",
    "libavutil",
    "libavdevice",
]

FFMPEG_LIB_NAMES = ["avcodec", "avformat", "avfilter", "avutil", "avdevice", "swresample", "swscale"]


def _pkg_config(flag: str, libs: list) -> list:
    """Query pkg-config for compiler/linker flags."""
    try:
        out = subprocess.check_output(
            ["pkg-config", flag] + libs, text=True
        ).strip()
        return out.split() if out else []
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []


def get_ffmpeg_config():
    """
    Get FFmpeg configuration for building.

    Priority:
    1. TORCHFFMPEG_FFMPEG_ROOT environment variable (for wheel builds)
    2. PKG_CONFIG_PATH environment (for vendored FFmpeg)
    3. pkg-config system lookup
    4. Conda/pip environment
    """
    include_dirs = []
    library_dirs = []
    libraries = list(FFMPEG_LIB_NAMES)
    extra_compile_args = []
    extra_link_args = []

    # Check for explicit FFmpeg root (used in wheel builds)
    ffmpeg_root = os.environ.get("TORCHFFMPEG_FFMPEG_ROOT")
    if ffmpeg_root:
        ffmpeg_root = Path(ffmpeg_root)
        if ffmpeg_root.exists():
            include_dir = ffmpeg_root / "include"
            lib_dir = ffmpeg_root / "lib"
            if include_dir.exists():
                include_dirs.append(str(include_dir))
            if lib_dir.exists():
                library_dirs.append(str(lib_dir))
                # Add rpath for the bundled libraries
                extra_link_args.append(f"-Wl,-rpath,$ORIGIN/../torchffmpeg.libs")
            print(f"Using FFmpeg from TORCHFFMPEG_FFMPEG_ROOT: {ffmpeg_root}")
            return {
                "include_dirs": include_dirs,
                "extra_compile_args": extra_compile_args,
                "libraries": libraries,
                "library_dirs": library_dirs,
                "extra_link_args": extra_link_args,
            }

    # Try pkg-config
    cflags = _pkg_config("--cflags", FFMPEG_LIBS)
    ldflags = _pkg_config("--libs", FFMPEG_LIBS)

    if cflags or ldflags:
        for f in cflags:
            if f.startswith("-I"):
                include_dirs.append(f[2:])
            else:
                extra_compile_args.append(f)

        libraries = []  # Reset - we'll get them from pkg-config
        for f in ldflags:
            if f.startswith("-l"):
                libraries.append(f[2:])
            elif f.startswith("-L"):
                library_dirs.append(f[2:])
            else:
                extra_link_args.append(f)

        if libraries:
            print(f"Using FFmpeg from pkg-config")
            return {
                "include_dirs": include_dirs,
                "extra_compile_args": extra_compile_args,
                "libraries": libraries,
                "library_dirs": library_dirs,
                "extra_link_args": extra_link_args,
            }

    # Fallback: try conda/pip environment
    env_prefix = sys.prefix
    env_lib = Path(env_prefix) / "lib"
    env_include = Path(env_prefix) / "include"

    # Check for FFmpeg in the environment
    if (env_lib / "libavcodec.so").exists() or (env_lib / "libavcodec.dylib").exists():
        library_dirs.append(str(env_lib))
        if env_include.exists():
            include_dirs.append(str(env_include))
        print(f"Using FFmpeg from Python environment: {env_prefix}")
    else:
        print("WARNING: FFmpeg not found. Build may fail.")
        print("  Set TORCHFFMPEG_FFMPEG_ROOT or install FFmpeg via conda/pip.")

    return {
        "include_dirs": include_dirs,
        "extra_compile_args": extra_compile_args,
        "libraries": libraries,
        "library_dirs": library_dirs,
        "extra_link_args": extra_link_args,
    }


# ---------------------------------------------------------------------------
# Discover PyTorch for build-time compilation
# ---------------------------------------------------------------------------

def get_torch_config():
    """Get PyTorch include directories and library paths."""
    try:
        import torch
        from torch.utils.cpp_extension import include_paths, library_paths

        include_dirs = include_paths()
        library_dirs = library_paths()

        # Add torch libraries including Python bindings
        libraries = ["torch", "torch_cpu", "torch_python", "c10"]
        if torch.cuda.is_available():
            libraries.extend(["torch_cuda", "c10_cuda"])

        return {
            "include_dirs": include_dirs,
            "library_dirs": library_dirs,
            "libraries": libraries,
            "cuda_available": torch.cuda.is_available(),
        }
    except ImportError:
        raise RuntimeError(
            "PyTorch is required to build torchffmpeg. "
            "Please install PyTorch first: pip install torch"
        )


# ---------------------------------------------------------------------------
# Discover pybind11
# ---------------------------------------------------------------------------

def get_pybind11_config():
    """Get pybind11 include directory."""
    try:
        import pybind11
        return {"include_dirs": [pybind11.get_include()]}
    except ImportError:
        raise RuntimeError(
            "pybind11 is required to build torchffmpeg. "
            "Please install pybind11 first: pip install pybind11"
        )


# ---------------------------------------------------------------------------
# Collect C++ sources
# ---------------------------------------------------------------------------

CSRC = Path(__file__).parent / "src" / "torchffmpeg" / "csrc"


def collect_sources():
    """Collect all C++ source files."""
    setup_dir = Path(__file__).parent
    return sorted(str(p.relative_to(setup_dir)) for p in CSRC.rglob("*.cpp"))


# ---------------------------------------------------------------------------
# Build extension
# ---------------------------------------------------------------------------

ffmpeg_cfg = get_ffmpeg_config()
torch_cfg = get_torch_config()
pybind_cfg = get_pybind11_config()

# Combine all include directories
include_dirs = (
    [str(Path(__file__).parent / "src")]  # so "torchffmpeg/csrc/..." includes work
    + ffmpeg_cfg["include_dirs"]
    + torch_cfg["include_dirs"]
    + pybind_cfg["include_dirs"]
)

# Combine all library directories
library_dirs = ffmpeg_cfg["library_dirs"] + torch_cfg["library_dirs"]

# Combine all libraries
libraries = ffmpeg_cfg["libraries"] + torch_cfg["libraries"]

# Compile arguments
extra_compile_args = ["-std=c++17", "-fPIC"] + ffmpeg_cfg["extra_compile_args"]

# Link arguments
extra_link_args = ffmpeg_cfg["extra_link_args"]

# Add rpath for torch libraries (they're not bundled)
for lib_dir in torch_cfg["library_dirs"]:
    extra_link_args.append(f"-Wl,-rpath,{lib_dir}")

# Define macros
define_macros = []
if torch_cfg["cuda_available"]:
    define_macros.append(("USE_CUDA", None))

ext = Extension(
    name="torchffmpeg._torchffmpeg",
    sources=collect_sources(),
    include_dirs=include_dirs,
    libraries=libraries,
    library_dirs=library_dirs,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    define_macros=define_macros,
    language="c++",
)

setup(
    ext_modules=[ext],
)
