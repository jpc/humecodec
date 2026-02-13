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
]

FFMPEG_LIB_NAMES = ["avcodec", "avformat", "avfilter", "avutil", "swresample", "swscale"]


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
    1. HUMECODEC_FFMPEG_ROOT environment variable (for wheel builds)
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
    ffmpeg_root = os.environ.get("HUMECODEC_FFMPEG_ROOT")
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
                extra_link_args.append(f"-Wl,-rpath,$ORIGIN/../humecodec.libs")
            print(f"Using FFmpeg from HUMECODEC_FFMPEG_ROOT: {ffmpeg_root}")
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
        print("  Set HUMECODEC_FFMPEG_ROOT or install FFmpeg via conda/pip.")

    return {
        "include_dirs": include_dirs,
        "extra_compile_args": extra_compile_args,
        "libraries": libraries,
        "library_dirs": library_dirs,
        "extra_link_args": extra_link_args,
    }


# ---------------------------------------------------------------------------
# Discover CUDA (without torch)
# ---------------------------------------------------------------------------

def get_cuda_config():
    """Detect CUDA toolkit for optional GPU support.

    Returns dict with include_dirs, library_dirs, libraries, and cuda_available.
    Detection priority:
    1. CUDA_HOME / CUDA_PATH environment variable
    2. nvcc on PATH
    3. Common install locations (/usr/local/cuda)
    """
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")

    if not cuda_home:
        # Try to find nvcc
        try:
            nvcc_path = subprocess.check_output(
                ["which", "nvcc"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            if nvcc_path:
                # nvcc is typically at <cuda_home>/bin/nvcc
                cuda_home = str(Path(nvcc_path).parent.parent)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

    if not cuda_home:
        # Check common locations
        for candidate in ["/usr/local/cuda", "/opt/cuda"]:
            if Path(candidate).exists():
                cuda_home = candidate
                break

    if not cuda_home or not Path(cuda_home).exists():
        return {
            "include_dirs": [],
            "library_dirs": [],
            "libraries": [],
            "cuda_available": False,
        }

    cuda_home = Path(cuda_home)
    include_dir = cuda_home / "include"
    lib_dir = cuda_home / "lib64"
    if not lib_dir.exists():
        lib_dir = cuda_home / "lib"

    # Verify cudart exists
    has_cudart = any(
        (lib_dir / name).exists()
        for name in ["libcudart.so", "libcudart.dylib", "cudart.lib"]
    )

    if not has_cudart:
        print(f"WARNING: CUDA found at {cuda_home} but cudart library not found")
        return {
            "include_dirs": [],
            "library_dirs": [],
            "libraries": [],
            "cuda_available": False,
        }

    print(f"Using CUDA from: {cuda_home}")
    return {
        "include_dirs": [str(include_dir)] if include_dir.exists() else [],
        "library_dirs": [str(lib_dir)] if lib_dir.exists() else [],
        "libraries": ["cudart"],
        "cuda_available": True,
    }


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
            "pybind11 is required to build humecodec. "
            "Please install pybind11 first: pip install pybind11"
        )


# ---------------------------------------------------------------------------
# Collect C++ sources
# ---------------------------------------------------------------------------

CSRC = Path(__file__).parent / "src" / "humecodec" / "csrc"


def collect_sources():
    """Collect all C++ source files."""
    setup_dir = Path(__file__).parent
    return sorted(str(p.relative_to(setup_dir)) for p in CSRC.rglob("*.cpp"))


# ---------------------------------------------------------------------------
# Build extension
# ---------------------------------------------------------------------------

ffmpeg_cfg = get_ffmpeg_config()
cuda_cfg = get_cuda_config()
pybind_cfg = get_pybind11_config()

# Combine all include directories
include_dirs = (
    [str(Path(__file__).parent / "src")]  # so "humecodec/csrc/..." includes work
    + ffmpeg_cfg["include_dirs"]
    + cuda_cfg["include_dirs"]
    + pybind_cfg["include_dirs"]
)

# Combine all library directories
library_dirs = ffmpeg_cfg["library_dirs"] + cuda_cfg["library_dirs"]

# Combine all libraries
libraries = ffmpeg_cfg["libraries"] + cuda_cfg["libraries"]

# Compile arguments (MSVC uses different flags from GCC/Clang)
if sys.platform == "win32":
    extra_compile_args = ["/std:c++17"] + ffmpeg_cfg["extra_compile_args"]
else:
    extra_compile_args = ["-std=c++17", "-fPIC"] + ffmpeg_cfg["extra_compile_args"]

# Link arguments
extra_link_args = ffmpeg_cfg["extra_link_args"]

# Define macros
define_macros = []
if cuda_cfg["cuda_available"]:
    define_macros.append(("USE_CUDA", None))

ext = Extension(
    name="humecodec._humecodec",
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
    options={"build_ext": {"parallel": True}},
)
