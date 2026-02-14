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


def _ffmpeg_root_config(ffmpeg_root, rpath):
    """Build config from an FFmpeg root directory with include/ and lib/."""
    include_dirs = []
    library_dirs = []
    extra_link_args = []

    ffmpeg_root = Path(ffmpeg_root)
    include_dir = ffmpeg_root / "include"
    lib_dir = ffmpeg_root / "lib"
    if include_dir.exists():
        include_dirs.append(str(include_dir))
    if lib_dir.exists():
        library_dirs.append(str(lib_dir))
        extra_link_args.append(f"-Wl,-rpath,{rpath}")

    return {
        "include_dirs": include_dirs,
        "extra_compile_args": [],
        "libraries": list(FFMPEG_LIB_NAMES),
        "library_dirs": library_dirs,
        "extra_link_args": extra_link_args,
    }


def _patch_ffmpeg_rpath(lib_dir):
    """Set RPATH=$ORIGIN on all shared libraries so they find each other."""
    try:
        subprocess.check_output(["patchelf", "--version"], stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("WARNING: patchelf not found, skipping RPATH patching.")
        print("  Install patchelf: pip install patchelf")
        return

    for so in Path(lib_dir).glob("*.so*"):
        if so.is_symlink() or not so.is_file():
            continue
        try:
            subprocess.check_call(
                ["patchelf", "--set-rpath", "$ORIGIN", str(so)],
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            pass


def _fetch_ffmpeg():
    """Download pre-built FFmpeg to a local .ffmpeg directory, return its path."""
    project_root = Path(__file__).parent
    ffmpeg_dir = project_root / ".ffmpeg"
    lib_dir = ffmpeg_dir / "lib"

    # Already fetched?
    if lib_dir.exists() and any(lib_dir.glob("libavcodec*")):
        return ffmpeg_dir

    fetch_script = project_root / "scripts" / "fetch-ffmpeg.py"
    if not fetch_script.exists():
        return None

    print("Fetching pre-built FFmpeg libraries...")
    try:
        subprocess.check_call(
            [sys.executable, str(fetch_script), str(ffmpeg_dir)],
        )
    except subprocess.CalledProcessError as e:
        print(f"WARNING: Failed to fetch FFmpeg: {e}")
        return None

    if lib_dir.exists() and any(lib_dir.glob("libavcodec*")):
        _patch_ffmpeg_rpath(lib_dir)
        return ffmpeg_dir

    return None


def get_ffmpeg_config():
    """
    Get FFmpeg configuration for building.

    Priority:
    1. HUMECODEC_FFMPEG_ROOT environment variable (for wheel builds)
    2. Auto-fetch pre-built FFmpeg (for editable/dev installs)
    3. Fail with instructions
    """
    # Check for explicit FFmpeg root (used in wheel builds)
    ffmpeg_root = os.environ.get("HUMECODEC_FFMPEG_ROOT")
    if ffmpeg_root:
        ffmpeg_root = Path(ffmpeg_root)
        if ffmpeg_root.exists():
            print(f"Using FFmpeg from HUMECODEC_FFMPEG_ROOT: {ffmpeg_root}")
            return _ffmpeg_root_config(ffmpeg_root, "$ORIGIN/../humecodec.libs")

    # Auto-fetch pre-built FFmpeg
    ffmpeg_dir = _fetch_ffmpeg()
    if ffmpeg_dir:
        lib_dir = ffmpeg_dir / "lib"
        print(f"Using pre-built FFmpeg from: {ffmpeg_dir}")
        return _ffmpeg_root_config(ffmpeg_dir, str(lib_dir))

    print("ERROR: FFmpeg not found. Set HUMECODEC_FFMPEG_ROOT or ensure")
    print("  scripts/fetch-ffmpeg.py is available to download it automatically.")
    sys.exit(1)


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
