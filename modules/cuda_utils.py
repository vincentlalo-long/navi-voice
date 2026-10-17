"""
CUDA & Tensor Library Environment Initializer.
Pre-loads nvidia site-packages shared libraries (cuBLAS, cuDNN, etc.) so that
CTranslate2, Faster-Whisper, and PyTorch find libcublas.so.12 seamlessly across all environments.
"""

import os
import sys
import glob
import ctypes
from pathlib import Path

_INITIALIZED = False


def setup_cuda_env():
    """Discover and preload nvidia shared libraries from virtualenv site-packages."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True

    try:
        # Search relative to virtualenv or current file
        module_dir = Path(__file__).resolve().parent
        voice_root = module_dir.parent
        venv_dirs = [
            voice_root / ".venv",
            Path(sys.prefix),
        ]

        lib_dirs = []
        for venv in venv_dirs:
            if venv.is_dir():
                found = list(venv.glob("lib*/python*/site-packages/nvidia/*/lib"))
                lib_dirs.extend([str(p) for p in found if p.is_dir()])

        # Remove duplicates
        unique_dirs = list(dict.fromkeys(lib_dirs))

        # 1. Update LD_LIBRARY_PATH in os.environ
        cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
        new_paths = [d for d in unique_dirs if d not in cur_ld]
        if new_paths:
            os.environ["LD_LIBRARY_PATH"] = ":".join(new_paths) + (":" + cur_ld if cur_ld else "")

        # 2. Pre-load specific essential libraries for CTranslate2 / Faster-Whisper
        # Only load cublas, cublasLt, and cudnn. EXCLUDE nvblas to avoid CPU BLAS crash.
        target_prefixes = ("libcublas.so", "libcublasLt.so", "libcudnn.so")
        for d in unique_dirs:
            p_dir = Path(d)
            for so_file in sorted(p_dir.glob("*.so*")):
                name = so_file.name
                if "nvblas" in name:
                    continue
                if any(name.startswith(p) for p in target_prefixes):
                    try:
                        ctypes.CDLL(str(so_file), mode=ctypes.RTLD_GLOBAL)
                    except Exception:
                        pass
    except Exception:
        pass


# Automatically execute setup on import
setup_cuda_env()
