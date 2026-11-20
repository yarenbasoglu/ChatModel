import ctypes
import importlib.util
import importlib.machinery
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def verify_backend() -> None:
    from gensim.models import word2vec_inner

    if word2vec_inner.FAST_VERSION == 2:
        return
    function_name = "our_dot_double" if word2vec_inner.FAST_VERSION == 0 else "our_dot_float"
    capsule = word2vec_inner.__pyx_capi__[function_name]
    get_name = ctypes.pythonapi.PyCapsule_GetName
    get_name.argtypes = [ctypes.py_object]
    get_name.restype = ctypes.c_char_p
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    get_pointer.restype = ctypes.c_void_p
    address = get_pointer(capsule, get_name(capsule))
    integer_pointer = ctypes.POINTER(ctypes.c_int)
    float_pointer = ctypes.POINTER(ctypes.c_float)
    signature = ctypes.CFUNCTYPE(ctypes.c_float, integer_pointer, float_pointer, integer_pointer, float_pointer, integer_pointer)
    dot = signature(address)
    size = ctypes.c_int(1)
    positive = (ctypes.c_float * 1)(1.0)
    negative = (ctypes.c_float * 1)(-1.0)
    result = dot(ctypes.byref(size), positive, ctypes.byref(size), negative, ctypes.byref(size))
    if result != -1.0:
        raise RuntimeError("Gensim BLAS regression detected. Run python -m scripts.fix_gensim_blas before training.")


def repair() -> None:
    probe = subprocess.run([sys.executable, "-m", "scripts.fix_gensim_blas", "--check"], capture_output=True)
    if probe.returncode == 0:
        print("Gensim BLAS regression check already passes.", flush=True)
        return
    import numpy as np
    from Cython.Build import cythonize
    from setuptools import Distribution, Extension

    package = Path(importlib.util.find_spec("gensim").origin).parent
    installed = package / "models" / ("word2vec_inner" + importlib.machinery.EXTENSION_SUFFIXES[0])
    with tempfile.TemporaryDirectory(prefix="chatmodel-gensim-") as temporary_directory:
        temporary = Path(temporary_directory)
        source_directory = temporary / "gensim" / "models"
        source_directory.mkdir(parents=True)
        for suffix in (".pyx", ".pxd"):
            shutil.copy2(package / "models" / f"word2vec_inner{suffix}", source_directory)
        declarations = source_directory / "word2vec_inner.pxd"
        source = declarations.read_text(encoding="utf-8")
        if source.count("except -1 nogil") != 2:
            raise RuntimeError("Unexpected Gensim source. Review upstream issue #3665 before applying this patch.")
        declarations.write_text(source.replace("except -1 nogil", "noexcept nogil"), encoding="utf-8")
        extension = Extension(
            "gensim.models.word2vec_inner", [str(source_directory / "word2vec_inner.pyx")],
            include_dirs=[np.get_include(), str(package / "models")],
        )
        distribution = Distribution({"ext_modules": cythonize([extension], include_path=[str(temporary)], quiet=True)})
        command = distribution.get_command_obj("build_ext")
        command.build_lib = str(temporary / "lib")
        command.build_temp = str(temporary / "build")
        distribution.run_command("build_ext")
        rebuilt = Path(command.get_ext_fullpath("gensim.models.word2vec_inner"))
        backup = temporary / installed.name
        shutil.copy2(installed, backup)
        replacement = installed.with_suffix(installed.suffix + ".new")
        shutil.copy2(rebuilt, replacement)
        replacement.replace(installed)
        try:
            subprocess.run([sys.executable, "-m", "scripts.fix_gensim_blas", "--check"], check=True)
        except subprocess.CalledProcessError:
            shutil.copy2(backup, replacement)
            replacement.replace(installed)
            raise
    print("Gensim BLAS -1.0 regression repaired and verified.", flush=True)


if __name__ == "__main__":
    if "--check" in sys.argv:
        verify_backend()
    else:
        repair()
