"""Manages CMake."""

from __future__ import annotations

import functools
import multiprocessing
import os
import re
import shutil
import sys
import sysconfig
from pathlib import Path
from subprocess import CalledProcessError, check_call, check_output, DEVNULL
from typing import cast, IO, Optional, Union

try:
    from packaging.version import Version
except ImportError:
    try:
        from setuptools.dist import Version  # type: ignore[attr-defined,no-redef]
    except ImportError:
        from distutils.version import (  # type: ignore[assignment,no-redef]
            LooseVersion as Version,
        )

CMakeValue = Optional[Union[bool, str]]

# Print to stderr
eprint = functools.partial(print, file=sys.stderr, flush=True)

# Ninja
# Use ninja if it is on the PATH. Previous version of PyTorch required the
# ninja python package, but we no longer use it, so we do not have to import it
USE_NINJA = False

class CMake:
    def __init__(self, build_dir: str = "build") -> None:
        self._cmake_command = "cmake"
        self.build_dir = build_dir
        self._cmake_cache_file = os.path.join(self.build_dir, "CMakeCache.txt")
        self._ninja_build_file = os.path.join(self.build_dir, "build.ninja")

    def run(self, args: list[str]) -> None:
        """Executes cmake with arguments and an environment."""

        command = [self._cmake_command] + args
        eprint(" ".join(command))
        try:
            check_call(command, cwd=self.build_dir)
        except (CalledProcessError, KeyboardInterrupt):
            sys.exit(1)

    @staticmethod
    def defines(args: list[str], **kwargs: CMakeValue) -> None:
        """Adds definitions to a cmake argument list."""
        for key, value in sorted(kwargs.items()):
            if value is not None:
                args.append(f"-D{key}={value}")

    @staticmethod
    def convert_cmake_value_to_python_value(
        cmake_value: str, cmake_type: str
    ) -> CMakeValue:
        r"""Convert a CMake value in a string form to a Python value.

        Args:
        cmake_value (string): The CMake value in a string form (e.g., "ON", "OFF", "1").
        cmake_type (string): The CMake type of :attr:`cmake_value`.

        Returns:
        A Python value corresponding to :attr:`cmake_value` with type :attr:`cmake_type`.
        """

        cmake_type = cmake_type.upper()
        up_val = cmake_value.upper()
        if cmake_type == "BOOL":
            # https://cmake.org/cmake/help/latest/manual/cmake-generator-expressions.7.html#genex:BOOL
            return not (
                up_val in ("FALSE", "OFF", "N", "NO", "0", "", "NOTFOUND")
                or up_val.endswith("-NOTFOUND")
            )
        elif cmake_type == "FILEPATH":
            if up_val.endswith("-NOTFOUND"):
                return None
            else:
                return cmake_value
        else:  # Directly return the cmake_value.
            return cmake_value

    @staticmethod
    def get_cmake_cache_variables_from_file(
        cmake_cache_file: IO[str],
    ) -> dict[str, CMakeValue]:
        r"""Gets values in CMakeCache.txt into a dictionary.

        Args:
        cmake_cache_file: A CMakeCache.txt file object.
        Returns:
        dict: A ``dict`` containing the value of cached CMake variables.
        """

        results = {}
        for i, line in enumerate(cmake_cache_file, 1):
            line = line.strip()
            if not line or line.startswith(("#", "//")):
                # Blank or comment line, skip
                continue

            # Almost any character can be part of variable name and value. As a practical matter, we assume the type must be
            # valid if it were a C variable name. It should match the following kinds of strings:
            #
            #   USE_CUDA:BOOL=ON
            #   "USE_CUDA":BOOL=ON
            #   USE_CUDA=ON
            #   USE_CUDA:=ON
            #   Intel(R) MKL-DNN_SOURCE_DIR:STATIC=/path/to/pytorch/third_party/ideep/mkl-dnn
            #   "OpenMP_COMPILE_RESULT_CXX_openmp:experimental":INTERNAL=FALSE
            matched = re.match(
                r'("?)(.+?)\1(?::\s*([a-zA-Z_-][a-zA-Z0-9_-]*)?)?\s*=\s*(.*)', line
            )
            if matched is None:  # Illegal line
                raise ValueError(f"Unexpected line {i} in {repr(cmake_cache_file)}: {line}")
            _, variable, type_, value = matched.groups()
            if type_ is None:
                type_ = ""
            if type_.upper() in ("INTERNAL", "STATIC"):
                # CMake internal variable, do not touch
                continue
            results[variable] = CMake.convert_cmake_value_to_python_value(value, type_)

        return results    

    def get_cmake_cache_variables(self) -> dict[str, CMakeValue]:
        r"""Gets values in CMakeCache.txt into a dictionary.
        Returns:
          dict: A ``dict`` containing the value of cached CMake variables.
        """
        with open(self._cmake_cache_file) as f:
            return CMake.get_cmake_cache_variables_from_file(f)

    def generate(
        self,
        version: str | None,
        build_python: bool,
        build_test: bool,
        rerun: bool,
    ) -> None:
        """Runs cmake to generate native build files."""

        if rerun and os.path.isfile(self._cmake_cache_file):
            os.remove(self._cmake_cache_file)

        cmake_cache_file_available = os.path.exists(self._cmake_cache_file)
        if cmake_cache_file_available:
            cmake_cache_variables = self.get_cmake_cache_variables()

        if cmake_cache_file_available and (
            not USE_NINJA or os.path.exists(self._ninja_build_file)
        ):
            eprint("Everything's in place. Do not rerun cmake generate.")
            return
        
        args = []
        if USE_NINJA:
            # Avoid conflicts in '-G' and the `CMAKE_GENERATOR`
            os.environ["CMAKE_GENERATOR"] = "Ninja"
            args.append("-GNinja")

        base_dir = str(Path(__file__).absolute().parents[2])
        install_dir = os.path.join(base_dir, "torch")

        os.makedirs(install_dir, exist_ok=True)
        os.makedirs(self.build_dir, exist_ok=True)

        # Store build options that are directly stored in environment variables
        build_options: dict[str, CMakeValue] = {}

        # Build options that do not start with "BUILD_", "USE_", or "CMAKE_" and are directly controlled by env vars.
        # This is a dict that maps environment variables to the corresponding variable name in CMake.
        additional_options = {
            # Key: environment variable name. Value: Corresponding variable name to be passed to CMake. If you are
            # adding a new build option to this block: Consider making these two names identical and adding this option
            # in the block below.
            "CUDNN_LIB_DIR": "CUDNN_LIBRARY",
            "USE_CUDA_STATIC_LINK": "CAFFE2_STATIC_LINK_CUDA",
        }
        additional_options.update(
            {
                # Build options that have the same environment variable name and CMake variable name and that do not start
                # with "BUILD_", "USE_", or "CMAKE_". If you are adding a new build option, also make sure you add it to
                # CMakeLists.txt.
                var: var
                for var in (
                    "UBSAN_FLAGS",
                    "BLAS",
                    "WITH_BLAS",
                    "CUDA_HOST_COMPILER",
                    "CUDA_NVCC_EXECUTABLE",
                    "CUDA_SEPARABLE_COMPILATION",
                    "CUDNN_LIBRARY",
                    "CUDNN_INCLUDE_DIR",
                    "CUDNN_ROOT",
                    "EXPERIMENTAL_SINGLE_THREAD_POOL",
                    "INSTALL_TEST",
                    "JAVA_HOME",
                    "INTEL_MKL_DIR",
                    "INTEL_OMP_DIR",
                    "MKL_THREADING",
                    "MKLDNN_CPU_RUNTIME",
                    "MSVC_Z7_OVERRIDE",
                    "CAFFE2_USE_MSVC_STATIC_RUNTIME",
                    "Numa_INCLUDE_DIR",
                    "Numa_LIBRARIES",
                    "ONNX_ML",
                    "ONNX_NAMESPACE",
                    "ATEN_THREADING",
                    "WERROR",
                    "OPENSSL_ROOT_DIR",
                    "STATIC_DISPATCH_BACKEND",
                    "SELECTED_OP_LIST",
                    "TORCH_CUDA_ARCH_LIST",
                    "TORCH_XPU_ARCH_LIST",
                    "TRACING_BASED",
                    "PYTHON_LIB_REL_PATH",
                )
            }
        )

        # Aliases which are lower priority than their canonical option
        low_priority_aliases = {
            "CUDA_HOST_COMPILER": "CMAKE_CUDA_HOST_COMPILER",
            "CUDAHOSTCXX": "CUDA_HOST_COMPILER",
            "CMAKE_CUDA_HOST_COMPILER": "CUDA_HOST_COMPILER",
            "CMAKE_CUDA_COMPILER": "CUDA_NVCC_EXECUTABLE",
            "CUDACXX": "CUDA_NVCC_EXECUTABLE",
        }
        for var, val in os.environ.items():
            # We currently pass over all environment variables that start with "BUILD_", "USE_", and "CMAKE_". This is
            # because we currently have no reliable way to get the list of all build options we have specified in
            # CMakeLists.txt. (`cmake -L` won't print dependent options when the dependency condition is not met.) We
            # will possibly change this in the future by parsing CMakeLists.txt ourselves (then additional_options would
            # also not be needed to be specified here).
            true_var = additional_options.get(var)
            if true_var is not None:
                build_options[true_var] = val
            elif var.startswith(("BUILD_", "USE_", "CMAKE_")) or var.endswith(
                ("EXITCODE", "EXITCODE__TRYRUN_OUTPUT")
            ):
                build_options[var] = val

            if var in low_priority_aliases:
                key = low_priority_aliases[var]
                if key not in build_options:
                    build_options[key] = val

        # The default value cannot be easily obtained in CMakeLists.txt. We set it here.
        py_lib_path = sysconfig.get_path("purelib")
        cmake_prefix_path = build_options.get("CMAKE_PREFIX_PATH", None)
        if cmake_prefix_path:
            build_options["CMAKE_PREFIX_PATH"] = (
                py_lib_path + ";" + cast(str, cmake_prefix_path)
            )
        else:
            build_options["CMAKE_PREFIX_PATH"] = py_lib_path

        # Some options must be post-processed. Ideally, this list will be shrunk to only one or two options in the
        # future, as CMake can detect many of these libraries pretty comfortably. We have them here for now before CMake
        # integration is completed. They appear here not in the CMake.defines call below because they start with either
        # "BUILD_" or "USE_" and must be overwritten here.
        use_numpy = True
        build_options.update(
            {
                # Note: Do not add new build options to this dict if it is directly read from environment variable -- you
                # only need to add one in `CMakeLists.txt`. All build options that start with "BUILD_", "USE_", or "CMAKE_"
                # are automatically passed to CMake; For other options you can add to additional_options above.
                "BUILD_PYTHON": build_python,
                "BUILD_TEST": build_test,
                # Most library detection should go to CMake script, except this one, which Python can do a much better job
                # due to NumPy's inherent Pythonic nature.
                "USE_NUMPY": use_numpy,
            }
        )

        # Detect build dependencies from python lib path (in order to set *_HOME variables)
        # NVSHMEM
        nvshmem_py_dir = py_lib_path + "/nvidia/nvshmem"
        if os.path.exists(nvshmem_py_dir):
            build_options["NVSHMEM_PY_DIR"] = nvshmem_py_dir

        # Options starting with CMAKE_
        cmake__options = {
            "CMAKE_INSTALL_PREFIX": install_dir,
        }

        # We set some CMAKE_* options in our Python build code instead of relying on the user's direct settings. Emit an
        # error if the user also attempts to set these CMAKE options directly.
        specified_cmake__options = set(build_options).intersection(cmake__options)
        if len(specified_cmake__options) > 0:
            eprint(
                ", ".join(specified_cmake__options)
                + " should not be specified in the environment variable. They are directly set by PyTorch build script."
            )
            sys.exit(1)
        build_options.update(cmake__options)

        if use_numpy:
            try:
                # This helps CMake find the correct include directory for NumPy
                # This is especially useful in cross compiled environments
                import numpy

                Python_NumPy_INCLUDE_DIR = numpy.get_include()
                build_options.update(
                    dict(Python_NumPy_INCLUDE_DIR=Python_NumPy_INCLUDE_DIR)
                )
            except ImportError:
                # use_numpy is just a hint.... so we can fail silently here
                pass

        CMake.defines(
            args,
            Python_EXECUTABLE=sys.executable,
            TORCH_BUILD_VERSION=version,
            **build_options,
        )

        # According to the CMake manual, we should pass the arguments first,
        # and put the directory as the last element. Otherwise, these flags
        # may not be passed correctly.
        # Reference:
        # 1. https://cmake.org/cmake/help/latest/manual/cmake.1.html#synopsis
        # 2. https://stackoverflow.com/a/27169347
        args.append(base_dir)
        eprint("cmake generate args: ", args)
        # args: '-DBUILD_PYTHON=True -DBUILD_TEST=True -DCMAKE_INSTALL_PREFIX=/home/user/pytorch/torch -DCMAKE_PREFIX_PATH=/home/user/anaconda3/lib/python3.13/site-packages;/home/user/anaconda3/envs/torch-dev: -DPython_EXECUTABLE=/home/user/anaconda3/bin/python3.13 -DPython_NumPy_INCLUDE_DIR=/home/user/anaconda3/lib/python3.13/site-packages/numpy/_core/include -DTORCH_BUILD_VERSION=2.10.0a0 -DUSE_NUMPY=True /home/user/pytorch'
        eprint("cmake generate envs: ", os.environ)
        self.run(args)

    def build(self) -> None:
        """Runs cmake to build binaries."""
        build_args = "--build . --target install --config Release".split() 

        if not USE_NINJA:
            max_jobs = str(multiprocessing.cpu_count())
            build_args += ["-j", max_jobs]
        self.run(build_args)

    def clear_cache(self) -> None:
        """Clears the CMake cache."""
        if os.path.isfile(self._cmake_cache_file):
            os.remove(self._cmake_cache_file)
        if os.path.isfile(self._ninja_build_file):
            os.remove(self._ninja_build_file)
