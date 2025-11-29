from __future__ import annotations

import os
import sys

import filecmp
import glob
import itertools
import json
import shutil
import subprocess
import sysconfig
from collections import defaultdict
from pathlib import Path
from typing import Any, ClassVar, IO

import setuptools.command.bdist_wheel
import setuptools.command.build_ext
import setuptools.command.sdist
import setuptools.errors
from setuptools import Command, Extension, find_packages, setup
from setuptools.dist import Distribution

# Settings assumed
# BUILD_LIBTORCH_WHL = Flase
# BUILD_PYTHON_ONLY = False
# VERBOSE_SCRIPT = True


CWD = Path(__file__).absolute().parent

sys.path.insert(0, str(CWD))  # this only affects the current process
# Add the current directory to PYTHONPATH so that we can import `tools` in subprocesses
os.environ["PYTHONPATH"] = os.pathsep.join(
    [
        str(CWD),
        os.getenv("PYTHONPATH", ""),
    ]
).rstrip(os.pathsep)

from tools.setup_helpers.cmake import CMake, CMakeValue

BUILD_DIR = "build"
RUN_BUILD_DEPS = True
RERUN_CMAKE = False
CMAKE_ONLY = False
filtered_args = []
for i, arg in enumerate(sys.argv):
    if arg == "install":
        print(
            (
                "WARNING: Redirecting 'python setup.py install' to 'pip install . -v --no-build-isolation',"
                " for more info see https://github.com/pytorch/pytorch/issues/152276"
            ),
            file=sys.stderr,
        )
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", ".", "-v", "--no-build-isolation"],
            env={**os.environ},
        )
        sys.exit(result.returncode)
    if arg == "--":
        filtered_args += sys.argv[i:]
        break
    if arg in ["clean", "dist_info", "egg_info", "sdist"]:
        RUN_BUILD_DEPS = False
    filtered_args.append(arg)
sys.argv = filtered_args

def report(*args: Any, file: IO[str] = sys.stderr, flush: bool = True, **kwargs: Any) -> None:
    print(*args, file=file, flush=flush, **kwargs)

# Constant known variables used throughout this file
TORCH_DIR = CWD / "torch"
TORCH_LIB_DIR = TORCH_DIR / "lib"
THIRD_PARTY_DIR = CWD / "third_party"

CMAKE_PYTHON_LIBRARY = Path(sysconfig.get_config_var("LIBDIR")) / sysconfig.get_config_var("INSTSONAME")

TORCH_PACKAGE_NAME = "torch"
LIBTORCH_PKG_NAME = "torch_no_python"

TORCH_VERSION = open(CWD / "version.txt").read().strip()
report(f"Building wheel {TORCH_PACKAGE_NAME}-{TORCH_VERSION}")

cmake = CMake()

# Windows has very bad support for symbolic links.
# Instead of using symlinks, we're going to copy files over
def mirror_files_into_torchgen() -> None:
    # (new_path, orig_path)
    # Directories are OK and are recursively mirrored.
    paths = [
        (
            CWD / "torchgen/packaged/ATen/native/native_functions.yaml",
            CWD / "aten/src/ATen/native/native_functions.yaml",
        ),
        (
            CWD / "torchgen/packaged/ATen/native/tags.yaml",
            CWD / "aten/src/ATen/native/tags.yaml",
        ),
        (
            CWD / "torchgen/packaged/ATen/templates",
            CWD / "aten/src/ATen/templates",
        ),
        (
            CWD / "torchgen/packaged/autograd",
            CWD / "tools/autograd",
        ),
        (
            CWD / "torchgen/packaged/autograd/templates",
            CWD / "tools/autograd/templates",
        ),
    ]
    for new_path, orig_path in paths:
        # Create the dirs involved in new_path if they don't exist
        if not new_path.exists():
            new_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy the files from the orig location to the new location
        if orig_path.is_file():
            shutil.copyfile(orig_path, new_path)
            continue
        if orig_path.is_dir():
            if new_path.exists():
                # copytree fails if the tree exists already, so remove it.
                shutil.rmtree(new_path)
            shutil.copytree(orig_path, new_path)
            continue
        raise RuntimeError("Check the file paths in `mirror_files_into_torchgen()`")

def checkout_nccl():
    nccl_pin_path = CWD / ".ci" / "docker" / "ci_commit_pins" / "nccl-cu12.txt"
    with nccl_pin_path.open(encoding="utf-8") as f:
        release_tag = f.read().strip()
        print(f"-- Checkout nccl release tag: {release_tag}")
        nccl_basedir = THIRD_PARTY_DIR / "nccl"
        if not nccl_basedir.exists():
            subprocess.check_call(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    release_tag,
                    "https://github.com/NVIDIA/nccl",
                ],
                cwd=THIRD_PARTY_DIR,
            )

# all the work we need to do _before_ setup runs
def build_deps() -> None:
    report(f"-- Building version {TORCH_VERSION}")

    checkout_nccl()
    cmake.generate(TORCH_VERSION, True, True, RERUN_CMAKE)
    if CMAKE_ONLY:
        return
    cmake.build()

    if CMAKE_ONLY:
        report(
            'Finished running cmake. Run "ccmake build" or '
            '"cmake-gui build" to adjust build options and '
            '"python -m pip install --no-build-isolation -v ." to build.'
        )
        sys.exit()

    # Use copies instead of symbolic files.
    # Windows has very poor support for them.
    sym_files = [
        CWD / "tools/shared/_utils_internal.py",
        CWD / "torch/utils/benchmark/utils/valgrind_wrapper/callgrind.h",
        CWD / "torch/utils/benchmark/utils/valgrind_wrapper/valgrind.h",
    ]
    orig_files = [
        CWD / "torch/_utils_internal.py",
        CWD / "third_party/valgrind-headers/callgrind.h",
        CWD / "third_party/valgrind-headers/valgrind.h",
    ]
    for sym_file, orig_file in zip(sym_files, orig_files):
        same = False
        if sym_file.exists():
            if filecmp.cmp(sym_file, orig_file):
                same = True
            else:
                sym_file.unlink()
        if not same:
            shutil.copyfile(orig_file, sym_file)


################################################################################
# Building dependent libraries
################################################################################

class build_ext(setuptools.command.build_ext.build_ext):
    def _embed_libomp(self) -> None:
        # Copy libiomp5.dylib/libomp.dylib inside the wheel package on MacOS
        build_lib = Path(self.build_lib)
        build_torch_lib_dir = build_lib / "torch" / "lib"
        build_torch_include_dir = build_lib / "torch" / "include"
        libtorch_cpu_path = build_torch_lib_dir / "libtorch_cpu.dylib"
        if not libtorch_cpu_path.exists():
            return
        # Parse libtorch_cpu load commands
        otool_cmds = (
            subprocess.check_output(["otool", "-l", str(libtorch_cpu_path)])
            .decode("utf-8")
            .split("\n")
        )
        rpaths: list[str] = []
        libs: list[str] = []
        for idx, line in enumerate(otool_cmds):
            if line.strip() == "cmd LC_LOAD_DYLIB":
                lib_name = otool_cmds[idx + 2].strip()
                assert lib_name.startswith("name ")
                libs.append(lib_name.split(" ", 1)[1].rsplit("(", 1)[0][:-1])

            if line.strip() == "cmd LC_RPATH":
                rpath = otool_cmds[idx + 2].strip()
                assert rpath.startswith("path ")
                rpaths.append(rpath.split(" ", 1)[1].rsplit("(", 1)[0][:-1])

        omplib_path: str = get_cmake_cache_vars()["OpenMP_libomp_LIBRARY"]  # type: ignore[assignment]
        omplib_name: str = get_cmake_cache_vars()["OpenMP_C_LIB_NAMES"]  # type: ignore[assignment]
        omplib_name += ".dylib"
        omplib_rpath_path = os.path.join("@rpath", omplib_name)

        # This logic is fragile and checks only two cases:
        # - libtorch_cpu depends on `@rpath/libomp.dylib`e (happens when built inside miniconda environment)
        # - libtorch_cpu depends on `/abs/path/to/libomp.dylib` (happens when built with libomp from homebrew)
        if not any(c in libs for c in [omplib_path, omplib_rpath_path]):
            return

        # Copy libomp/libiomp5 from rpath locations
        target_lib = build_torch_lib_dir / omplib_name
        libomp_relocated = False
        install_name_tool_args: list[str] = []
        for rpath in rpaths:
            source_lib = os.path.join(rpath, omplib_name)
            if not os.path.exists(source_lib):
                continue
            self.copy_file(source_lib, target_lib)
            # Delete old rpath and add @loader_lib to the rpath
            # This should prevent delocate from attempting to package another instance
            # of OpenMP library in torch wheel as well as loading two libomp.dylib into
            # the address space, as libraries are cached by their unresolved names
            install_name_tool_args = [
                "-rpath",
                rpath,
                "@loader_path",
            ]
            libomp_relocated = True
            break
        if not libomp_relocated and os.path.exists(omplib_path):
            self.copy_file(omplib_path, target_lib)
            install_name_tool_args = [
                "-change",
                omplib_path,
                omplib_rpath_path,
            ]
            if "@loader_path" not in rpaths:
                install_name_tool_args += [
                    "-add_rpath",
                    "@loader_path",
                ]
            libomp_relocated = True
        if libomp_relocated:
            install_name_tool_args = [
                "install_name_tool",
                *install_name_tool_args,
                str(libtorch_cpu_path),
            ]
            subprocess.check_call(install_name_tool_args)
        # Copy omp.h from OpenMP_C_FLAGS and copy it into include folder
        omp_cflags: str = get_cmake_cache_vars()["OpenMP_C_FLAGS"]  # type: ignore[assignment]
        if not omp_cflags:
            return
        for include_dir in [
            Path(f.removeprefix("-I"))
            for f in omp_cflags.split(" ")
            if f.startswith("-I")
        ]:
            omp_h = include_dir / "omp.h"
            if not omp_h.exists():
                continue
            target_omp_h = build_torch_include_dir / "omp.h"
            self.copy_file(omp_h, target_omp_h)
            break

    def run(self) -> None:
        # Report build options. This is run after the build completes so # `CMakeCache.txt` exists
        # and we can get an accurate report on what is used and what is not.
        cmake_cache_vars = get_cmake_cache_vars()
        if cmake_cache_vars["USE_NUMPY"]:
            report("-- Building with NumPy bindings")
        else:
            report("-- NumPy not found")
        if cmake_cache_vars["USE_CUDNN"]:
            report(
                "-- Detected cuDNN at "
                f"{cmake_cache_vars['CUDNN_LIBRARY']}, "
                f"{cmake_cache_vars['CUDNN_INCLUDE_DIR']}"
            )
        else:
            report("-- Not using cuDNN")
        if cmake_cache_vars["USE_CUDA"]:
            report(f"-- Detected CUDA at {cmake_cache_vars['CUDA_TOOLKIT_ROOT_DIR']}")
        else:
            report("-- Not using CUDA")
        if cmake_cache_vars["USE_XPU"]:
            report(f"-- Detected XPU runtime at {cmake_cache_vars['SYCL_LIBRARY_DIR']}")
        else:
            report("-- Not using XPU")
        if cmake_cache_vars["USE_MKLDNN"]:
            report("-- Using MKLDNN")
            if cmake_cache_vars["USE_MKLDNN_ACL"]:
                report("-- Using Compute Library for the Arm architecture with MKLDNN")
            else:
                report(
                    "-- Not using Compute Library for the Arm architecture with MKLDNN"
                )
            if cmake_cache_vars["USE_MKLDNN_CBLAS"]:
                report("-- Using CBLAS in MKLDNN")
            else:
                report("-- Not using CBLAS in MKLDNN")
        else:
            report("-- Not using MKLDNN")
        if cmake_cache_vars["USE_NCCL"] and cmake_cache_vars["USE_SYSTEM_NCCL"]:
            report(
                "-- Using system provided NCCL library at "
                f"{cmake_cache_vars['NCCL_LIBRARIES']}, "
                f"{cmake_cache_vars['NCCL_INCLUDE_DIRS']}"
            )
        elif cmake_cache_vars["USE_NCCL"]:
            report("-- Building NCCL library")
        else:
            report("-- Not using NCCL")
        if cmake_cache_vars["USE_DISTRIBUTED"]:
            report("-- Building with distributed package: ")
            report(f"  -- USE_TENSORPIPE={cmake_cache_vars['USE_TENSORPIPE']}")
            report(f"  -- USE_GLOO={cmake_cache_vars['USE_GLOO']}")
            report(f"  -- USE_MPI={cmake_cache_vars['USE_OPENMPI']}")
        else:
            report("-- Building without distributed package")
        if cmake_cache_vars["STATIC_DISPATCH_BACKEND"]:
            report(
                "-- Using static dispatch with "
                f"backend {cmake_cache_vars['STATIC_DISPATCH_BACKEND']}"
            )
        if cmake_cache_vars["USE_LIGHTWEIGHT_DISPATCH"]:
            report("-- Using lightweight dispatch")

        if cmake_cache_vars["USE_ITT"]:
            report("-- Using ITT")
        else:
            report("-- Not using ITT")

        super().run()

    def build_extensions(self) -> None:
        self.create_compile_commands()

        build_lib = Path(self.build_lib).resolve()

        # Copy functorch extension
        for ext in self.extensions:
            if ext.name != "functorch._C":
                continue
            fullname = self.get_ext_fullname(ext.name)
            filename = Path(self.get_ext_filename(fullname))
            src = filename.with_stem("functorch")
            dst = build_lib / filename
            if src.exists():
                report(f"Copying {ext.name} from {src} to {dst}")
                dst.parent.mkdir(parents=True, exist_ok=True)
                self.copy_file(src, dst)

        super().build_extensions()

    def get_outputs(self) -> list[str]:
        outputs = super().get_outputs()
        outputs.append(os.path.join(self.build_lib, "caffe2"))
        report(f"setup.py::get_outputs returning {outputs}")
        return outputs

    def create_compile_commands(self) -> None:
        def load(file: Path) -> list[dict[str, Any]]:
            return json.loads(file.read_text(encoding="utf-8"))

        ninja_files = (CWD / BUILD_DIR).glob("*compile_commands.json")
        cmake_files = (CWD / "torch" / "lib" / "build").glob("*/compile_commands.json")
        all_commands = [
            entry
            for f in itertools.chain(ninja_files, cmake_files)
            for entry in load(f)
        ]

        # cquery does not like c++ compiles that start with gcc.
        # It forgets to include the c++ header directories.
        # We can work around this by replacing the gcc calls that python
        # setup.py generates with g++ calls instead
        for command in all_commands:
            if command["command"].startswith("gcc "):
                command["command"] = "g++ " + command["command"][4:]

        new_contents = json.dumps(all_commands, indent=2)
        contents = ""
        compile_commands_json = CWD / "compile_commands.json"
        if compile_commands_json.exists():
            contents = compile_commands_json.read_text(encoding="utf-8")
        if contents != new_contents:
            compile_commands_json.write_text(new_contents, encoding="utf-8")


class concat_license_files:
    """Merge LICENSE and LICENSES_BUNDLED.txt as a context manager

    LICENSE is the main PyTorch license, LICENSES_BUNDLED.txt is auto-generated
    from all the licenses found in ./third_party/. We concatenate them so there
    is a single license file in the sdist and wheels with all of the necessary
    licensing info.
    """

    def __init__(self, include_files: bool = False) -> None:
        self.f1 = CWD / "LICENSE"
        self.f2 = THIRD_PARTY_DIR / "LICENSES_BUNDLED.txt"
        self.include_files = include_files
        self.bsd_text = ""

    def __enter__(self) -> None:
        """Concatenate files"""

        old_path = sys.path
        sys.path.append(str(THIRD_PARTY_DIR))
        try:
            from build_bundled import create_bundled  # type: ignore[import-not-found]
        finally:
            sys.path = old_path

        self.bsd_text = self.f1.read_text(encoding="utf-8")

        with self.f1.open(mode="a", encoding="utf-8") as f1:
            f1.write("\n\n")
            create_bundled(
                str(THIRD_PARTY_DIR.resolve()),
                f1,
                include_files=self.include_files,
            )

    def __exit__(self, *exc_info: object) -> None:
        """Restore content of f1"""
        self.f1.write_text(self.bsd_text, encoding="utf-8")


# Need to create the proper LICENSE.txt for the wheel
class bdist_wheel(setuptools.command.bdist_wheel.bdist_wheel):
    def run(self) -> None:
        with concat_license_files(include_files=True):
            super().run()

    def write_wheelfile(self, *args: Any, **kwargs: Any) -> None:
        super().write_wheelfile(*args, **kwargs)


class clean(Command):
    user_options: ClassVar[list[tuple[str, str | None, str]]] = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        ignores = (CWD / ".gitignore").read_text(encoding="utf-8")
        for wildcard in filter(None, ignores.splitlines()):
            if wildcard.strip().startswith("#"):
                if "BEGIN NOT-CLEAN-FILES" in wildcard:
                    # Marker is found and stop reading .gitignore.
                    break
                # Ignore lines which begin with '#'.
            else:
                # Don't remove absolute paths from the system
                wildcard = wildcard.lstrip("./")
                for filename in glob.iglob(wildcard):
                    try:
                        os.remove(filename)
                    except OSError:
                        shutil.rmtree(filename, ignore_errors=True)


# Need to dump submodule hashes and create the proper LICENSE.txt for the sdist
class sdist(setuptools.command.sdist.sdist):
    def run(self) -> None:
        with concat_license_files():
            super().run()


def get_cmake_cache_vars() -> defaultdict[str, CMakeValue]:
    try:
        return defaultdict(lambda: False, cmake.get_cmake_cache_variables())
    except FileNotFoundError:
        # CMakeCache.txt does not exist.
        # Probably running "python setup.py clean" over a clean directory.
        return defaultdict(lambda: False)


def configure_extension_build() -> tuple[
    list[Extension],  # ext_modules
    dict[str, type[Command]],  # cmdclass
    list[str],  # packages
    dict[str, list[str]],  # entry_points
    list[str],  # extra_install_requires
]:
    r"""Configures extension build options according to system environment and user's choice.

    Returns:
      The input to parameters ext_modules, cmdclass, packages, and entry_points as required in setuptools.setup.
    """

    cmake_cache_vars = get_cmake_cache_vars()

    ################################################################################
    # Configure compile flags
    ################################################################################

    library_dirs: list[str] = [str(TORCH_LIB_DIR)]
    extra_install_requires: list[str] = []

    extra_link_args = []
    extra_compile_args = [
        "-Wall",
        "-Wextra",
        "-Wno-strict-overflow",
        "-Wno-unused-parameter",
        "-Wno-missing-field-initializers",
        "-Wno-unknown-pragmas",
        # Python 2.6 requires -fno-strict-aliasing, see
        # http://legacy.python.org/dev/peps/pep-3123/
        # We also depend on it in our code (even Python 3).
        "-fno-strict-aliasing",
    ]

    main_compile_args: list[str] = []
    main_libraries: list[str] = ["torch_python"]

    main_link_args: list[str] = []
    main_sources: list[str] = ["torch/csrc/stub.c"]

    # pypi cuda package that requires installation of cuda runtime, cudnn and cublas
    # should be included in all wheels uploaded to pypi
    pytorch_extra_install_requires = os.getenv("PYTORCH_EXTRA_INSTALL_REQUIREMENTS")
    if pytorch_extra_install_requires:
        report(f"pytorch_extra_install_requirements: {pytorch_extra_install_requires}")
        extra_install_requires.extend(
            map(str.strip, pytorch_extra_install_requires.split("|"))
        )

    def make_relative_rpath_args(path: str) -> list[str]:
        return ["-Wl,-rpath,$ORIGIN/" + path]

    ################################################################################
    # Declare extensions and package
    ################################################################################

    ext_modules: list[Extension] = []
    # packages that we want to install into site-packages and include them in wheels
    includes = ["torch", "torch.*", "torchgen", "torchgen.*"]
    # exclude folders that they look like Python packages but are not wanted in wheels
    excludes = ["tools", "tools.*", "caffe2", "caffe2.*"]
    if cmake_cache_vars["BUILD_FUNCTORCH"]:
        includes.extend(["functorch", "functorch.*"])
    else:
        excludes.extend(["functorch", "functorch.*"])
    packages = find_packages(include=includes, exclude=excludes)
    C = Extension(
        "torch._C",
        libraries=main_libraries,
        sources=main_sources,
        language="c",
        extra_compile_args=[
            *main_compile_args,
            *extra_compile_args,
        ],
        include_dirs=[],
        library_dirs=library_dirs,
        extra_link_args=[
            *extra_link_args,
            *main_link_args,
            *make_relative_rpath_args("lib"),
        ],
    )
    ext_modules.append(C)

    # These extensions are built by cmake and copied manually in build_extensions()
    # inside the build_ext implementation
    if cmake_cache_vars["BUILD_FUNCTORCH"]:
        ext_modules.append(Extension(name="functorch._C", sources=[]))

    cmdclass = {
        "bdist_wheel": bdist_wheel,
        "build_ext": build_ext,
        "clean": clean,
        "sdist": sdist,
    }

    entry_points = {
        "console_scripts": [
            "torchrun = torch.distributed.run:main",
        ],
        "torchrun.logs_specs": [
            "default = torch.distributed.elastic.multiprocessing:DefaultLogsSpecs",
        ],
    }

    if cmake_cache_vars["USE_DISTRIBUTED"]:
        # Only enable fr_trace command if distributed is enabled
        entry_points["console_scripts"].append(
            "torchfrtrace = tools.flight_recorder.fr_trace:main",
        )
    return ext_modules, cmdclass, packages, entry_points, extra_install_requires

def main() -> None:
    install_requires = [
        "filelock",
        "typing-extensions>=4.10.0",
        'setuptools ; python_version >= "3.12"',
        "sympy>=1.13.3",
        "networkx>=2.5.1",
        "jinja2",
        "fsspec>=0.8.5",
    ]

    # Parse the command line and check the arguments before we proceed with
    # building deps and setup. We need to set values so `--help` works.
    dist = Distribution()
    dist.script_name = os.path.basename(sys.argv[0])
    dist.script_args = sys.argv[1:]
    try:
        dist.parse_command_line()
    except setuptools.errors.BaseError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    mirror_files_into_torchgen()
    if RUN_BUILD_DEPS:
        build_deps()

    (
        ext_modules,
        cmdclass,
        packages,
        entry_points,
        extra_install_requires,
    ) = configure_extension_build()
    install_requires += extra_install_requires

    torch_package_data = [
        "py.typed",
        "bin/*",
        "test/*",
        "*.pyi",
        "**/*.pyi",
        "lib/*.pdb",
        "lib/**/*.pdb",
        "lib/*shm*",
        "lib/torch_shm_manager",
        "lib/*.h",
        "lib/**/*.h",
        "include/*.h",
        "include/**/*.h",
        "include/*.hpp",
        "include/**/*.hpp",
        "include/*.cuh",
        "include/**/*.cuh",
        "csrc/inductor/aoti_runtime/model.h",
        "_inductor/codegen/*.h",
        "_inductor/codegen/aoti_runtime/*.h",
        "_inductor/codegen/aoti_runtime/*.cpp",
        "_inductor/script.ld",
        "_inductor/kernel/flex/templates/*.jinja",
        "_export/serde/*.yaml",
        "_export/serde/*.thrift",
        "share/cmake/ATen/*.cmake",
        "share/cmake/Caffe2/*.cmake",
        "share/cmake/Caffe2/public/*.cmake",
        "share/cmake/Caffe2/Modules_CUDA_fix/*.cmake",
        "share/cmake/Caffe2/Modules_CUDA_fix/upstream/*.cmake",
        "share/cmake/Caffe2/Modules_CUDA_fix/upstream/FindCUDA/*.cmake",
        "share/cmake/Gloo/*.cmake",
        "share/cmake/Tensorpipe/*.cmake",
        "share/cmake/Torch/*.cmake",
        "utils/benchmark/utils/*.cpp",
        "utils/benchmark/utils/valgrind_wrapper/*.cpp",
        "utils/benchmark/utils/valgrind_wrapper/*.h",
        "utils/model_dump/skeleton.html",
        "utils/model_dump/code.js",
        "utils/model_dump/*.mjs",
        "_dynamo/graph_break_registry.json",
        "tools/dynamo/gb_id_mapping.py",
    ]

    torch_package_data += [
        "lib/libtorch_python.so",
        "lib/libtorch_python.dylib",
        "lib/libtorch_python.dll",
    ]
    torch_package_data += [
        "lib/*.so*",
        "lib/*.dylib*",
        "lib/*.dll",
        "lib/*.lib",
    ]
    # XXX: Why not use wildcards ["lib/aotriton.images/*", "lib/aotriton.images/**/*"] here?
    aotriton_image_path = TORCH_DIR / "lib" / "aotriton.images"
    aks2_files = [
        file.relative_to(TORCH_DIR).as_posix()
        for file in aotriton_image_path.rglob("*")
        if file.is_file()
    ]
    torch_package_data += aks2_files
    if get_cmake_cache_vars()["USE_TENSORPIPE"]:
        torch_package_data += [
            "include/tensorpipe/*.h",
            "include/tensorpipe/**/*.h",
        ]
    if get_cmake_cache_vars()["USE_KINETO"]:
        torch_package_data += [
            "include/kineto/*.h",
            "include/kineto/**/*.h",
        ]
    torchgen_package_data = [
        "packaged/*",
        "packaged/**/*",
    ]
    package_data = {
        "torch": torch_package_data,
    }
    exclude_package_data = {}

    package_data["torchgen"] = torchgen_package_data
    exclude_package_data["torchgen"] = ["*.py[co]"]

    setup(
        name=TORCH_PACKAGE_NAME,
        version=TORCH_VERSION,
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        packages=packages,
        entry_points=entry_points,
        install_requires=install_requires,
        package_data=package_data,
        exclude_package_data=exclude_package_data,
        # Disable automatic inclusion of data files because we want to
        # explicitly control with `package_data` above.
        include_package_data=False,
    )

if __name__ == "__main__":
    main()
