cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
# cmake_policy(SET CMP0022 NEW) cmake_policy(SET CMP0023 NEW)

# Use compiler ID "AppleClang" instead of "Clang" for XCode. Not setting this
# sometimes makes XCode C compiler gets detected as "Clang", even when the C++
# one is detected as "AppleClang".
cmake_policy(SET CMP0010 NEW)
cmake_policy(SET CMP0025 NEW)
cmake_policy(SET CMP0126 OLD)

# Enables CMake to set LTO on compilers other than Intel.
cmake_policy(SET CMP0069 NEW)
# Enable the policy for CMake subprojects. protobuf currently causes issues
# set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)

# Suppress warning flags in default MSVC configuration.  It's not mandatory that
# we do this (and we don't if cmake is old), but it's nice when it's possible,
# and it's possible on our Windows configs.
cmake_policy(SET CMP0092 NEW)
# Don't remove the FindCUDA module
cmake_policy(SET CMP0146 OLD)

function(print_all_variables)
    message(STATUS "--- CMake Variables ---")
    get_cmake_property(_variableNames VARIABLES)
    list(SORT _variableNames)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
    message(STATUS "-----------------------")
    message(STATUS "")
endfunction()

message(STATUS "Before project command")
print_all_variables()

# Prohibit in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source build are not supported")
endif()

# ---[ Project and semantic versioning.
project(Torch CXX C)

message(STATUS "After project command:")
print_all_variables()

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  set(LINUX TRUE)
else()
  set(LINUX FALSE)
endif()

set(CMAKE_INSTALL_MESSAGE NEVER)

# check and set CMAKE_CXX_STANDARD
string(FIND "${CMAKE_CXX_FLAGS}" "-std=c++" env_cxx_standard)
if(env_cxx_standard GREATER -1)
  message(
    WARNING
      "C++ standard version definition detected in environment variable."
      "PyTorch requires -std=c++17. Please remove -std=c++ settings in your environment."
  )
endif()
set(CMAKE_CXX_STANDARD
    17
    CACHE STRING
          "The C++ standard whose features are requested to build this target.")
set(CMAKE_C_STANDARD
    11
    CACHE STRING
          "The C standard whose features are requested to build this target.")

# ---[ Utils
include(cmake/public/utils.cmake)

# --- [ Check that minimal gcc version is 9.3+
if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.3)
  message(
    FATAL_ERROR
      "GCC-9.3 or newer is required to compile PyTorch, but found ${CMAKE_CXX_COMPILER_VERSION}"
  )
endif()

# This define is needed to preserve behavior given anticpated changes to
# cccl/thrust
# https://nvidia.github.io/cccl/libcudacxx/standard_api/numerics_library/complex.html
string(APPEND CMAKE_CUDA_FLAGS
       " -DLIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS")

if(LINUX)
  set(CXX_STANDARD_REQUIRED ON)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_LINK_WHAT_YOU_USE TRUE)

# One variable that determines whether the current cmake process is being run
# with the main Caffe2 library. This is useful for building modules - if modules
# are built with the main Caffe2 library then one does not need to do find
# caffe2 in the cmake script. One can usually guard it in some way like if(NOT
# CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO) find_package(Caffe2 REQUIRED) endif()
set(CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO ON)

# Googletest's cmake files are going to set it on once they are processed. Let's
# set it at the very beginning so that the entire build is deterministic.
set(THREADS_PREFER_PTHREAD_FLAG ON)

if(NOT DEFINED BLAS_SET_BY_USER)
  if(DEFINED BLAS)
    set(BLAS_SET_BY_USER TRUE)
  else()
    message(STATUS "Not forcing any particular BLAS to be found")
    set(BLAS_SET_BY_USER FALSE)
  endif()
  set(BLAS_SET_BY_USER
      ${BLAS_SET_BY_USER}
      CACHE STRING
            "Marks whether BLAS was manually set by user or auto-detected")
endif()

set(CPU_INTEL ON)
set(CPU_AARCH64 OFF)
set(CPU_POWER OFF)

# ---[ Options. Note to developers: if you add an option below, make sure you
# also add it to cmake/Summary.cmake so that the summary prints out the option
# values.
include(CMakeDependentOption)
option(ATEN_NO_TEST "Do not build ATen test binaries" OFF)
option(BUILD_BINARY "Build C++ binaries" OFF)
option(BUILD_CUSTOM_PROTOBUF
       "Build and use Caffe2's own protobuf under third_party" ON)
option(BUILD_PYTHON "Build Python binaries" ON)
option(BUILD_LITE_INTERPRETER "Master flag to build Lite Interpreter" OFF)
option(BUILD_SHARED_LIBS "Build libcaffe2.so" ON)
cmake_dependent_option(
  CAFFE2_LINK_LOCAL_PROTOBUF "If set, build protobuf inside libcaffe2.so." ON
  "BUILD_SHARED_LIBS AND BUILD_CUSTOM_PROTOBUF" OFF)
cmake_dependent_option(
  CAFFE2_USE_MSVC_STATIC_RUNTIME "Using MSVC static runtime libraries" ON
  "NOT BUILD_SHARED_LIBS" OFF)
option(BUILD_TEST "Build C++ test binaries (need gtest and gbenchmark)" OFF)
option(BUILD_AOT_INDUCTOR_TEST "Build C++ test binaries for aot-inductor" OFF)
option(BUILD_STATIC_RUNTIME_BENCHMARK
       "Build C++ binaries for static runtime benchmarks (need gbenchmark)" OFF)
option(
  BUILD_MOBILE_BENCHMARK
  "Build C++ test binaries for mobile (ARM) targets(need gtest and gbenchmark)"
  OFF)
option(
  BUILD_MOBILE_TEST
  "Build C++ test binaries for mobile (ARM) targets(need gtest and gbenchmark)"
  OFF)
option(BUILD_JNI "Build JNI bindings" OFF)
option(BUILD_MOBILE_AUTOGRAD
       "Build autograd function in mobile build (in development)" OFF)
cmake_dependent_option(INSTALL_TEST "Install test binaries if BUILD_TEST is on"
                       ON "BUILD_TEST" OFF)
option(USE_CPP_CODE_COVERAGE "Compile C/C++ with code coverage flags" OFF)
option(USE_COLORIZE_OUTPUT "Colorize output during compilation" ON)
option(USE_ASAN "Use Address+Undefined Sanitizers" OFF)
option(USE_LSAN "Use Leak Sanitizer" OFF)
option(USE_TSAN "Use Thread Sanitizer" OFF)
option(USE_CUDA "Use CUDA" ON)
option(USE_XPU "Use XPU" ON)
cmake_dependent_option(
  BUILD_LAZY_CUDA_LINALG "Build cuda linalg ops as separate library" ON
  "USE_CUDA AND LINUX AND BUILD_PYTHON" OFF)
cmake_dependent_option(USE_ROCM "Use ROCm" ON "LINUX OR WIN32" OFF)
cmake_dependent_option(USE_ROCM_CK_GEMM "Use ROCm Composable Kernel for GEMMs" ON "USE_ROCM;NOT WIN32" OFF)
option(USE_ROCM_CK_SDPA "Use ROCm Composable Kernel for SDPA" OFF)
option(CAFFE2_STATIC_LINK_CUDA "Statically link CUDA libraries" OFF)
cmake_dependent_option(USE_CUDNN "Use cuDNN" ON "USE_CUDA" OFF)
cmake_dependent_option(USE_STATIC_CUDNN "Use cuDNN static libraries" OFF
                       "USE_CUDNN" OFF)
cmake_dependent_option(USE_CUSPARSELT "Use cuSPARSELt" ON "USE_CUDA" OFF)
cmake_dependent_option(USE_CUDSS "Use cuDSS" ON "USE_CUDA" OFF)
# USE_ROCM is guarded against in Dependencies.cmake because USE_ROCM is not properly defined here
cmake_dependent_option(USE_CUFILE "Use cuFile" ON "USE_CUDA AND NOT WIN32" OFF)
option(USE_FBGEMM "Use FBGEMM (quantized 8-bit server operators)" ON)
option(USE_KINETO "Use Kineto profiling library" ON)
option(USE_CUPTI_SO "Use CUPTI as a shared library" ON)
option(USE_GFLAGS "Use GFLAGS" OFF)
option(USE_GLOG "Use GLOG" OFF)
option(USE_LITE_PROTO "Use lite protobuf instead of full." OFF)
option(USE_MAGMA "Use MAGMA" ON)
option(USE_PYTORCH_METAL "Use Metal for PyTorch iOS build" OFF)
option(USE_PYTORCH_METAL_EXPORT "Export Metal models on MacOSX desktop" OFF)
option(USE_NATIVE_ARCH "Use -march=native" OFF)
cmake_dependent_option(USE_MPS "Use MPS for macOS build" ON "MPS_FOUND" OFF)
option(USE_DISTRIBUTED "Enable default distributed backends" ON)
cmake_dependent_option(USE_NCCL "Use NCCL" ON
                       "USE_DISTRIBUTED;USE_CUDA OR USE_ROCM;UNIX;NOT APPLE" OFF)
cmake_dependent_option(USE_XCCL "Use XCCL" ON
                       "USE_DISTRIBUTED;USE_XPU;UNIX;NOT APPLE" OFF)
cmake_dependent_option(USE_RCCL "Use RCCL" ON USE_NCCL OFF)
cmake_dependent_option(USE_RCCL "Use RCCL" ON "USE_NCCL;NOT WIN32" OFF)
cmake_dependent_option(USE_STATIC_NCCL "Use static NCCL" OFF "USE_NCCL" OFF)
cmake_dependent_option(USE_SYSTEM_NCCL "Use system-wide NCCL" OFF "USE_NCCL"
                       OFF)
cmake_dependent_option(USE_NVSHMEM "Use NVSHMEM" ON
                       "USE_DISTRIBUTED;USE_CUDA OR USE_ROCM;UNIX;NOT APPLE" OFF)
option(USE_NNAPI "Use NNAPI" OFF)
option(USE_NNPACK "Use NNPACK" ON)
cmake_dependent_option(USE_NUMA "Use NUMA. Only available on Linux." ON "LINUX"
                       OFF)
cmake_dependent_option(USE_NVRTC "Use NVRTC. Only available if USE_CUDA is on."
                       OFF "USE_CUDA" OFF)
option(USE_NUMPY "Use NumPy" ON)
option(USE_OBSERVERS "Use observers module." OFF)
option(USE_OPENCL "Use OpenCL" OFF)
option(USE_OPENMP "Use OpenMP for parallel code" ON)
option(USE_PRECOMPILED_HEADERS "Use pre-compiled headers to accelerate build."
       OFF)

option(USE_PROF "Use profiling" OFF)
option(USE_PYTORCH_QNNPACK "Use ATen/QNNPACK (quantized 8-bit operators)" ON)
option(USE_SNPE "Use Qualcomm's SNPE library" OFF)
option(USE_EIGEN_SPARSE "Use Eigen Sparse Matrices" OFF)
option(USE_SYSTEM_EIGEN_INSTALL
    "Use system Eigen instead of the one under third_party" OFF)
cmake_dependent_option(
    USE_VALGRIND "Use Valgrind. Only available on Linux." ON
    "LINUX" OFF)

if(NOT DEFINED USE_VULKAN)
  cmake_dependent_option(USE_VULKAN "Use Vulkan GPU backend" ON "ANDROID" OFF)
endif()

option(USE_SOURCE_DEBUG_ON_MOBILE "Enable" ON)
option(USE_LITE_INTERPRETER_PROFILER "Enable" ON)
cmake_dependent_option(
  USE_LITE_AOTI "Include AOTI sources" OFF
  "BUILD_LITE_INTERPRETER" OFF)
option(USE_VULKAN_FP16_INFERENCE "Vulkan - Use fp16 inference" OFF)
option(USE_VULKAN_RELAXED_PRECISION
       "Vulkan - Use relaxed precision math in the kernels (mediump)" OFF)
# option USE_XNNPACK: try to enable xnnpack by default.
option(USE_XNNPACK "Use XNNPACK" ON)
option(USE_ROCM_KERNEL_ASSERT "Use Kernel Assert for ROCm" OFF)
# Ensure that an ITT build is the default for x86 CPUs
cmake_dependent_option(USE_ITT "Use Intel(R) VTune Profiler ITT functionality"
                       ON "CPU_INTEL" OFF)
# Ensure that an MKLDNN build is the default for x86 CPUs but optional for
# AArch64 (dependent on -DUSE_MKLDNN).
cmake_dependent_option(
  USE_MKLDNN "Use MKLDNN. Only available on x86, x86_64, AArch64, and ppc64le."
  "${CPU_INTEL}" "CPU_INTEL OR CPU_AARCH64 OR CPU_POWER" OFF)
cmake_dependent_option(
  USE_MKLDNN_ACL "Use Compute Library for the Arm architecture." OFF
  "USE_MKLDNN AND CPU_AARCH64" OFF)
set(MKLDNN_ENABLE_CONCURRENT_EXEC ${USE_MKLDNN})
cmake_dependent_option(USE_MKLDNN_CBLAS "Use CBLAS in MKLDNN" OFF "USE_MKLDNN"
                       OFF)
option(USE_STATIC_MKL "Prefer to link with MKL statically (Unix only)" OFF)
cmake_dependent_option(
  USE_MPI "Use MPI for Caffe2. Only available if USE_DISTRIBUTED is on." ON
  "USE_DISTRIBUTED" OFF)
cmake_dependent_option(
  USE_UCC "Use UCC. Only available if USE_DISTRIBUTED is on." OFF
  "USE_DISTRIBUTED" OFF)
cmake_dependent_option(USE_SYSTEM_UCC "Use system-wide UCC" OFF "USE_UCC" OFF)
cmake_dependent_option(USE_C10D_UCC "USE C10D UCC" ON "USE_DISTRIBUTED;USE_UCC"
                       OFF)
cmake_dependent_option(
    USE_GLOO "Use Gloo. Only available if USE_DISTRIBUTED is on." ON
    "USE_DISTRIBUTED" OFF)
cmake_dependent_option(
  USE_GLOO_WITH_OPENSSL "Use Gloo with OpenSSL. Only available if USE_GLOO is on." OFF
    "USE_GLOO AND LINUX AND NOT INTERN_BUILD_MOBILE" OFF)
cmake_dependent_option(
  USE_GLOO_IBVERBS "Use Gloo with ibverbs backend. Only available if USE_GLOO is on." OFF
    "USE_GLOO AND LINUX AND NOT INTERN_BUILD_MOBILE" OFF)
cmake_dependent_option(
    USE_C10D_GLOO "USE C10D GLOO" ON "USE_DISTRIBUTED;USE_GLOO" OFF)
cmake_dependent_option(
    USE_C10D_NCCL "USE C10D NCCL" ON "USE_DISTRIBUTED;USE_NCCL" OFF)
cmake_dependent_option(
    USE_C10D_XCCL "USE C10D XCCL" ON "USE_DISTRIBUTED;USE_XCCL" OFF)
cmake_dependent_option(
    USE_C10D_MPI "USE C10D MPI" ON "USE_DISTRIBUTED;USE_MPI" OFF)
cmake_dependent_option(
    USE_TENSORPIPE "Use TensorPipe. Only available if USE_DISTRIBUTED is on." ON
    "USE_DISTRIBUTED AND NOT WIN32" OFF)
option(ONNX_ML "Enable traditional ONNX ML API." ON)
option(HAVE_SOVERSION "Whether to add SOVERSION to the shared objects" OFF)
option(BUILD_LIBTORCH_CPU_WITH_DEBUG
       "Enable RelWithDebInfo for libtorch_cpu target only" OFF)
cmake_dependent_option(
  USE_CCACHE "Attempt using CCache to wrap the compilation" ON "UNIX" OFF)
option(WERROR "Build with -Werror supported by the compiler" OFF)
option(
  DEBUG_CUDA
  "When compiling DEBUG, also attempt to compile CUDA with debug flags (may cause nvcc to OOM)"
  OFF)
option(USE_COREML_DELEGATE "Use the CoreML backend through delegate APIs" OFF)
option(USE_PER_OPERATOR_HEADERS
       "Whether ATen should generate separate headers for each operator" ON)
cmake_dependent_option(
  BUILD_LAZY_TS_BACKEND
  "Build the lazy Torchscript backend, not compatible with mobile builds" ON
  "NOT INTERN_BUILD_MOBILE" OFF)
cmake_dependent_option(BUILD_FUNCTORCH "Build Functorch" ON "BUILD_PYTHON" OFF)
cmake_dependent_option(BUILD_BUNDLE_PTXAS "Bundle PTX into torch/bin fodler"
                       OFF "USE_CUDA" OFF)
cmake_dependent_option(USE_KLEIDIAI "Use KleidiAI for the ARM CPU & AARCH64 architecture." ON
                        "CPU_AARCH64" OFF)

option(USE_MIMALLOC "Use mimalloc" OFF)
# Enable third party mimalloc library to improve memory allocation performance
# on Windows.
option(USE_MIMALLOC_ON_MKL "Use mimalloc on MKL" OFF)

if(USE_CCACHE)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER
        "${CCACHE_PROGRAM}"
        CACHE STRING "C compiler launcher")
    set(CMAKE_CXX_COMPILER_LAUNCHER
        "${CCACHE_PROGRAM}"
        CACHE STRING "CXX compiler launcher")
    set(CMAKE_CUDA_COMPILER_LAUNCHER
        "${CCACHE_PROGRAM}"
        CACHE STRING "CUDA compiler launcher")
  else()
    message(
      STATUS
        "Could not find ccache. Consider installing ccache to speed up compilation."
    )
  endif()
endif()

if(USE_GLOO_WITH_OPENSSL)
  set(USE_TCP_OPENSSL_LOAD
      ON
      CACHE STRING "")
endif()

# Linux distributions do not want too many embedded sources, in that sense we
# need to be able to build pytorch with an (almost) empty third_party directory.
# USE_SYSTEM_LIBS is a shortcut variable to toggle all the # USE_SYSTEM_*
# variables on. Individual USE_SYSTEM_* variables can be toggled with
# USE_SYSTEM_LIBS being "OFF".
option(USE_SYSTEM_LIBS "Use all available system-provided libraries." OFF)
option(USE_SYSTEM_CPUINFO "Use system-provided cpuinfo." OFF)
option(USE_SYSTEM_SLEEF "Use system-provided sleef." OFF)
option(USE_SYSTEM_GLOO "Use system-provided gloo." OFF)
option(USE_SYSTEM_FP16 "Use system-provided fp16." OFF)
option(USE_SYSTEM_PYBIND11 "Use system-provided PyBind11." OFF)
option(USE_SYSTEM_PTHREADPOOL "Use system-provided pthreadpool." OFF)
option(USE_SYSTEM_PSIMD "Use system-provided psimd." OFF)
option(USE_SYSTEM_FXDIV "Use system-provided fxdiv." OFF)
option(USE_SYSTEM_BENCHMARK "Use system-provided google benchmark." OFF)
option(USE_SYSTEM_ONNX "Use system-provided onnx." OFF)
option(USE_SYSTEM_XNNPACK "Use system-provided xnnpack." OFF)
option(USE_SYSTEM_NVTX "Use system-provided nvtx." OFF)
option(USE_GOLD_LINKER "Use ld.gold to link" OFF)
if(USE_SYSTEM_LIBS)
  set(USE_SYSTEM_CPUINFO ON)
  set(USE_SYSTEM_SLEEF ON)
  set(USE_SYSTEM_GLOO ON)
  set(BUILD_CUSTOM_PROTOBUF OFF)
  set(USE_SYSTEM_EIGEN_INSTALL ON)
  set(USE_SYSTEM_FP16 ON)
  set(USE_SYSTEM_PTHREADPOOL ON)
  set(USE_SYSTEM_PSIMD ON)
  set(USE_SYSTEM_FXDIV ON)
  set(USE_SYSTEM_BENCHMARK ON)
  set(USE_SYSTEM_ONNX ON)
  set(USE_SYSTEM_XNNPACK ON)
  set(USE_SYSTEM_PYBIND11 ON)
  if(USE_NCCL)
    set(USE_SYSTEM_NCCL ON)
  endif()
  set(USE_SYSTEM_NVTX ON)
endif()

# /Z7 override option When generating debug symbols, CMake default to use the
# flag /Zi. However, it is not compatible with sccache. So we rewrite it off.
# But some users don't use sccache; this override is for them.
cmake_dependent_option(
  MSVC_Z7_OVERRIDE
  "Work around sccache bug by replacing /Zi and /ZI with /Z7 when using MSVC (if you are not using sccache, you can turn this OFF)"
  ON
  "MSVC"
  OFF)

if(NOT USE_SYSTEM_ONNX)
  set(ONNX_NAMESPACE
      "onnx_torch"
      CACHE
        STRING
        "A namespace for ONNX; needed to build with other frameworks that share ONNX."
  )
else()
  set(ONNX_NAMESPACE
      "onnx"
      CACHE
        STRING
        "A namespace for ONNX; needed to build with other frameworks that share ONNX."
  )
endif()
set(SELECTED_OP_LIST
    ""
    CACHE
      STRING
      "Path to the yaml file that contains the list of operators to include for custom build. Include all operators by default."
)
option(
  STATIC_DISPATCH_BACKEND
  "Name of the backend for which static dispatch code is generated, e.g.: CPU."
  "")
option(
  USE_LIGHTWEIGHT_DISPATCH
  "Enable codegen unboxing for ATen ops, need to work with static dispatch in order to work properly."
  OFF)
if(USE_LIGHTWEIGHT_DISPATCH AND NOT STATIC_DISPATCH_BACKEND)
  message(
    FATAL_ERROR
      "Need to enable static dispatch after enabling USE_LIGHTWEIGHT_DISPATCH.")
endif()
option(TRACING_BASED
       "Master flag to build Lite Interpreter with tracing build option" OFF)
# This is a fix for a rare build issue on Ubuntu: symbol lookup error:
# miniconda3/envs/pytorch-py3.7/lib/libmkl_intel_lp64.so: undefined symbol:
# mkl_blas_dsyrk
# https://software.intel.com/en-us/articles/symbol-lookup-error-when-linking-intel-mkl-with-gcc-on-ubuntu
if(LINUX)
  set(CMAKE_SHARED_LINKER_FLAGS
      "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-as-needed")

  set(ENV_LDFLAGS "$ENV{LDFLAGS}")
  string(STRIP "${ENV_LDFLAGS}" ENV_LDFLAGS)
  # Do not append linker flags passed via env var if they already there
  if(NOT ${CMAKE_SHARED_LINKER_FLAGS} MATCHES "${ENV_LDFLAGS}")
     set(CMAKE_SHARED_LINKER_FLAGS
         "${CMAKE_SHARED_LINKER_FLAGS} ${ENV_LDFLAGS}")
  endif()
endif()

string(APPEND CMAKE_CUDA_FLAGS " -Xfatbin -compress-all")

if(USE_KLEIDIAI AND CMAKE_C_COMPILER_VERSION)
    if(CMAKE_C_COMPILER_VERSION VERSION_LESS 11)
      set(USE_KLEIDIAI OFF)
      message(WARNING "Disabling KleidiAI: Requires at least GCC 11 or Clang 11")
    endif()
endif()

# INTERN_BUILD_ATEN_OPS is used to control whether to build ATen/TH operators.
set(INTERN_BUILD_ATEN_OPS ON)

if(NOT DEFINED USE_BLAS)
  set(USE_BLAS ON)
endif()

# ---[ Version numbers for generated libraries
file(READ version.txt TORCH_DEFAULT_VERSION)
# Strip trailing newline
string(REGEX REPLACE "\n$" "" TORCH_DEFAULT_VERSION "${TORCH_DEFAULT_VERSION}")
if("${TORCH_DEFAULT_VERSION} " STREQUAL " ")
  message(WARNING "Could not get version from base 'version.txt'")
  # If we can't get the version from the version file we should probably set it
  # to something non-sensical like 0.0.0
  set(TORCH_DEFAULT_VERSION, "0.0.0")
endif()
set(TORCH_BUILD_VERSION
    "${TORCH_DEFAULT_VERSION}"
    CACHE STRING "Torch build version")
if(DEFINED ENV{PYTORCH_BUILD_VERSION})
  set(TORCH_BUILD_VERSION
      "$ENV{PYTORCH_BUILD_VERSION}"
      CACHE STRING "Torch build version" FORCE)
endif()
if(NOT TORCH_BUILD_VERSION)
  # An empty string was specified so force version to the default
  set(TORCH_BUILD_VERSION
      "${TORCH_DEFAULT_VERSION}"
      CACHE STRING "Torch build version" FORCE)
endif()
caffe2_parse_version_str(TORCH ${TORCH_BUILD_VERSION})
set(TORCH_SOVERSION "${TORCH_VERSION_MAJOR}.${TORCH_VERSION_MINOR}")

# ---[ CMake scripts + modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# ---[ CMake build directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

enable_testing()

# ---[ Build variables set within the cmake tree
include(cmake/BuildVariables.cmake)
set(CAFFE2_ALLOWLIST
    ""
    CACHE STRING "A allowlist file of files that one should build.")

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - defaulting to Release")
  set(CMAKE_BUILD_TYPE
      "Release"
      CACHE
        STRING
        "Choose the type of build from: Debug Release RelWithDebInfo MinSizeRel Coverage."
        FORCE)
endif()

# ---[ Misc checks to cope with various compiler modes
include(cmake/MiscCheck.cmake)

# External projects
include(ExternalProject)

set(BUILD_ONEDNN_GRAPH OFF)

message(STATUS "Before include(cmake/Dependencies.cmake) command")
print_all_variables()

include(cmake/Dependencies.cmake)

message(STATUS "After include(cmake/Dependencies.cmake) command:")
print_all_variables()

cmake_dependent_option(
  USE_FLASH_ATTENTION
  "Whether to build the flash_attention kernel for scaled dot product attention.\
  Will be disabled if not supported by the platform"
  ON
  "USE_CUDA OR USE_ROCM"
  OFF)

cmake_dependent_option(
  USE_FBGEMM_GENAI
  "Whether to build FBGEMM GenAI quantized GEMM kernels.\
  Will be disabled if not supported by the platform"
  ON
  "USE_ROCM"
  OFF)

IF(USE_FBGEMM_GENAI AND USE_ROCM AND NOT "gfx942" IN_LIST PYTORCH_ROCM_ARCH)
  message(WARNING "Unsupported ROCM arch for FBGEMM GenAI, will set USE_FBGEMM_GENAI to OFF")
  set(USE_FBGEMM_GENAI off)
endif()

# Set USE_FBGEMM_GENAI to ON for CUDA build on SM100.
if(USE_CUDA AND "$ENV{TORCH_CUDA_ARCH_LIST}" MATCHES "10.0" AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
  message(STATUS "Setting USE_FBGEMM_GENAI to ON, doing CUDA build for SM100a")
  set(USE_FBGEMM_GENAI ON)
endif()

# CAVEAT: Again, Flash Attention2 will error while building for sm52 while Mem
# Eff Attention won't
cmake_dependent_option(
  USE_MEM_EFF_ATTENTION
  "Enable memory-efficient attention for scaled dot product attention.\
  Will be disabled if not supported by the platform" ON
  "USE_CUDA OR USE_ROCM" OFF)

#
# Cannot be put into Dependencies.cmake due circular dependency:
# USE_FLASH_ATTENTION -> USE_ROCM -> Dependencies.cmake -> aotriton.cmake
#
if(USE_ROCM)
  if(USE_FLASH_ATTENTION OR USE_MEM_EFF_ATTENTION)
    include(cmake/External/aotriton.cmake)
  endif()
endif()

if(DEBUG_CUDA)
  string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -lineinfo")
  string(APPEND CMAKE_CUDA_FLAGS_RELWITHDEBINFO " -lineinfo")
  # CUDA-12.1 crashes when trying to compile with --source-in-ptx See
  # https://github.com/pytorch/pytorch/issues/102372#issuecomment-1572526893
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.1)
    string(APPEND CMAKE_CUDA_FLAGS_DEBUG " --source-in-ptx")
    string(APPEND CMAKE_CUDA_FLAGS_RELWITHDEBINFO " --source-in-ptx")
  endif()
endif(DEBUG_CUDA)

if(USE_FBGEMM)
  string(APPEND CMAKE_CXX_FLAGS " -DUSE_FBGEMM")
endif()

if(USE_FBGEMM_GENAI)
  string(APPEND CMAKE_CXX_FLAGS " -DUSE_FBGEMM_GENAI")
endif()

if(USE_PYTORCH_QNNPACK)
  string(APPEND CMAKE_CXX_FLAGS " -DUSE_PYTORCH_QNNPACK")
endif()

# Enable sleef on macOS with Apple silicon by default
if((${CMAKE_SYSTEM_NAME} STREQUAL "Darwin") AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "arm64"))
  message(STATUS "Running on macOS with Apple silicon")
  string(APPEND CMAKE_CXX_FLAGS " -DAT_BUILD_ARM_VEC256_WITH_SLEEF")
  add_definitions(-DAT_BUILD_ARM_VEC256_WITH_SLEEF)
endif()

# Enable sleef on Arm(R) architecture by default (except Android)
if((NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Android")
  AND("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "aarch64"))
  string(APPEND CMAKE_CXX_FLAGS " -DAT_BUILD_ARM_VEC256_WITH_SLEEF")
  add_definitions(-DAT_BUILD_ARM_VEC256_WITH_SLEEF)
endif()


if(USE_XNNPACK)
  string(APPEND CMAKE_CXX_FLAGS " -DUSE_XNNPACK")
endif()

if(USE_VULKAN)
  string(APPEND CMAKE_CXX_FLAGS " -DUSE_VULKAN")
  string(APPEND CMAKE_CXX_FLAGS " -DUSE_VULKAN_API")

  if(USE_VULKAN_FP16_INFERENCE)
    string(APPEND CMAKE_CXX_FLAGS " -DUSE_VULKAN_FP16_INFERENCE")
  endif()

  if(USE_VULKAN_RELAXED_PRECISION)
    string(APPEND CMAKE_CXX_FLAGS " -DUSE_VULKAN_RELAXED_PRECISION")
  endif()

endif()

if(BUILD_LITE_INTERPRETER)
  string(APPEND CMAKE_CXX_FLAGS " -DBUILD_LITE_INTERPRETER")
endif()

if(TRACING_BASED)
  string(APPEND CMAKE_CXX_FLAGS " -DTRACING_BASED")
endif()

if(USE_PYTORCH_METAL)
  string(APPEND CMAKE_CXX_FLAGS " -DUSE_PYTORCH_METAL")
endif()

if(USE_PYTORCH_METAL_EXPORT)
  string(APPEND CMAKE_CXX_FLAGS " -DUSE_PYTORCH_METAL_EXPORT")
endif()

if(USE_SOURCE_DEBUG_ON_MOBILE)
  string(APPEND CMAKE_CXX_FLAGS " -DSYMBOLICATE_MOBILE_DEBUG_HANDLE")
endif()

if(BUILD_LITE_INTERPRETER AND USE_LITE_INTERPRETER_PROFILER)
  string(APPEND CMAKE_CXX_FLAGS " -DEDGE_PROFILER_USE_KINETO")
endif()

if(USE_COREML_DELEGATE)
  string(APPEND CMAKE_CXX_FLAGS " -DUSE_COREML_DELEGATE")
endif()

# ---[ Allowlist file if allowlist is specified
include(cmake/Allowlist.cmake)

# ---[ Set link flag, handle additional deps for gcc 4.8 and above
if(CMAKE_COMPILER_IS_GNUCXX AND NOT ANDROID)
  message(
    STATUS
      "GCC ${CMAKE_CXX_COMPILER_VERSION}: Adding gcc and gcc_s libs to link line"
  )
  list(APPEND Caffe2_DEPENDENCY_LIBS gcc_s gcc)
endif()

# ---[ Build flags Re-include to override append_cxx_flag_if_supported from
# third_party/FBGEMM
include(cmake/public/utils.cmake)
if(USE_COLORIZE_OUTPUT)
  set(CMAKE_COLOR_DIAGNOSTICS ON)
endif()
if(NOT MSVC)
  string(APPEND CMAKE_CXX_FLAGS " -O2 -fPIC")

  # This prevents use of `c10::optional`, `c10::nullopt` etc within the codebase
  string(APPEND CMAKE_CXX_FLAGS " -DC10_NODEPRECATED")
  string(APPEND CMAKE_CUDA_FLAGS " -DC10_NODEPRECATED")
  string(APPEND CMAKE_OBJCXX_FLAGS " -DC10_NODEPRECATED")

  # Eigen fails to build with some versions, so convert this to a warning
  # Details at http://eigen.tuxfamily.org/bz/show_bug.cgi?id=1459
  string(APPEND CMAKE_CXX_FLAGS " -Wall")
  string(APPEND CMAKE_CXX_FLAGS " -Wextra")
  append_cxx_flag_if_supported("-Werror=return-type" CMAKE_CXX_FLAGS)
  append_cxx_flag_if_supported("-Werror=non-virtual-dtor" CMAKE_CXX_FLAGS)
  append_cxx_flag_if_supported("-Werror=braced-scalar-init" CMAKE_CXX_FLAGS)
  append_cxx_flag_if_supported("-Werror=range-loop-construct" CMAKE_CXX_FLAGS)
  append_cxx_flag_if_supported("-Werror=bool-operation" CMAKE_CXX_FLAGS)
  append_cxx_flag_if_supported("-Wnarrowing" CMAKE_CXX_FLAGS)
  append_cxx_flag_if_supported("-Wno-missing-field-initializers"
                               CMAKE_CXX_FLAGS)
  append_cxx_flag_if_supported("-Wno-unknown-pragmas" CMAKE_CXX_FLAGS)
  append_cxx_flag_if_supported("-Wno-unused-parameter" CMAKE_CXX_FLAGS)
  append_cxx_flag_if_supported("-Wno-strict-overflow" CMAKE_CXX_FLAGS)
  append_cxx_flag_if_supported("-Wno-strict-aliasing" CMAKE_CXX_FLAGS)
  append_cxx_flag_if_supported("-Wno-stringop-overflow" CMAKE_CXX_FLAGS)
  append_cxx_flag_if_supported("-Wvla-extension" CMAKE_CXX_FLAGS)
  append_cxx_flag_if_supported("-Wsuggest-override" CMAKE_CXX_FLAGS)
  append_cxx_flag_if_supported("-Wnewline-eof" CMAKE_CXX_FLAGS)
  append_cxx_flag_if_supported("-Winconsistent-missing-override"
                               CMAKE_CXX_FLAGS)
  append_cxx_flag_if_supported("-Winconsistent-missing-destructor-override"
                               CMAKE_CXX_FLAGS)
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-pass-failed")
  endif()
  if(CMAKE_COMPILER_IS_GNUCXX)
    # Suppress "The ABI for passing parameters with 64-byte alignment has
    # changed in GCC 4.6"
    string(APPEND CMAKE_CXX_FLAGS " -Wno-psabi")
  endif()

  # Use ld.gold if available, fall back to ld.bfd (the default ld) if not
  if(USE_GOLD_LINKER)
    if(USE_DISTRIBUTED AND USE_MPI)
      # Same issue as here with default MPI on Ubuntu
      # https://bugs.launchpad.net/ubuntu/+source/deal.ii/+bug/1841577
      message(WARNING "Refusing to use gold when USE_MPI=1")
    else()
      execute_process(
        COMMAND "${CMAKE_C_COMPILER}" -fuse-ld=gold -Wl,--version
        ERROR_QUIET
        OUTPUT_VARIABLE LD_VERSION)
      if(NOT "${LD_VERSION}" MATCHES "GNU gold")
        message(
          WARNING
            "USE_GOLD_LINKER was set but ld.gold isn't available, turning it off"
        )
        set(USE_GOLD_LINKER OFF)
      else()
        message(STATUS "ld.gold is available, using it to link")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold")
        set(CMAKE_SHARED_LINKER_FLAGS
            "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=gold")
        set(CMAKE_MODULE_LINKER_FLAGS
            "${CMAKE_MODULE_LINKER_FLAGS} -fuse-ld=gold")
      endif()
    endif()
  endif()

  append_cxx_flag_if_supported("-Wno-error=old-style-cast" CMAKE_CXX_FLAGS)
  append_cxx_flag_if_supported("-Wconstant-conversion" CMAKE_CXX_FLAGS)
  append_cxx_flag_if_supported("-Wno-aligned-allocation-unavailable"
                               CMAKE_CXX_FLAGS)
  append_cxx_flag_if_supported("-Qunused-arguments" CMAKE_CXX_FLAGS)

  append_cxx_flag_if_supported("-faligned-new" CMAKE_CXX_FLAGS)

  if(WERROR)
    append_cxx_flag_if_supported("-Werror" CMAKE_CXX_FLAGS)
    if(NOT COMPILER_SUPPORT_WERROR)
      set(WERROR FALSE)
    endif()
  endif()
  append_cxx_flag_if_supported("-Wno-maybe-uninitialized" CMAKE_CXX_FLAGS)
  append_cxx_flag_if_supported("-fstandalone-debug" CMAKE_CXX_FLAGS_DEBUG)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    if(CMAKE_BUILD_TYPE MATCHES Debug)
      message(Warning "Applying -Og optimization for aarch64 GCC debug build to workaround ICE")
    endif()
    string(APPEND CMAKE_CXX_FLAGS_DEBUG " -fno-omit-frame-pointer -Og")
    string(APPEND CMAKE_LINKER_FLAGS_DEBUG " -fno-omit-frame-pointer -Og")
  else()
    string(APPEND CMAKE_CXX_FLAGS_DEBUG " -fno-omit-frame-pointer -O0")
    string(APPEND CMAKE_LINKER_FLAGS_DEBUG " -fno-omit-frame-pointer -O0")
  endif()
  append_cxx_flag_if_supported("-fno-math-errno" CMAKE_CXX_FLAGS)
  append_cxx_flag_if_supported("-fno-trapping-math" CMAKE_CXX_FLAGS)
  append_cxx_flag_if_supported("-Werror=format" CMAKE_CXX_FLAGS)
  if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
    append_cxx_flag_if_supported("-Wno-dangling-reference" CMAKE_CXX_FLAGS)
    append_cxx_flag_if_supported("-Wno-error=dangling-reference" CMAKE_CXX_FLAGS)
  endif()
else()
  # Define export functions for AOTI.
  add_compile_definitions(EXPORT_AOTI_FUNCTIONS)

  # skip unwanted includes from windows.h
  add_compile_definitions(WIN32_LEAN_AND_MEAN)
  # Windows SDK broke compatibility since version 25131, but introduced this
  # define for backward compatibility.
  add_compile_definitions(_UCRT_LEGACY_INFINITY)
  # disable min/max macros
  add_compile_definitions(NOMINMAX)
  # Turn off these warnings on Windows. destructor was implicitly defined as
  # delete
  append_cxx_flag_if_supported("/wd4624" CMAKE_CXX_FLAGS)
  # unknown pragma
  append_cxx_flag_if_supported("/wd4068" CMAKE_CXX_FLAGS)
  # unexpected tokens following preprocessor directive - expected a newline
  append_cxx_flag_if_supported("/wd4067" CMAKE_CXX_FLAGS)
  # conversion from 'size_t' to 'unsigned int', possible loss of data
  append_cxx_flag_if_supported("/wd4267" CMAKE_CXX_FLAGS)
  # no suitable definition provided for explicit template instantiation request
  append_cxx_flag_if_supported("/wd4661" CMAKE_CXX_FLAGS)
  # recursive on all control paths, function will cause runtime stack overflow
  append_cxx_flag_if_supported("/wd4717" CMAKE_CXX_FLAGS)
  # conversion from '_Ty' to '_Ty', possible loss of data
  append_cxx_flag_if_supported("/wd4244" CMAKE_CXX_FLAGS)
  # unsafe use of type 'bool' in operation
  append_cxx_flag_if_supported("/wd4804" CMAKE_CXX_FLAGS)
  # inconsistent dll linkage
  append_cxx_flag_if_supported("/wd4273" CMAKE_CXX_FLAGS)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  include(CheckCSourceCompiles)
  check_c_source_compiles(
    "#include <arm_neon.h>
int main() {
  float a[] = {1.0, 1.0};
  float32x4x2_t v;
  v.val[0] = vcombine_f32 (vcreate_f32 (0UL), vcreate_f32 (0UL));
  v.val[1] = vcombine_f32 (vcreate_f32 (0UL), vcreate_f32 (0UL));
  vst1q_f32_x2(a, v);
  return 0;
}"
    HAS_VST1)

  if(NOT HAS_VST1)
    string(APPEND CMAKE_CXX_FLAGS " -DMISSING_ARM_VST1")
  endif()
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  include(CheckCSourceCompiles)
  check_c_source_compiles(
    "#include <arm_neon.h>
int main() {
  float a[] = {1.0, 1.0};
  vld1q_f32_x2(a);
  return 0;
}"
    HAS_VLD1)

  if(NOT HAS_VLD1)
    string(APPEND CMAKE_CXX_FLAGS " -DMISSING_ARM_VLD1")
  endif()
endif()

# Add code coverage flags to supported compilers
if(USE_CPP_CODE_COVERAGE)
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    string(APPEND CMAKE_C_FLAGS " --coverage -fprofile-abs-path")
    string(APPEND CMAKE_CXX_FLAGS " --coverage -fprofile-abs-path")
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    string(APPEND CMAKE_C_FLAGS " -fprofile-instr-generate -fcoverage-mapping")
    string(APPEND CMAKE_CXX_FLAGS
           " -fprofile-instr-generate -fcoverage-mapping")
  else()
    message(
      ERROR
      "Code coverage for compiler ${CMAKE_CXX_COMPILER_ID} is unsupported")
  endif()

endif()

if(APPLE)
  if(USE_MPS)
    string(APPEND CMAKE_OBJCXX_FLAGS " -DUSE_MPS -fno-objc-arc")
    string(APPEND CMAKE_CXX_FLAGS " -DUSE_MPS")
    string(
      APPEND
      CMAKE_SHARED_LINKER_FLAGS
      " -weak_framework Foundation -weak_framework MetalPerformanceShaders -weak_framework MetalPerformanceShadersGraph -weak_framework Metal -weak_framework IOKit"
    )
    # To suppress MPSGraph availability warnings
    append_cxx_flag_if_supported("-Wno-unguarded-availability-new"
                                 CMAKE_OBJCXX_FLAGS)
  endif()
  append_cxx_flag_if_supported("-Wno-missing-braces" CMAKE_CXX_FLAGS)
endif()

if(EMSCRIPTEN)
  string(
    APPEND
    CMAKE_CXX_FLAGS
    " -Wno-implicit-function-declaration -DEMSCRIPTEN -s DISABLE_EXCEPTION_CATCHING=0"
  )
endif()

append_cxx_flag_if_supported("-Wno-stringop-overflow" CMAKE_CXX_FLAGS)

if(ANDROID AND (NOT ANDROID_DEBUG_SYMBOLS))
  if(CMAKE_COMPILER_IS_GNUCXX)
    string(APPEND CMAKE_CXX_FLAGS " -s")
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    string(APPEND CMAKE_CXX_FLAGS " -g0")
  else()
    string(APPEND CMAKE_EXE_LINKER_FLAGS " -s")
  endif()
endif()

if(NOT APPLE AND UNIX)
  list(APPEND Caffe2_DEPENDENCY_LIBS dl)
endif()

# Prefix path to Caffe2 headers. If a directory containing installed Caffe2
# headers was inadvertently added to the list of include directories, prefixing
# PROJECT_SOURCE_DIR means this source tree always takes precedence.
include_directories(BEFORE ${PROJECT_SOURCE_DIR})

# Prefix path to generated Caffe2 headers. These need to take precedence over
# their empty counterparts located in PROJECT_SOURCE_DIR.
include_directories(BEFORE ${PROJECT_BINARY_DIR})

include_directories(BEFORE ${PROJECT_SOURCE_DIR}/aten/src/)
include_directories(BEFORE ${CMAKE_BINARY_DIR}/aten/src/)

if(USE_MIMALLOC)
  set(MI_OVERRIDE OFF)
  set(MI_BUILD_SHARED OFF)
  set(MI_BUILD_OBJECT OFF)
  set(MI_BUILD_TESTS OFF)
  add_definitions(-DUSE_MIMALLOC)
  add_subdirectory(third_party/mimalloc)
  include_directories(third_party/mimalloc/include)
endif()

if(USE_MIMALLOC AND USE_MIMALLOC_ON_MKL)
  add_definitions(-DUSE_MIMALLOC_ON_MKL)
endif()

# ---[ Main build
add_subdirectory(torch/headeronly)  # headeronly headers
add_subdirectory(c10)
add_subdirectory(caffe2)

# ---[ CMake related files Uninstall option.
if(NOT TARGET caffe2_uninstall)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake IMMEDIATE @ONLY)

  add_custom_target(
    caffe2_uninstall COMMAND ${CMAKE_COMMAND} -P
                             ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()

# ---[ Make configuration files for cmake to allow dependent libraries easier
# access to Caffe2.

if((NOT USE_GLOG)
   OR(NOT USE_GFLAGS)
   OR BUILD_CUSTOM_PROTOBUF)
  message(WARNING "Generated cmake files are only fully tested if one builds "
                  "with system glog, gflags, and protobuf. Other settings may "
                  "generate files that are not well tested.")
endif()

if(USE_CUDA OR USE_ROCM)
  # TODO: check if we should include other cuda dependency libraries to the
  # interface as well.

endif()

# Note(jiayq): when building static libraries, all PRIVATE dependencies will
# also become interface libraries, and as a result if there are any dependency
# libraries that are not exported, the following install export script will
# fail. As a result, we will only provide the targets cmake files for shared lib
# installation. For more info, read:
# https://cmake.org/pipermail/cmake/2016-May/063400.html
if(BUILD_SHARED_LIBS)
  configure_file(${PROJECT_SOURCE_DIR}/cmake/Caffe2Config.cmake.in
                 ${PROJECT_BINARY_DIR}/Caffe2Config.cmake @ONLY)
  install(
    FILES ${PROJECT_BINARY_DIR}/Caffe2Config.cmake
    DESTINATION share/cmake/Caffe2
    COMPONENT dev)
  install(
    FILES ${PROJECT_SOURCE_DIR}/cmake/public/cuda.cmake
          ${PROJECT_SOURCE_DIR}/cmake/public/xpu.cmake
          ${PROJECT_SOURCE_DIR}/cmake/public/glog.cmake
          ${PROJECT_SOURCE_DIR}/cmake/public/gflags.cmake
          ${PROJECT_SOURCE_DIR}/cmake/public/mkl.cmake
          ${PROJECT_SOURCE_DIR}/cmake/public/mkldnn.cmake
          ${PROJECT_SOURCE_DIR}/cmake/public/protobuf.cmake
          ${PROJECT_SOURCE_DIR}/cmake/public/utils.cmake
          ${PROJECT_SOURCE_DIR}/cmake/public/LoadHIP.cmake
    DESTINATION share/cmake/Caffe2/public
    COMPONENT dev)
  install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/cmake/Modules_CUDA_fix
    DESTINATION share/cmake/Caffe2/
    COMPONENT dev)
  install(
    FILES ${PROJECT_SOURCE_DIR}/cmake/Modules/FindCUDAToolkit.cmake
    DESTINATION share/cmake/Caffe2/
    COMPONENT dev)
  install(
    FILES ${PROJECT_SOURCE_DIR}/cmake/Modules/FindCUSPARSELT.cmake
    DESTINATION share/cmake/Caffe2/
    COMPONENT dev)
  install(
    FILES ${PROJECT_SOURCE_DIR}/cmake/Modules/FindCUDSS.cmake
    DESTINATION share/cmake/Caffe2/
    COMPONENT dev)
  install(
    FILES ${PROJECT_SOURCE_DIR}/cmake/Modules/FindSYCLToolkit.cmake
    DESTINATION share/cmake/Caffe2/
    COMPONENT dev)
  if(NOT BUILD_LIBTORCHLESS)
    install(
      EXPORT Caffe2Targets
      DESTINATION share/cmake/Caffe2
      FILE Caffe2Targets.cmake
      COMPONENT dev)
  endif()
else()
  message(WARNING "Generated cmake files are only available when building "
                  "shared libs.")
endif()

# ---[ Binaries Binaries will be built after the Caffe2 main libraries and the
# modules are built. For the binaries, they will be linked to the Caffe2 main
# libraries, as well as all the modules that are built with Caffe2 (the ones
# built in the previous Modules section above).
if(BUILD_BINARY)
  add_subdirectory(binaries)
endif()

# ---[ JNI
if(BUILD_JNI)
  if(NOT MSVC)
    string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-variable")
  endif()
  set(BUILD_LIBTORCH_WITH_JNI 1)
  set(FBJNI_SKIP_TESTS 1)
  add_subdirectory(android/pytorch_android)
endif()

include(cmake/Summary.cmake)
caffe2_print_configuration_summary()

if(BUILD_FUNCTORCH)
  add_subdirectory(functorch)
endif()

# Parse custom debug info
if(DEFINED USE_CUSTOM_DEBINFO)
  string(REPLACE ";" " " SOURCE_FILES "${USE_CUSTOM_DEBINFO}")
  message(STATUS "Source files with custom debug infos: ${SOURCE_FILES}")

  string(REGEX REPLACE " +" ";" SOURCE_FILES_LIST "${SOURCE_FILES}")

  # Set the COMPILE_FLAGS property for each source file
  foreach(SOURCE_FILE ${SOURCE_FILES_LIST})
    # We have to specify the scope here. We do this by specifying the targets we
    # care about and caffe2/ for all test targets defined there
    if(BUILD_LIBTORCHLESS)
      caffe2_update_option(USE_CUDA OFF)
      set(ALL_PT_TARGETS "torch_python;${C10_LIB};${TORCH_CPU_LIB};${TORCH_LIB}")
    else()
      # @todo test if we can remove this
      set(ALL_PT_TARGETS "torch_python;c10;torch_cpu;torch")
    endif()
    set_source_files_properties(
      ${SOURCE_FILE} DIRECTORY "caffe2/" TARGET_DIRECTORY ${ALL_PT_TARGETS}
      PROPERTIES COMPILE_FLAGS "-g")
  endforeach()

  # Link everything with debug info when any file is in debug mode
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -g")
endif()

# Bundle PTXAS if needed
if(BUILD_BUNDLE_PTXAS AND USE_CUDA)
  if(NOT EXISTS "${PROJECT_SOURCE_DIR}/build/bin/ptxas")
    message(STATUS "Copying PTXAS into the bin folder")
    file(COPY "${CUDAToolkit_BIN_DIR}/ptxas"
         DESTINATION "${PROJECT_BINARY_DIR}")
  endif()
  install(PROGRAMS "${PROJECT_BINARY_DIR}/ptxas"
          DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()
