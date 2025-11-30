# RPATH stuff
# see https://cmake.org/Wiki/CMake_RPATH_handling
# Use separate rpaths during build and install phases
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
# Don't use the install-rpath during the build phase
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "$ORIGIN")
# Automatically add all linked folders that are NOT in the build directory to
# the rpath (per library?)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

 # UBSAN triggers when compiling protobuf, so we need to disable it.
set(UBSAN_FLAG "-fsanitize=undefined")

macro(disable_ubsan)
  if(CMAKE_C_FLAGS MATCHES ${UBSAN_FLAG} OR CMAKE_CXX_FLAGS MATCHES ${UBSAN_FLAG})
    set(CAFFE2_UBSAN_ENABLED ON)
    string(REPLACE ${UBSAN_FLAG} "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
    string(REPLACE ${UBSAN_FLAG} "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    message(STATUS, "***RYIN*** disable_ubsan")
  endif()
endmacro()

macro(enable_ubsan)
  if(CAFFE2_UBSAN_ENABLED)
    set(CMAKE_C_FLAGS "${UBSAN_FLAG} ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${UBSAN_FLAG} ${CMAKE_CXX_FLAGS}")
    message(STATUS, "***RYIN*** enable_ubsan")
  endif()
endmacro()

caffe2_update_option(USE_XPU OFF)

include(${CMAKE_CURRENT_LIST_DIR}/ProtoBuf.cmake)

# ---[ Threads
find_package(Threads REQUIRED)
list(APPEND Caffe2_DEPENDENCY_LIBS Threads::Threads)

# ---[ BLAS
set(AT_MKLDNN_ACL_ENABLED 0)
set(AT_MKLDNN_ENABLED 0)
set(AT_MKL_ENABLED 0)
set(AT_KLEIDIAI_ENABLED 0)
set(AT_USE_EIGEN_SPARSE 0)
# setting default preferred BLAS options if not already present.
set(BLAS "MKL" CACHE STRING "Selected BLAS library")
set_property(CACHE BLAS PROPERTY STRINGS "ATLAS;BLIS;Eigen;FLAME;Generic;MKL;OpenBLAS;vecLib;APL")
message(STATUS "Trying to find preferred BLAS backend of choice: " ${BLAS})

find_package(MKL QUIET)  ## RY: Assume MKL not found
include(${CMAKE_CURRENT_LIST_DIR}/public/mkl.cmake)
# message(STATUS "MKL libraries: ${MKL_LIBRARIES}")
# message(STATUS "MKL include directory: ${MKL_INCLUDE_DIR}")
# message(STATUS "MKL OpenMP type: ${MKL_OPENMP_TYPE}")
# message(STATUS "MKL OpenMP library: ${MKL_OPENMP_LIBRARY}")
# include_directories(AFTER SYSTEM ${MKL_INCLUDE_DIR})
# list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS caffe2::mkl)
# set(CAFFE2_USE_MKL ON)
# set(BLAS_INFO "mkl")
# set(BLAS_FOUND 1)
# set(BLAS_LIBRARIES ${MKL_LIBRARIES})
message(WARNING "MKL could not be found. Defaulting to Eigen")
set(CAFFE2_USE_EIGEN_FOR_BLAS ON)
set(CAFFE2_USE_MKL OFF)

set(AT_MKL_SEQUENTIAL 0)
message(WARNING "Preferred BLAS (" ${BLAS} ") cannot be found, now searching for a general BLAS library")
find_package(BLAS) # Assume BLAS_FOUND is OFF
set(USE_BLAS 0)

# --- [ PocketFFT
# Assume MKL_FOUND is OFF, leading to AT_MKL_ENABLED=0
set(POCKETFFT_INCLUDE_DIR "${Torch_SOURCE_DIR}/third_party/pocketfft/")
set(AT_POCKETFFT_ENABLED 1)
message(STATUS "Using pocketfft in directory: ${POCKETFFT_INCLUDE_DIR}")

# ---[ Dependencies
# NNPACK and family (QNNPACK, PYTORCH_QNNPACK, and XNNPACK) can download and
# compile their dependencies in isolation as part of their build.  These dependencies
# are then linked statically with PyTorch.  To avoid the possibility of a version
# mismatch between these shared dependencies, explicitly declare our intent to these
# libraries that we are interested in using the exact same source dependencies for all.

set(CAFFE2_THIRD_PARTY_ROOT "${PROJECT_SOURCE_DIR}/third_party")
set(CPUINFO_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/cpuinfo" CACHE STRING "cpuinfo source directory")
set(FP16_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/FP16" CACHE STRING "FP16 source directory")
set(FXDIV_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/FXdiv" CACHE STRING "FXdiv source directory")
set(PSIMD_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/psimd" CACHE STRING "PSimd source directory")
set(PTHREADPOOL_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/pthreadpool" CACHE STRING "pthreadpool source directory")

set(CONFU_DEPENDENCIES_SOURCE_DIR ${PROJECT_BINARY_DIR}/confu-srcs CACHE PATH "Confu-style dependencies source directory")
set(CONFU_DEPENDENCIES_BINARY_DIR ${PROJECT_BINARY_DIR}/confu-deps CACHE PATH "Confu-style dependencies binary directory")

set(USE_PTHREADPOOL ON CACHE BOOL "" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_PTHREADPOOL")

set(PTHREADPOOL_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/pthreadpool" CACHE STRING "pthreadpool source directory")
set(PTHREADPOOL_BUILD_TESTS OFF CACHE BOOL "")
set(PTHREADPOOL_BUILD_BENCHMARKS OFF CACHE BOOL "")
set(PTHREADPOOL_LIBRARY_TYPE "static" CACHE STRING "")
set(PTHREADPOOL_ALLOW_DEPRECATED_API ON CACHE BOOL "")
add_subdirectory("${PTHREADPOOL_SOURCE_DIR}" "${CONFU_DEPENDENCIES_BINARY_DIR}/pthreadpool")
set_property(TARGET pthreadpool PROPERTY POSITION_INDEPENDENT_CODE ON)
list(APPEND Caffe2_DEPENDENCY_LIBS pthreadpool)

# ---[ Caffe2 uses cpuinfo library in the thread pool
# ---[ But it doesn't support s390x/powerpc and thus not used on s390x/powerpc
set(CPUINFO_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../third_party/cpuinfo" CACHE STRING "cpuinfo source directory")
set(CPUINFO_BUILD_TOOLS OFF CACHE BOOL "")
set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "")
set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "")
set(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "")
set(CPUINFO_LIBRARY_TYPE "static" CACHE STRING "")
set(CPUINFO_LOG_LEVEL "error" CACHE STRING "")
add_subdirectory("${CPUINFO_SOURCE_DIR}" "${CONFU_DEPENDENCIES_BINARY_DIR}/cpuinfo")
# We build static version of cpuinfo but link
# them into a shared library for Caffe2, so they need PIC.
set_property(TARGET cpuinfo PROPERTY POSITION_INDEPENDENT_CODE ON)
list(APPEND Caffe2_DEPENDENCY_LIBS cpuinfo)

# ---[ PYTORCH_QNNPACK
if(USE_PYTORCH_QNNPACK)  # option in root CMakeLists.txt, default to ON
  set(PYTORCH_QNNPACK_SOURCE_DIR "${PROJECT_SOURCE_DIR}/aten/src/ATen/native/quantized/cpu/qnnpack" CACHE STRING "QNNPACK source directory")
  set(PYTORCH_QNNPACK_BUILD_TESTS OFF CACHE BOOL "")
  set(PYTORCH_QNNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "")
  set(PYTORCH_QNNPACK_LIBRARY_TYPE "static" CACHE STRING "")
  add_subdirectory("${PYTORCH_QNNPACK_SOURCE_DIR}" "${CONFU_DEPENDENCIES_BINARY_DIR}/pytorch_qnnpack")
  # We build static versions of QNNPACK and pthreadpool but link
  # them into a shared library for Caffe2, so they need PIC.
  set_property(TARGET pytorch_qnnpack PROPERTY POSITION_INDEPENDENT_CODE ON)
  # QNNPACK depends on gemmlowp headers
  target_include_directories(pytorch_qnnpack PRIVATE "${CAFFE2_THIRD_PARTY_ROOT}/gemmlowp")
endif()
list(APPEND Caffe2_DEPENDENCY_LIBS pytorch_qnnpack)

include(${CMAKE_CURRENT_LIST_DIR}/External/nnpack.cmake)  # third_party/NNPACK is built in nnpack.cmake
list(APPEND Caffe2_DEPENDENCY_LIBS nnpack)

# ---[ XNNPACK
if(USE_XNNPACK AND NOT USE_SYSTEM_XNNPACK)  # ON AND NOT OFF => ON
  set(XNNPACK_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/XNNPACK" CACHE STRING "XNNPACK source directory")
  set(XNNPACK_INCLUDE_DIR "${XNNPACK_SOURCE_DIR}/include" CACHE STRING "XNNPACK include directory")
  set(XNNPACK_LIBRARY_TYPE "static" CACHE STRING "")
  set(XNNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "")
  set(XNNPACK_BUILD_TESTS OFF CACHE BOOL "")

  # Disable ARM BF16 and FP16 vector for now; unused and causes build failures because
  # these new ISA features may not be supported on older compilers
  set(XNNPACK_ENABLE_ARM_BF16 OFF CACHE BOOL "")

  # Disable AVXVNNI for now, older clang versions seem not to support it
  # (clang 12 is where avx-vnni support is added)
  set(XNNPACK_ENABLE_AVXVNNI OFF CACHE BOOL "")

  # Disable I8MM For CI since clang 9 does not support neon i8mm.
  set(XNNPACK_ENABLE_ARM_I8MM OFF CACHE BOOL "")

  # Disable avxvnni int8
  set(XNNPACK_ENABLE_AVXVNNIINT8 OFF CACHE BOOL "")

  set(__caffe2_CMAKE_POSITION_INDEPENDENT_CODE_FLAG ${CMAKE_POSITION_INDEPENDENT_CODE})
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  add_subdirectory("${XNNPACK_SOURCE_DIR}" "${CONFU_DEPENDENCIES_BINARY_DIR}/XNNPACK")

  # Revert to whatever it was before
  set(CMAKE_POSITION_INDEPENDENT_CODE ${__caffe2_CMAKE_POSITION_INDEPENDENT_CODE_FLAG})

  include_directories(SYSTEM ${XNNPACK_INCLUDE_DIR})
  list(APPEND Caffe2_DEPENDENCY_LIBS XNNPACK microkernels-prod)
elseif(NOT TARGET XNNPACK AND USE_SYSTEM_XNNPACK)  # else block will not be executed
  add_library(XNNPACK SHARED IMPORTED)
  add_library(microkernels-prod SHARED IMPORTED)
  find_library(XNNPACK_LIBRARY XNNPACK)
  find_library(microkernels-prod_LIBRARY microkernels-prod)
  set_property(TARGET XNNPACK PROPERTY IMPORTED_LOCATION "${XNNPACK_LIBRARY}")
  set_property(TARGET microkernels-prod PROPERTY IMPORTED_LOCATION "${microkernels-prod_LIBRARY}")
  if(NOT XNNPACK_LIBRARY OR NOT microkernels-prod_LIBRARY)
    message(FATAL_ERROR "Cannot find XNNPACK")
  endif()
  message("-- Found XNNPACK: ${XNNPACK_LIBRARY}")
  list(APPEND Caffe2_DEPENDENCY_LIBS XNNPACK microkernels-prod)
endif()

# ---[ Googletest and benchmark
# Preserve build options.
set(TEMP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})

# We will build gtest as static libs and embed it directly into the binary.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs" FORCE)

# For gtest, we will simply embed it into our test binaries, so we won't
# need to install it.
set(INSTALL_GTEST OFF CACHE BOOL "Install gtest." FORCE)
set(BUILD_GMOCK ON CACHE BOOL "Build gmock." FORCE)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/googletest)
include_directories(BEFORE SYSTEM ${CMAKE_CURRENT_LIST_DIR}/../third_party/googletest/googletest/include)
include_directories(BEFORE SYSTEM ${CMAKE_CURRENT_LIST_DIR}/../third_party/googletest/googlemock/include)

# We will not need to test benchmark lib itself.
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing as we don't need it.")
# We will not need to install benchmark since we link it statically.
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable benchmark install to avoid overwriting vendor install.")
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/benchmark)

# Recover build options.
set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS} CACHE BOOL "Build shared libs" FORCE)

set(CAFFE2_THIRD_PARTY_ROOT "${PROJECT_SOURCE_DIR}/third_party")
set(FBGEMM_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/fbgemm" CACHE STRING "FBGEMM source directory")
set(FBGEMM_BUILD_TESTS OFF CACHE BOOL "")
set(FBGEMM_BUILD_BENCHMARKS OFF CACHE BOOL "")
set(FBGEMM_LIBRARY_TYPE "static" CACHE STRING "")
add_subdirectory("${FBGEMM_SOURCE_DIR}")
target_compile_options_if_supported(asmjit -Wno-unused-but-set-variable)
target_compile_options_if_supported(asmjit -Wno-unused-variable)
list(APPEND Caffe2_DEPENDENCY_LIBS fbgemm)
caffe2_update_option(USE_FBGEMM ON)

caffe2_update_option(USE_NUMA OFF)  # find_package(Numa) => NUMA_FOUND is OFF

find_package(ITT) # FOUND
include_directories(SYSTEM ${ITT_INCLUDE_DIR})
list(APPEND Caffe2_DEPENDENCY_LIBS ${ITT_LIBRARIES})
list(APPEND TORCH_PYTHON_LINK_LIBRARIES ${ITT_LIBRARIES})

# ---[ Caffe2 depends on FP16 library for half-precision conversions
if(NOT TARGET fp16 AND NOT USE_SYSTEM_FP16)
  set(PSIMD_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/psimd" CACHE STRING "PSimd source directory")
  set(FP16_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/FP16" CACHE STRING "FP16 source directory")
  set(FP16_BUILD_TESTS OFF CACHE BOOL "")
  set(FP16_BUILD_BENCHMARKS OFF CACHE BOOL "")
  message(WARNING "FP16 is only cmake-2.8 compatible")
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
  add_subdirectory("${FP16_SOURCE_DIR}" "${CONFU_DEPENDENCIES_BINARY_DIR}/FP16")
  unset(CMAKE_POLICY_VERSION_MINIMUM)
endif()
list(APPEND Caffe2_DEPENDENCY_LIBS fp16)

execute_process(COMMAND "which" "python3" RESULT_VARIABLE _exitcode OUTPUT_VARIABLE _py_exe)
if(${_exitcode} EQUAL 0)
  string(STRIP ${_py_exe} Python_EXECUTABLE)
  message(STATUS "Setting Python to ${Python_EXECUTABLE}")
endif()

# ---[ EIGEN
# Due to license considerations, we will only use the MPL2 parts of Eigen.
set(EIGEN_MPL2_ONLY 1)
message(STATUS "Using third party subdirectory Eigen.")
set(EIGEN3_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../third_party/eigen)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

if(BUILD_PYTHON)  # option in root CMakeLists.txt, default to ON
  set(PYTHON_COMPONENTS Development.Module)
  if(USE_NUMPY)
    list(APPEND PYTHON_COMPONENTS NumPy)
  endif()
  find_package(Python COMPONENTS Interpreter OPTIONAL_COMPONENTS ${PYTHON_COMPONENTS})
else()
  find_package(Python COMPONENTS Interpreter)
endif()

if(NOT Python_Interpreter_FOUND)
  message(FATAL_ERROR "Python3 could not be found.")
endif()

if(${Python_VERSION} VERSION_LESS 3.9)
  message(FATAL_ERROR
    "Found Python libraries version ${Python_VERSION}. Python < 3.9 is no longer supported by PyTorch.")
endif()

# ---[ Python + Numpy
if(BUILD_PYTHON)
  if(Python_Development.Module_FOUND)
    if(USE_NUMPY)
      if(NOT Python_NumPy_FOUND)
        message(WARNING "NumPy could not be found. Not building with NumPy. Suppress this warning with -DUSE_NUMPY=OFF")
        caffe2_update_option(USE_NUMPY OFF)
      else()
        caffe2_update_option(USE_NUMPY ON)
      endif()
    endif()
    # Observers are required in the python build
    caffe2_update_option(USE_OBSERVERS ON)
  else()
    message(WARNING "Python dependencies not met. Not compiling with python. Suppress this warning with -DBUILD_PYTHON=OFF")
    caffe2_update_option(BUILD_PYTHON OFF)
  endif()
endif()

# ---[ pybind11
message(STATUS "Using third_party/pybind11.")
set(pybind11_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/../third_party/pybind11/include)
install(DIRECTORY ${pybind11_INCLUDE_DIRS} DESTINATION ${CMAKE_INSTALL_PREFIX} FILES_MATCHING PATTERN "*.h")
message(STATUS "pybind11 include dirs: " "${pybind11_INCLUDE_DIRS}")
add_library(pybind::pybind11 INTERFACE IMPORTED)
target_include_directories(pybind::pybind11 SYSTEM INTERFACE ${pybind11_INCLUDE_DIRS})
target_link_libraries(pybind::pybind11 INTERFACE Python::Module)

# ---[ OpenTelemetry API headers
# Assume find_package(OpenTelemetryApi) NOTFOUND
message(STATUS "Using third_party/opentelemetry-cpp.")
set(OpenTelemetryApi_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/../third_party/opentelemetry-cpp/api/include)
message(STATUS "opentelemetry api include dirs: " "${OpenTelemetryApi_INCLUDE_DIRS}")
add_library(opentelemetry::api INTERFACE IMPORTED)
target_include_directories(opentelemetry::api SYSTEM INTERFACE ${OpenTelemetryApi_INCLUDE_DIRS})

# Assume find_package(MPI) NOTFOUND
message(WARNING "Not compiling with MPI. Suppress this warning with -DUSE_MPI=OFF")
caffe2_update_option(USE_MPI OFF)

# ---[ OpenMP
include(${CMAKE_CURRENT_LIST_DIR}/Modules/FindOpenMP.cmake)
# Assume OPENMP_FOUND is ON
message(STATUS "Adding OpenMP CXX_FLAGS: " ${OpenMP_CXX_FLAGS})
message(STATUS "Will link against OpenMP libraries: ${OpenMP_CXX_LIBRARIES}")
add_library(caffe2::openmp INTERFACE IMPORTED)
target_link_libraries(caffe2::openmp INTERFACE OpenMP::OpenMP_CXX)
list(APPEND Caffe2_DEPENDENCY_LIBS caffe2::openmp)

# ---[ nvtx
find_path(nvtx3_dir NAMES nvtx3 PATHS "${PROJECT_SOURCE_DIR}/third_party/NVTX/c/include" NO_DEFAULT_PATH)
find_package_handle_standard_args(nvtx3 DEFAULT_MSG nvtx3_dir)
# ASSUME nvtx3_FOUND is ON
add_library(torch::nvtx3 INTERFACE IMPORTED)
target_include_directories(torch::nvtx3 INTERFACE "${nvtx3_dir}")
target_compile_definitions(torch::nvtx3 INTERFACE TORCH_CUDA_USE_NVTX3)

# ASSUME public/LoadHIP.cmake returns PYTORCH_FOUND_HIP OFF
caffe2_update_option(USE_ROCM OFF)
caffe2_update_option(USE_NCCL OFF)
caffe2_update_option(USE_XCCL OFF)

if(USE_TENSORPIPE)  # option in root CMakeLists.txt, default to ON
  set(TP_BUILD_LIBUV ON CACHE BOOL "" FORCE)
  add_compile_options(-DTORCH_USE_LIBUV)
  include_directories(BEFORE SYSTEM ${CMAKE_CURRENT_LIST_DIR}/../third_party/tensorpipe/third_party/libuv/include)
  set(TP_STATIC_OR_SHARED STATIC CACHE STRING "" FORCE)

  # Tensorpipe uses cuda_add_library
  torch_update_find_cuda_flags()
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/tensorpipe)
  # Suppress warning to unblock libnop compilation by clang-17
  # See https://github.com/pytorch/pytorch/issues/151316
  target_compile_options_if_supported(tensorpipe -Wno-missing-template-arg-list-after-template-kw)

  list(APPEND Caffe2_DEPENDENCY_LIBS tensorpipe)
  list(APPEND Caffe2_DEPENDENCY_LIBS nlohmann)
  list(APPEND Caffe2_DEPENDENCY_LIBS moodycamel)
endif()

if(USE_GLOO)  # option in root CMakeLists.txt, default to ON
  # Don't install gloo
  set(GLOO_INSTALL OFF CACHE BOOL "" FORCE)
  set(GLOO_STATIC_OR_SHARED STATIC CACHE STRING "" FORCE)

  if(USE_GLOO_IBVERBS)  # OFF
    set(USE_IBVERBS ON)
  endif()

  # Build BFloat16 cuda kernels
  set(GLOO_USE_TORCH_DTYPES 1)
  set(GLOO_TORCH_DIR ${PROJECT_SOURCE_DIR} ${CMAKE_BINARY_DIR})

  # Temporarily override variables to avoid building Gloo tests/benchmarks
  set(__BUILD_TEST ${BUILD_TEST})
  set(__BUILD_BENCHMARK ${BUILD_BENCHMARK})
  set(BUILD_TEST OFF)
  set(BUILD_BENCHMARK OFF)
  if(USE_DISTRIBUED AND USE_TENSORPIPE)
    get_target_property(_include_dirs uv_a INCLUDE_DIRECTORIES)
    set_target_properties(uv_a PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_include_dirs}")
  endif()
  set(GLOO_USE_CUDA_TOOLKIT ON CACHE BOOL "" FORCE)

  # Disable NCCL/RCCL since we don't use Gloo+NCCL, make sure to re-enable it!
  set(USE_NCCL_SAVED ${USE_NCCL})
  set(USE_RCCL_SAVED ${USE_RCCL})
  set(USE_NCCL OFF)
  set(USE_RCCL OFF)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/gloo)
  set(USE_NCCL ${USE_NCCL_SAVED})
  set(USE_RCCL ${USE_RCCL_SAVED})

  # Here is a little bit hacky. We have to put PROJECT_BINARY_DIR in front
  # of PROJECT_SOURCE_DIR with/without conda system. The reason is that
  # gloo generates a new config.h in the binary directory.
  include_directories(BEFORE SYSTEM ${CMAKE_CURRENT_LIST_DIR}/../third_party/gloo)
  include_directories(BEFORE SYSTEM ${PROJECT_BINARY_DIR}/third_party/gloo)
  set(BUILD_TEST ${__BUILD_TEST})
  set(BUILD_BENCHMARK ${__BUILD_BENCHMARK})

  list(APPEND Caffe2_DEPENDENCY_LIBS gloo)
  add_compile_options(-DCAFFE2_USE_GLOO)
endif()

# ---[ Onnx
if(CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO AND NOT INTERN_DISABLE_ONNX)  # ON AND NOT OFF => ON
  set(TEMP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
  set(BUILD_SHARED_LIBS OFF)
  set(ONNX_USE_MSVC_STATIC_RUNTIME ${CAFFE2_USE_MSVC_STATIC_RUNTIME})
  set(ONNX_USE_LITE_PROTO OFF)
  # If linking local protobuf, make sure ONNX has the same protobuf
  # patches as Caffe2 and Caffe proto. This forces some functions to
  # not be inline and instead route back to the statically-linked protobuf.
  if(CAFFE2_LINK_LOCAL_PROTOBUF)
    set(ONNX_PROTO_POST_BUILD_SCRIPT ${PROJECT_SOURCE_DIR}/cmake/ProtoBufPatch.cmake)
  endif()
  if(ONNX_ML)  # ON
    add_definitions(-DONNX_ML=1)
  endif()
  add_definitions(-DONNXIFI_ENABLE_EXT=1)
  set(Python3_EXECUTABLE "${Python_EXECUTABLE}")
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/onnx EXCLUDE_FROM_ALL)
  
  add_definitions(-DONNX_NAMESPACE=${ONNX_NAMESPACE})
  caffe2_interface_library(onnx onnx_library)
  list(APPEND Caffe2_DEPENDENCY_WHOLE_LINK_LIBS onnx_library)

  # Recover the build shared libs option.
  set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS})
endif()

# --[ ATen checks
set(USE_LAPACK 0)

if(NOT INTERN_BUILD_MOBILE)  # NOT OFF => ON
  set(TORCH_CUDA_ARCH_LIST $ENV{TORCH_CUDA_ARCH_LIST})
  string(APPEND CMAKE_CUDA_FLAGS " $ENV{TORCH_NVCC_FLAGS}")
  set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

  # Top-level build config
  if(WERROR)  # OFF
    string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler -Werror -Xcompiler -Wno-error=sign-compare ")
  endif()

  string(APPEND CMAKE_CUDA_FLAGS " -Wno-deprecated-gpu-targets --expt-extended-lambda")

  # use cub in a safe manner, see:
  # https://github.com/pytorch/pytorch/pull/55292
  string(APPEND CMAKE_CUDA_FLAGS " -DCUB_WRAPPED_NAMESPACE=at_cuda_detail")

  message(STATUS "Found CUDA with FP16 support, compiling with torch.cuda.HalfTensor")
  string(APPEND CMAKE_CUDA_FLAGS " -DCUDA_HAS_FP16=1"
                                 " -D__CUDA_NO_HALF_OPERATORS__"
                                 " -D__CUDA_NO_HALF_CONVERSIONS__"
                                 " -D__CUDA_NO_HALF2_OPERATORS__"
                                 " -D__CUDA_NO_BFLOAT16_CONVERSIONS__")

  string(APPEND CMAKE_C_FLAGS_RELEASE " -DNDEBUG")
  string(APPEND CMAKE_CXX_FLAGS_RELEASE " -DNDEBUG")
  if(NOT GENERATOR_IS_MULTI_CONFIG)
    if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
      message(STATUS "Adding -DNDEBUG to compile flags")
      string(APPEND CMAKE_C_FLAGS " -DNDEBUG")
      string(APPEND CMAKE_CXX_FLAGS " -DNDEBUG")
    else()
      message(STATUS "Removing -DNDEBUG from compile flags")
      string(REGEX REPLACE "[-/]DNDEBUG" "" CMAKE_C_FLAGS "" ${CMAKE_C_FLAGS})
      string(REGEX REPLACE "[-/]DNDEBUG" "" CMAKE_CXX_FLAGS "" ${CMAKE_CXX_FLAGS})
    endif()
  endif()
  string(REGEX REPLACE "[-/]DNDEBUG" "" CMAKE_C_FLAGS_DEBUG "" ${CMAKE_C_FLAGS_DEBUG})
  string(REGEX REPLACE "[-/]DNDEBUG" "" CMAKE_CXX_FLAGS_DEBUG "" ${CMAKE_CXX_FLAGS_DEBUG})

  set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)

  caffe2_update_option(USE_MAGMA OFF)

  find_package(LAPACK)
  if(LAPACK_FOUND)  # ON
    set(USE_LAPACK 1)
    list(APPEND Caffe2_PRIVATE_DEPENDENCY_LIBS ${LAPACK_LIBRARIES})
  endif()

  set(AT_CUDA_ENABLED 0)
  set(AT_ROCM_ENABLED 0)

  if(USE_MKLDNN)  # ON
    include(${CMAKE_CURRENT_LIST_DIR}/public/mkldnn.cmake)
    # ASSUME MKLDNN_FOUND is ON
    set(AT_MKLDNN_ENABLED 1)
    include_directories(AFTER SYSTEM ${MKLDNN_INCLUDE_DIR})
  else()
    message("disabling MKLDNN because USE_MKLDNN is not set")
  endif()

  if(UNIX AND NOT APPLE)  # ON AND NOT OFF => ON
     include(CheckLibraryExists)
     # https://github.com/libgit2/libgit2/issues/2128#issuecomment-35649830
     CHECK_LIBRARY_EXISTS(rt clock_gettime "time.h" NEED_LIBRT)
     if(NEED_LIBRT)  # ON
       list(APPEND Caffe2_DEPENDENCY_LIBS rt)
       set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} rt)
     endif(NEED_LIBRT)
  endif(UNIX AND NOT APPLE)

  if(UNIX)  # ON
    set(CMAKE_EXTRA_INCLUDE_FILES "sys/mman.h")
    CHECK_FUNCTION_EXISTS(mmap HAVE_MMAP)
    if(HAVE_MMAP)  # ON
      add_definitions(-DHAVE_MMAP=1)
    endif(HAVE_MMAP)
    # done for lseek: https://www.gnu.org/software/libc/manual/html_node/File-Position-Primitive.html
    add_definitions(-D_FILE_OFFSET_BITS=64)
    CHECK_FUNCTION_EXISTS(shm_open HAVE_SHM_OPEN)
    if(HAVE_SHM_OPEN)
      add_definitions(-DHAVE_SHM_OPEN=1)
    endif(HAVE_SHM_OPEN)
    CHECK_FUNCTION_EXISTS(shm_unlink HAVE_SHM_UNLINK)
    if(HAVE_SHM_UNLINK)
      add_definitions(-DHAVE_SHM_UNLINK=1)
    endif(HAVE_SHM_UNLINK)
    CHECK_FUNCTION_EXISTS(malloc_usable_size HAVE_MALLOC_USABLE_SIZE)
    if(HAVE_MALLOC_USABLE_SIZE)
      add_definitions(-DHAVE_MALLOC_USABLE_SIZE=1)
    endif(HAVE_MALLOC_USABLE_SIZE)
  endif(UNIX)

  add_definitions(-DUSE_EXTERNAL_MZCRC)
  add_definitions(-DMINIZ_DISABLE_ZIP_READER_CRC32_CHECKS)
endif()

#
# End ATen checks
#
set(TEMP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs" FORCE)
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/fmt)

# Disable compiler feature checks for `fmt`.
#
# CMake compiles a little program to check compiler features. Some of our build
# configurations (notably the mobile build analyzer) will populate
# CMAKE_CXX_FLAGS in ways that break feature checks. Since we already know
# `fmt` is compatible with a superset of the compilers that PyTorch is, it
# shouldn't be too bad to just disable the checks.
set_target_properties(fmt-header-only PROPERTIES INTERFACE_COMPILE_FEATURES "")

list(APPEND Caffe2_DEPENDENCY_LIBS fmt::fmt-header-only)
set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS} CACHE BOOL "Build shared libs" FORCE)

if(USE_KINETO)
  set(LIBKINETO_NOCUPTI ON CACHE STRING "" FORCE)
  set(LIBKINETO_NOROCTRACER ON CACHE STRING "" FORCE)
  set(LIBKINETO_NOXPUPTI ON CACHE STRING "" FORCE)

  if(LIBKINETO_NOCUPTI AND LIBKINETO_NOROCTRACER AND LIBKINETO_NOXPUPTI)
    message(STATUS "Using CPU-only version of Kineto")
  endif()

  set(KINETO_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/kineto/libkineto" CACHE STRING "")
  set(KINETO_BUILD_TESTS OFF CACHE BOOL "")
  set(KINETO_LIBRARY_TYPE "static" CACHE STRING "")

  message(STATUS "Configuring Kineto dependency:")
  message(STATUS "  KINETO_SOURCE_DIR = ${KINETO_SOURCE_DIR}")
  message(STATUS "  KINETO_BUILD_TESTS = ${KINETO_BUILD_TESTS}")
  message(STATUS "  KINETO_LIBRARY_TYPE = ${KINETO_LIBRARY_TYPE}")

  add_subdirectory("${KINETO_SOURCE_DIR}")
  set_property(TARGET kineto PROPERTY POSITION_INDEPENDENT_CODE ON)
  list(APPEND Caffe2_DEPENDENCY_LIBS kineto)
  string(APPEND CMAKE_CXX_FLAGS " -DUSE_KINETO")
  string(APPEND CMAKE_CXX_FLAGS " -DLIBKINETO_NOCUPTI")
  string(APPEND CMAKE_CXX_FLAGS " -DLIBKINETO_NOROCTRACER")
  string(APPEND CMAKE_CXX_FLAGS " -DLIBKINETO_NOXPUPTI=ON")
endif()

# Include google/FlatBuffers
include(${CMAKE_CURRENT_LIST_DIR}/FlatBuffers.cmake)

# Include cpp-httplib
add_library(httplib INTERFACE IMPORTED)
target_include_directories(httplib SYSTEM INTERFACE ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)

# Include nlohmann-json
add_library(nlohmann INTERFACE IMPORTED)
include_directories(nlohmann SYSTEM INTERFACE ${PROJECT_SOURCE_DIR}/third_party/nlohmann/include)

# Include moodycamel
add_library(moodycamel INTERFACE IMPORTED)
include_directories(moodycamel SYSTEM INTERFACE ${PROJECT_SOURCE_DIR}/third_party/concurrentqueue)
