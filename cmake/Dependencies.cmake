# RPATH stuff
# see https://cmake.org/Wiki/CMake_RPATH_handling
set(_rpath_portable_origin $ORIGIN)
# Use separate rpaths during build and install phases
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
# Don't use the install-rpath during the build phase
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${_rpath_portable_origin}")
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
  endif()
endmacro()

macro(enable_ubsan)
  if(CAFFE2_UBSAN_ENABLED)
    set(CMAKE_C_FLAGS "${UBSAN_FLAG} ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${UBSAN_FLAG} ${CMAKE_CXX_FLAGS}")
  endif()
endmacro()

caffe2_update_option(USE_XPU OFF)

# ---[ Custom Protobuf
if(CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO AND NOT INTERN_BUILD_MOBILE)
  disable_ubsan()
  include(${CMAKE_CURRENT_LIST_DIR}/ProtoBuf.cmake)  # included
  enable_ubsan()
endif()

# ---[ Threads
find_package(Threads REQUIRED)
if(TARGET Threads::Threads)
  list(APPEND Caffe2_DEPENDENCY_LIBS Threads::Threads)
else()
  message(FATAL_ERROR
      "Cannot find threading library. PyTorch requires Threads to compile.")
endif()

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

find_package(MKL QUIET)
include(${CMAKE_CURRENT_LIST_DIR}/public/mkl.cmake)
if(MKL_FOUND)
  message(STATUS "MKL libraries: ${MKL_LIBRARIES}")
  message(STATUS "MKL include directory: ${MKL_INCLUDE_DIR}")
  message(STATUS "MKL OpenMP type: ${MKL_OPENMP_TYPE}")
  message(STATUS "MKL OpenMP library: ${MKL_OPENMP_LIBRARY}")
  include_directories(AFTER SYSTEM ${MKL_INCLUDE_DIR})
  list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS caffe2::mkl)
  set(CAFFE2_USE_MKL ON)
  set(BLAS_INFO "mkl")
  set(BLAS_FOUND 1)
  set(BLAS_LIBRARIES ${MKL_LIBRARIES})
else()
  message(WARNING "MKL could not be found. Defaulting to Eigen")
  set(CAFFE2_USE_EIGEN_FOR_BLAS ON)
  set(CAFFE2_USE_MKL OFF)
endif()

set(AT_MKL_SEQUENTIAL 0)
set(USE_BLAS 1)

if(MKL_FOUND)
  if("${MKL_THREADING}" STREQUAL "SEQ")
    set(AT_MKL_SEQUENTIAL 1)
  endif()
  set(AT_MKL_ENABLED 1)
endif()

# --- [ PocketFFT
set(AT_POCKETFFT_ENABLED 0)

# ---[ Dependencies
# NNPACK and family (QNNPACK, PYTORCH_QNNPACK, and XNNPACK) can download and
# compile their dependencies in isolation as part of their build.  These dependencies
# are then linked statically with PyTorch.  To avoid the possibility of a version
# mismatch between these shared dependencies, explicitly declare our intent to these
# libraries that we are interested in using the exact same source dependencies for all.

set(DISABLE_NNPACK_AND_FAMILY OFF)

set(CAFFE2_THIRD_PARTY_ROOT "${PROJECT_SOURCE_DIR}/third_party")

if(NOT DEFINED CPUINFO_SOURCE_DIR)
  set(CPUINFO_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/cpuinfo" CACHE STRING "cpuinfo source directory")
endif()
if(NOT DEFINED FP16_SOURCE_DIR)
  set(FP16_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/FP16" CACHE STRING "FP16 source directory")
endif()
if(NOT DEFINED FXDIV_SOURCE_DIR)
  set(FXDIV_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/FXdiv" CACHE STRING "FXdiv source directory")
endif()
if(NOT DEFINED PSIMD_SOURCE_DIR)
  set(PSIMD_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/psimd" CACHE STRING "PSimd source directory")
endif()
if(NOT DEFINED PTHREADPOOL_SOURCE_DIR)
  set(PTHREADPOOL_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/pthreadpool" CACHE STRING "pthreadpool source directory")
endif()

set(CONFU_DEPENDENCIES_SOURCE_DIR ${PROJECT_BINARY_DIR}/confu-srcs
  CACHE PATH "Confu-style dependencies source directory")
set(CONFU_DEPENDENCIES_BINARY_DIR ${PROJECT_BINARY_DIR}/confu-deps
  CACHE PATH "Confu-style dependencies binary directory")

# ---[ pthreadpool
# Only add a dependency on pthreadpool if we are on a mobile build
# or are building any of the libraries in the {Q/X}NNPACK family.
if(INTERN_BUILD_MOBILE OR NOT DISABLE_NNPACK_AND_FAMILY)
  set(USE_PTHREADPOOL ON CACHE BOOL "" FORCE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_PTHREADPOOL")

  if(NOT TARGET pthreadpool)
    if(USE_SYSTEM_PTHREADPOOL)
      add_library(pthreadpool SHARED IMPORTED)
      find_library(PTHREADPOOL_LIBRARY pthreadpool)
      set_property(TARGET pthreadpool PROPERTY IMPORTED_LOCATION "${PTHREADPOOL_LIBRARY}")
      if(NOT PTHREADPOOL_LIBRARY)
        message(FATAL_ERROR "Cannot find pthreadpool")
      endif()
      message("-- Found pthreadpool: ${PTHREADPOOL_LIBRARY}")
    else()
      if(NOT DEFINED PTHREADPOOL_SOURCE_DIR)
        set(CAFFE2_THIRD_PARTY_ROOT "${PROJECT_SOURCE_DIR}/third_party")
        set(PTHREADPOOL_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/pthreadpool" CACHE STRING "pthreadpool source directory")
      endif()

      set(PTHREADPOOL_BUILD_TESTS OFF CACHE BOOL "")
      set(PTHREADPOOL_BUILD_BENCHMARKS OFF CACHE BOOL "")
      set(PTHREADPOOL_LIBRARY_TYPE "static" CACHE STRING "")
      set(PTHREADPOOL_ALLOW_DEPRECATED_API ON CACHE BOOL "")
      add_subdirectory(
        "${PTHREADPOOL_SOURCE_DIR}"
        "${CONFU_DEPENDENCIES_BINARY_DIR}/pthreadpool")
      set_property(TARGET pthreadpool PROPERTY POSITION_INDEPENDENT_CODE ON)
    endif()

    list(APPEND Caffe2_DEPENDENCY_LIBS pthreadpool)
  endif()
else()
  set(USE_PTHREADPOOL OFF CACHE BOOL "" FORCE)
endif()

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(s390x|ppc64le)$")
  # ---[ Caffe2 uses cpuinfo library in the thread pool
  # ---[ But it doesn't support s390x/powerpc and thus not used on s390x/powerpc
  if(NOT TARGET cpuinfo AND USE_SYSTEM_CPUINFO)
    add_library(cpuinfo SHARED IMPORTED)
    find_library(CPUINFO_LIBRARY cpuinfo)
    if(NOT CPUINFO_LIBRARY)
      message(FATAL_ERROR "Cannot find cpuinfo")
    endif()
    message("Found cpuinfo: ${CPUINFO_LIBRARY}")
    set_target_properties(cpuinfo PROPERTIES IMPORTED_LOCATION "${CPUINFO_LIBRARY}")
  elseif(NOT TARGET cpuinfo)
    if(NOT DEFINED CPUINFO_SOURCE_DIR)
      set(CPUINFO_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../third_party/cpuinfo" CACHE STRING "cpuinfo source directory")
    endif()

    set(CPUINFO_BUILD_TOOLS OFF CACHE BOOL "")
    set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "")
    set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "")
    set(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "")
    set(CPUINFO_LIBRARY_TYPE "static" CACHE STRING "")
    set(CPUINFO_LOG_LEVEL "error" CACHE STRING "")
    add_subdirectory(
      "${CPUINFO_SOURCE_DIR}"
      "${CONFU_DEPENDENCIES_BINARY_DIR}/cpuinfo")
    # We build static version of cpuinfo but link
    # them into a shared library for Caffe2, so they need PIC.
    set_property(TARGET cpuinfo PROPERTY POSITION_INDEPENDENT_CODE ON)
  endif()
  list(APPEND Caffe2_DEPENDENCY_LIBS cpuinfo)
endif()


# ---[ PYTORCH_QNNPACK
set(CAFFE2_THIRD_PARTY_ROOT "${PROJECT_SOURCE_DIR}/third_party")
if(USE_PYTORCH_QNNPACK)
    if(NOT DEFINED PYTORCH_QNNPACK_SOURCE_DIR)
      set(PYTORCH_QNNPACK_SOURCE_DIR "${PROJECT_SOURCE_DIR}/aten/src/ATen/native/quantized/cpu/qnnpack" CACHE STRING "QNNPACK source directory")
    endif()

    if(NOT TARGET pytorch_qnnpack)
      set(PYTORCH_QNNPACK_BUILD_TESTS OFF CACHE BOOL "")
      set(PYTORCH_QNNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "")
      set(PYTORCH_QNNPACK_LIBRARY_TYPE "static" CACHE STRING "")
      add_subdirectory(
        "${PYTORCH_QNNPACK_SOURCE_DIR}"
        "${CONFU_DEPENDENCIES_BINARY_DIR}/pytorch_qnnpack")
      # We build static versions of QNNPACK and pthreadpool but link
      # them into a shared library for Caffe2, so they need PIC.
      set_property(TARGET pytorch_qnnpack PROPERTY POSITION_INDEPENDENT_CODE ON)
      set_property(TARGET cpuinfo PROPERTY POSITION_INDEPENDENT_CODE ON)
      # QNNPACK depends on gemmlowp headers
      target_include_directories(pytorch_qnnpack PRIVATE "${CAFFE2_THIRD_PARTY_ROOT}/gemmlowp")
    endif()

    list(APPEND Caffe2_DEPENDENCY_LIBS pytorch_qnnpack)
endif()

# ---[ NNPACK
if(USE_NNPACK)
  include(${CMAKE_CURRENT_LIST_DIR}/External/nnpack.cmake)
  if(NNPACK_FOUND)
    if(TARGET nnpack)
      # ---[ NNPACK is being built together with Caffe2: explicitly specify dependency
      list(APPEND Caffe2_DEPENDENCY_LIBS nnpack)
    else()
      include_directories(SYSTEM ${NNPACK_INCLUDE_DIRS})
      list(APPEND Caffe2_DEPENDENCY_LIBS ${NNPACK_LIBRARIES})
    endif()
  else()
    message(WARNING "Not compiling with NNPACK. Suppress this warning with -DUSE_NNPACK=OFF")
    caffe2_update_option(USE_NNPACK OFF)
  endif()
endif()

# ---[ XNNPACK
if(USE_XNNPACK AND NOT USE_SYSTEM_XNNPACK)
  if(NOT DEFINED XNNPACK_SOURCE_DIR)
    set(XNNPACK_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/XNNPACK" CACHE STRING "XNNPACK source directory")
  endif()

  if(NOT DEFINED XNNPACK_INCLUDE_DIR)
    set(XNNPACK_INCLUDE_DIR "${XNNPACK_SOURCE_DIR}/include" CACHE STRING "XNNPACK include directory")
  endif()

  if(NOT TARGET XNNPACK OR NOT TARGET microkernels-prod)
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

    # Older MSVC versions don't support AVX512FP. TODO Minimum version support?
    IF(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
      set(XNNPACK_ENABLE_AVX512FP16  OFF CACHE BOOL "")
    ENDIF()

    # Conditionally disable AVX512AMX, as it requires Clang 11 or later. Note that
    # XNNPACK does conditionally compile this based on GCC version. Once it also does
    # so based on Clang version, this logic can be removed.
    IF(CMAKE_C_COMPILER_ID STREQUAL "Clang")
      IF(CMAKE_C_COMPILER_VERSION VERSION_LESS "11")
        set(XNNPACK_ENABLE_AVX512AMX OFF CACHE BOOL "")
      ENDIF()
    ENDIF()

    # Setting this global PIC flag for all XNNPACK targets.
    # This is needed for Object libraries within XNNPACK which must
    # be PIC to successfully link this static libXNNPACK with pytorch
    set(__caffe2_CMAKE_POSITION_INDEPENDENT_CODE_FLAG ${CMAKE_POSITION_INDEPENDENT_CODE})
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    add_subdirectory(
      "${XNNPACK_SOURCE_DIR}"
      "${CONFU_DEPENDENCIES_BINARY_DIR}/XNNPACK")

    # Revert to whatever it was before
    set(CMAKE_POSITION_INDEPENDENT_CODE ${__caffe2_CMAKE_POSITION_INDEPENDENT_CODE_FLAG})
  endif()

  include_directories(SYSTEM ${XNNPACK_INCLUDE_DIR})
  list(APPEND Caffe2_DEPENDENCY_LIBS XNNPACK microkernels-prod)
elseif(NOT TARGET XNNPACK AND USE_SYSTEM_XNNPACK)
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
if(BUILD_TEST OR BUILD_MOBILE_BENCHMARK OR BUILD_MOBILE_TEST)
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
endif()

# ---[ FBGEMM
if(USE_FBGEMM)
  set(CAFFE2_THIRD_PARTY_ROOT "${PROJECT_SOURCE_DIR}/third_party")
  if(NOT DEFINED FBGEMM_SOURCE_DIR)
    set(FBGEMM_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/fbgemm" CACHE STRING "FBGEMM source directory")
  endif()
  if(USE_FBGEMM AND NOT TARGET fbgemm)
    set(FBGEMM_BUILD_TESTS OFF CACHE BOOL "")
    set(FBGEMM_BUILD_BENCHMARKS OFF CACHE BOOL "")
    set(FBGEMM_LIBRARY_TYPE "static" CACHE STRING "")
    add_subdirectory("${FBGEMM_SOURCE_DIR}")

    target_compile_options_if_supported(asmjit -Wno-unused-but-set-variable)
    target_compile_options_if_supported(asmjit -Wno-unused-variable)
  endif()
  if(USE_FBGEMM)
    list(APPEND Caffe2_DEPENDENCY_LIBS fbgemm)
  endif()
endif()

if(USE_FBGEMM)
  caffe2_update_option(USE_FBGEMM ON)
else()
  caffe2_update_option(USE_FBGEMM OFF)
endif()

# ---[ NUMA
if(USE_NUMA)
  find_package(Numa)  # not found
  if(NOT NUMA_FOUND)
    message(WARNING "Not compiling with NUMA. Suppress this warning with -DUSE_NUMA=OFF")
    caffe2_update_option(USE_NUMA OFF)
  endif()
endif()

if(USE_ITT)
  find_package(ITT)  # found
  if(ITT_FOUND)
    include_directories(SYSTEM ${ITT_INCLUDE_DIR})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${ITT_LIBRARIES})
    list(APPEND TORCH_PYTHON_LINK_LIBRARIES ${ITT_LIBRARIES})
  else()
    message(WARNING "Not compiling with ITT. Suppress this warning with -DUSE_ITT=OFF")
    set(USE_ITT OFF CACHE BOOL "" FORCE)
    caffe2_update_option(USE_ITT OFF)
  endif()
endif()

# ---[ Caffe2 depends on FP16 library for half-precision conversions
if(NOT TARGET fp16 AND NOT USE_SYSTEM_FP16)
  set(CAFFE2_THIRD_PARTY_ROOT "${PROJECT_SOURCE_DIR}/third_party")
  # PSIMD is required by FP16
  if(NOT DEFINED PSIMD_SOURCE_DIR)
    set(PSIMD_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/psimd" CACHE STRING "PSimd source directory")
  endif()
  if(NOT DEFINED FP16_SOURCE_DIR)
    set(FP16_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/FP16" CACHE STRING "FP16 source directory")
  endif()

  set(FP16_BUILD_TESTS OFF CACHE BOOL "")
  set(FP16_BUILD_BENCHMARKS OFF CACHE BOOL "")
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0.0")
    message(WARNING "FP16 is only cmake-2.8 compatible")
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    add_subdirectory(
      "${FP16_SOURCE_DIR}"
      "${CONFU_DEPENDENCIES_BINARY_DIR}/FP16")
    unset(CMAKE_POLICY_VERSION_MINIMUM)
  else()
    add_subdirectory(
      "${FP16_SOURCE_DIR}"
      "${CONFU_DEPENDENCIES_BINARY_DIR}/FP16")
  endif()
elseif(NOT TARGET fp16 AND USE_SYSTEM_FP16)
  add_library(fp16 STATIC "/usr/include/fp16.h")
  set_target_properties(fp16 PROPERTIES LINKER_LANGUAGE C)
endif()
list(APPEND Caffe2_DEPENDENCY_LIBS fp16)

# ---[ Python Interpreter
# If not given a Python installation, then use the current active Python
if(NOT Python_EXECUTABLE)
  execute_process(
    COMMAND "which" "python3" RESULT_VARIABLE _exitcode OUTPUT_VARIABLE _py_exe)
  if(${_exitcode} EQUAL 0)
    if(NOT MSVC)
      string(STRIP ${_py_exe} Python_EXECUTABLE)
    endif()
    message(STATUS "Setting Python to ${Python_EXECUTABLE}")
  endif()
endif()

# ---[ EIGEN
# Due to license considerations, we will only use the MPL2 parts of Eigen.
set(EIGEN_MPL2_ONLY 1)
message(STATUS "Using third party subdirectory Eigen.")
set(EIGEN3_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../third_party/eigen)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

if(BUILD_PYTHON)
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
if(USE_SYSTEM_PYBIND11)   # OFF
  find_package(pybind11 CONFIG)
  if(NOT pybind11_FOUND)
    find_package(pybind11)
  endif()
  if(NOT pybind11_FOUND)
    message(FATAL "Cannot find system pybind11")
  endif()
else()
    message(STATUS "Using third_party/pybind11.")
    set(pybind11_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/../third_party/pybind11/include)
    install(DIRECTORY ${pybind11_INCLUDE_DIRS}
            DESTINATION ${CMAKE_INSTALL_PREFIX}
            FILES_MATCHING PATTERN "*.h")
endif()
message(STATUS "pybind11 include dirs: " "${pybind11_INCLUDE_DIRS}")
add_library(pybind::pybind11 INTERFACE IMPORTED)
target_include_directories(pybind::pybind11 SYSTEM INTERFACE ${pybind11_INCLUDE_DIRS})
target_link_libraries(pybind::pybind11 INTERFACE Python::Module)

# ---[ OpenTelemetry API headers
find_package(OpenTelemetryApi)
if(NOT OpenTelemetryApi_FOUND)
  message(STATUS "Using third_party/opentelemetry-cpp.")
  set(OpenTelemetryApi_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/../third_party/opentelemetry-cpp/api/include)
endif()
message(STATUS "opentelemetry api include dirs: " "${OpenTelemetryApi_INCLUDE_DIRS}")
add_library(opentelemetry::api INTERFACE IMPORTED)
target_include_directories(opentelemetry::api SYSTEM INTERFACE ${OpenTelemetryApi_INCLUDE_DIRS})

# ---[ MPI
if(USE_MPI)
  find_package(MPI)
  if(MPI_CXX_FOUND)
    message(STATUS "MPI support found")
    message(STATUS "MPI compile flags: " ${MPI_CXX_COMPILE_FLAGS})
    message(STATUS "MPI include path: " ${MPI_CXX_INCLUDE_PATH})
    message(STATUS "MPI LINK flags path: " ${MPI_CXX_LINK_FLAGS})
    message(STATUS "MPI libraries: " ${MPI_CXX_LIBRARIES})
    find_program(OMPI_INFO
      NAMES ompi_info
      HINTS ${MPI_CXX_LIBRARIES}/../bin)
    if(OMPI_INFO)
      execute_process(COMMAND ${OMPI_INFO}
                      OUTPUT_VARIABLE _output)
      if(_output MATCHES "smcuda")
        message(STATUS "Found OpenMPI with CUDA support built.")
      else()
        message(WARNING "OpenMPI found, but it is not built with CUDA support.")
        set(CAFFE2_FORCE_FALLBACK_CUDA_MPI 1)
      endif()
    endif()
  else()
    message(WARNING "Not compiling with MPI. Suppress this warning with -DUSE_MPI=OFF")
    caffe2_update_option(USE_MPI OFF)
  endif()
endif()

# ---[ OpenMP
if(USE_OPENMP AND NOT TARGET caffe2::openmp)
  include(${CMAKE_CURRENT_LIST_DIR}/Modules/FindOpenMP.cmake)
  if(OPENMP_FOUND)
    message(STATUS "Adding OpenMP CXX_FLAGS: " ${OpenMP_CXX_FLAGS})
    if(OpenMP_CXX_LIBRARIES)
      message(STATUS "Will link against OpenMP libraries: ${OpenMP_CXX_LIBRARIES}")
    endif()
    add_library(caffe2::openmp INTERFACE IMPORTED)
    target_link_libraries(caffe2::openmp INTERFACE OpenMP::OpenMP_CXX)
    list(APPEND Caffe2_DEPENDENCY_LIBS caffe2::openmp)
  else()
    message(WARNING "Not compiling with OpenMP. Suppress this warning with -DUSE_OPENMP=OFF")
    caffe2_update_option(USE_OPENMP OFF)
  endif()
endif()

find_path(nvtx3_dir NAMES nvtx3 PATHS "${PROJECT_SOURCE_DIR}/third_party/NVTX/c/include" NO_DEFAULT_PATH)
find_package_handle_standard_args(nvtx3 DEFAULT_MSG nvtx3_dir)
if(nvtx3_FOUND)
  add_library(torch::nvtx3 INTERFACE IMPORTED)
  target_include_directories(torch::nvtx3 INTERFACE "${nvtx3_dir}")
  target_compile_definitions(torch::nvtx3 INTERFACE TORCH_CUDA_USE_NVTX3)
else()
  message(WARNING "Cannot find NVTX3, find old NVTX instead")
  add_library(torch::nvtoolsext INTERFACE IMPORTED)
  set_property(TARGET torch::nvtoolsext PROPERTY INTERFACE_LINK_LIBRARIES CUDA::nvToolsExt)
endif()

# ---[ HIP
if(USE_ROCM)   # ON
  include(${CMAKE_CURRENT_LIST_DIR}/public/LoadHIP.cmake)
  if(PYTORCH_FOUND_HIP)  # OFF, whole if block is skipped
    message(INFO "Compiling with HIP for AMD.")
    caffe2_update_option(USE_ROCM ON)

    if(USE_NCCL AND NOT USE_SYSTEM_NCCL)
      message(INFO "Forcing USE_SYSTEM_NCCL to ON since it's required by using RCCL")
      caffe2_update_option(USE_SYSTEM_NCCL ON)
    endif()

    list(APPEND HIP_CXX_FLAGS -fPIC)
    list(APPEND HIP_CXX_FLAGS -D__HIP_PLATFORM_AMD__=1)
    list(APPEND HIP_CXX_FLAGS -DCUDA_HAS_FP16=1)
    list(APPEND HIP_CXX_FLAGS -DUSE_ROCM)
    list(APPEND HIP_CXX_FLAGS -D__HIP_NO_HALF_OPERATORS__=1)
    list(APPEND HIP_CXX_FLAGS -D__HIP_NO_HALF_CONVERSIONS__=1)
    list(APPEND HIP_CXX_FLAGS -DTORCH_HIP_VERSION=${TORCH_HIP_VERSION})
    list(APPEND HIP_CXX_FLAGS -Wno-shift-count-negative)
    list(APPEND HIP_CXX_FLAGS -Wno-shift-count-overflow)
    list(APPEND HIP_CXX_FLAGS -Wno-duplicate-decl-specifier)
    list(APPEND HIP_CXX_FLAGS -DCAFFE2_USE_MIOPEN)
    list(APPEND HIP_CXX_FLAGS -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)
    list(APPEND HIP_CXX_FLAGS -std=c++17)
    list(APPEND HIP_CXX_FLAGS -DHIPBLAS_V2)
    list(APPEND HIP_CXX_FLAGS -DHIP_ENABLE_WARP_SYNC_BUILTINS)
    if(HIPBLASLT_OUTER_VEC)
      list(APPEND HIP_CXX_FLAGS -DHIPBLASLT_OUTER_VEC)
    endif()
    if(HIPBLASLT_VEC_EXT)
      list(APPEND HIP_CXX_FLAGS -DHIPBLASLT_VEC_EXT)
    endif()
    if(USE_ROCM_CK_GEMM)
      list(APPEND HIP_CXX_FLAGS -DUSE_ROCM_CK_GEMM)
    endif()
    list(APPEND HIP_HIPCC_FLAGS --offload-compress)
    add_definitions(-DROCM_VERSION=${ROCM_VERSION_DEV_INT})
    add_definitions(-DTORCH_HIP_VERSION=${TORCH_HIP_VERSION})
    message("TORCH_HIP_VERSION=${TORCH_HIP_VERSION} is added as a compiler defines")

    if(CMAKE_BUILD_TYPE MATCHES Debug)
       list(APPEND HIP_CXX_FLAGS -g2)
       list(APPEND HIP_CXX_FLAGS -O0)
       list(APPEND HIP_HIPCC_FLAGS -fdebug-info-for-profiling)
    endif(CMAKE_BUILD_TYPE MATCHES Debug)

    # needed for compat with newer versions of hip-clang that introduced C++20 mangling rules
    list(APPEND HIP_HIPCC_FLAGS -fclang-abi-compat=17)

    set(HIP_CLANG_FLAGS ${HIP_CXX_FLAGS})
    # Ask hcc to generate device code during compilation so we can use
    # host linker to link.
    list(APPEND HIP_CLANG_FLAGS -fno-gpu-rdc)
    foreach(pytorch_rocm_arch ${PYTORCH_ROCM_ARCH})
      list(APPEND HIP_CLANG_FLAGS --offload-arch=${pytorch_rocm_arch})
    endforeach()

    set(Caffe2_HIP_INCLUDE
       $<INSTALL_INTERFACE:include> ${Caffe2_HIP_INCLUDE})
    # This is needed for library added by hip_add_library (same for hip_add_executable)
    hip_include_directories(${Caffe2_HIP_INCLUDE})

    set(Caffe2_PUBLIC_HIP_DEPENDENCY_LIBS
      hip::amdhip64 MIOpen hiprtc::hiprtc) # libroctx will be linked in with MIOpen

    # Math libraries
    list(APPEND Caffe2_PUBLIC_HIP_DEPENDENCY_LIBS
      roc::hipblas roc::rocblas hip::hipfft hip::hiprand roc::hipsparse roc::hipsolver roc::hipblaslt roc::rocsolver)
    # hipsparselt is an optional component that will eventually be enabled by default.
    if(hipsparselt_FOUND)
      list(APPEND Caffe2_PUBLIC_HIP_DEPENDENCY_LIBS
        roc::hipsparselt
      )
    endif()

    # ---[ Kernel asserts
    # Kernel asserts is disabled for ROCm by default.
    # It can be turned on by turning on the env USE_ROCM_KERNEL_ASSERT to the build system.
    if(USE_ROCM_KERNEL_ASSERT)
      message(STATUS "Enabling Kernel Assert for ROCm")
    else()
      message(STATUS "Disabling Kernel Assert for ROCm")
    endif()

  else()
    caffe2_update_option(USE_ROCM OFF)   # this is executed
  endif()
endif()

# ---[ NCCL
if(USE_NCCL)   # ON
  if(NOT (USE_CUDA OR USE_ROCM))  # NOT (OFF OR OFF) => ON
    message(WARNING
        "Not using CUDA/ROCM, so disabling USE_NCCL. Suppress this warning with "
        "-DUSE_NCCL=OFF.")
    caffe2_update_option(USE_NCCL OFF)
  elseif(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(WARNING "NCCL is currently only supported under Linux.")
    caffe2_update_option(USE_NCCL OFF)
  elseif(USE_CUDA)
    include(${CMAKE_CURRENT_LIST_DIR}/External/nccl.cmake)
    list(APPEND Caffe2_CUDA_DEPENDENCY_LIBS __caffe2_nccl)
  elseif(USE_ROCM)
    include(${CMAKE_CURRENT_LIST_DIR}/External/rccl.cmake)
    list(APPEND Caffe2_CUDA_DEPENDENCY_LIBS __caffe2_nccl)
  endif()
endif()

# ---[ XCCL
if(USE_XCCL)    # ON
  if(NOT USE_XPU)   # ON
    message(WARNING "Not using XPU, so disabling USE_XCCL. Suppress this warning with -DUSE_XCCL=OFF.")
    caffe2_update_option(USE_XCCL OFF)
  endif()
endif()

if(USE_TENSORPIPE)  # ON
  set(TP_BUILD_LIBUV ON CACHE BOOL "" FORCE)
  add_compile_options(-DTORCH_USE_LIBUV)
  include_directories(BEFORE SYSTEM ${CMAKE_CURRENT_LIST_DIR}/../third_party/tensorpipe/third_party/libuv/include)
  set(TP_STATIC_OR_SHARED STATIC CACHE STRING "" FORCE)

  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/tensorpipe)
  # Suppress warning to unblock libnop compilation by clang-17
  # See https://github.com/pytorch/pytorch/issues/151316
  target_compile_options_if_supported(tensorpipe -Wno-missing-template-arg-list-after-template-kw)

  list(APPEND Caffe2_DEPENDENCY_LIBS tensorpipe)
  list(APPEND Caffe2_DEPENDENCY_LIBS nlohmann)
  list(APPEND Caffe2_DEPENDENCY_LIBS moodycamel)
endif()

if(USE_GLOO)   # ON
  if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(WARNING "Gloo can only be used on 64-bit systems.")
    caffe2_update_option(USE_GLOO OFF)
  else()
    # Don't install gloo
    set(GLOO_INSTALL OFF CACHE BOOL "" FORCE)
    set(GLOO_STATIC_OR_SHARED STATIC CACHE STRING "" FORCE)

    if(USE_GLOO_IBVERBS)   # OF
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
    if(USE_ROCM)
      set(ENV{GLOO_ROCM_ARCH} "${PYTORCH_ROCM_ARCH}")
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

    # Add explicit dependency since NCCL is built from third_party.
    # Without dependency, make -jN with N>1 can fail if the NCCL build
    # hasn't finished when CUDA targets are linked.
    if(NOT USE_SYSTEM_NCCL AND USE_NCCL AND NOT USE_ROCM)  # (NOT OFF AND OFF AND NOT OFF) => OFF
      add_dependencies(gloo_cuda nccl_external)
    endif()
    # Pick the right dependency depending on USE_CUDA
    list(APPEND Caffe2_DEPENDENCY_LIBS gloo)
    add_compile_options(-DCAFFE2_USE_GLOO)
  endif()
endif()

# ---[ Onnx
if(CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO AND NOT INTERN_DISABLE_ONNX)  # (ON AND NOT OFF) => ON
  if(EXISTS "${CAFFE2_CUSTOM_PROTOC_EXECUTABLE}")    # OFF
    set(ONNX_CUSTOM_PROTOC_EXECUTABLE ${CAFFE2_CUSTOM_PROTOC_EXECUTABLE})
  endif()
  set(TEMP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
  set(BUILD_SHARED_LIBS OFF)
  set(ONNX_USE_MSVC_STATIC_RUNTIME ${CAFFE2_USE_MSVC_STATIC_RUNTIME})
  set(ONNX_USE_LITE_PROTO ${CAFFE2_USE_LITE_PROTO})
  # If linking local protobuf, make sure ONNX has the same protobuf
  # patches as Caffe2 and Caffe proto. This forces some functions to
  # not be inline and instead route back to the statically-linked protobuf.
  if(CAFFE2_LINK_LOCAL_PROTOBUF)
    set(ONNX_PROTO_POST_BUILD_SCRIPT ${PROJECT_SOURCE_DIR}/cmake/ProtoBufPatch.cmake)
  endif()
  if(ONNX_ML)   # ON
    add_definitions(-DONNX_ML=1)
  endif()
  add_definitions(-DONNXIFI_ENABLE_EXT=1)
  set(Python3_EXECUTABLE "${Python_EXECUTABLE}")
  if(NOT USE_SYSTEM_ONNX)  # NOT OFF => ON
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/onnx EXCLUDE_FROM_ALL)
  endif()

  add_definitions(-DONNX_NAMESPACE=${ONNX_NAMESPACE})
  caffe2_interface_library(onnx onnx_library)
  list(APPEND Caffe2_DEPENDENCY_WHOLE_LINK_LIBS onnx_library)
  # Recover the build shared libs option.
  set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS})
endif()

# --[ ATen checks
set(USE_LAPACK 0)

if(NOT INTERN_BUILD_MOBILE)
  set(TORCH_CUDA_ARCH_LIST $ENV{TORCH_CUDA_ARCH_LIST})
  string(APPEND CMAKE_CUDA_FLAGS " $ENV{TORCH_NVCC_FLAGS}")
  set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

  # Top-level build config
  ############################################
  # Flags
  # When using MSVC
  # Detect CUDA architecture and get best NVCC flags
  # finding cuda must be first because other things depend on the result
  #
  # NB: We MUST NOT run this find_package if NOT USE_CUDA is set, because upstream
  # FindCUDA has a bug where it will still attempt to make use of NOTFOUND
  # compiler variables to run various probe tests.  We could try to fix
  # this, but since FindCUDA upstream is subsumed by first-class support
  # for CUDA language, it seemed not worth fixing.

  if(WERROR)   # OFF
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND ${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER_EQUAL 13)
      string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler -Wno-dangling-reference ")
    endif()
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler -Wno-extra-semi ")
    endif()
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND ${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER_EQUAL 13))
      string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler -Werror -Xcompiler -Wno-error=sign-compare ")
    endif()
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

  find_package(LAPACK)
  if(LAPACK_FOUND)    # ON
    set(USE_LAPACK 1)
    list(APPEND Caffe2_PRIVATE_DEPENDENCY_LIBS ${LAPACK_LIBRARIES})
  endif()

  if(NOT USE_CUDA)
    message("disabling CUDA because NOT USE_CUDA is set")
    set(AT_CUDA_ENABLED 0)
  else()
    set(AT_CUDA_ENABLED 1)
  endif()

  if(NOT USE_ROCM)
    message("disabling ROCM because NOT USE_ROCM is set")
    message(STATUS "MIOpen not found. Compiling without MIOpen support")
    set(AT_ROCM_ENABLED 0)
  else()
    include_directories(BEFORE ${MIOPEN_INCLUDE_DIRS})
    set(AT_ROCM_ENABLED 1)
  endif()

  if(USE_MKLDNN)  # ON
    if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
      message(WARNING
        "x64 operating system is required for MKLDNN. "
        "Not compiling with MKLDNN. "
        "Turn this warning off by USE_MKLDNN=OFF.")
      set(USE_MKLDNN OFF)
    endif()
    if(USE_MKLDNN_ACL)
      set(AT_MKLDNN_ACL_ENABLED 1)
    endif()
  endif()
  if(USE_MKLDNN)   # ON
    include(${CMAKE_CURRENT_LIST_DIR}/public/mkldnn.cmake)
    if(MKLDNN_FOUND)   # ON
      set(AT_MKLDNN_ENABLED 1)
      include_directories(AFTER SYSTEM ${MKLDNN_INCLUDE_DIR})
    else()
      message(WARNING "MKLDNN could not be found.")
      caffe2_update_option(USE_MKLDNN OFF)
    endif()
  else()
    message("disabling MKLDNN because USE_MKLDNN is not set")
  endif()

  if(UNIX AND NOT APPLE)
     include(CheckLibraryExists)
     # https://github.com/libgit2/libgit2/issues/2128#issuecomment-35649830
     CHECK_LIBRARY_EXISTS(rt clock_gettime "time.h" NEED_LIBRT)
     if(NEED_LIBRT)
       list(APPEND Caffe2_DEPENDENCY_LIBS rt)
       set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} rt)
     endif(NEED_LIBRT)
  endif(UNIX AND NOT APPLE)

  if(UNIX)
    set(CMAKE_EXTRA_INCLUDE_FILES "sys/mman.h")
    CHECK_FUNCTION_EXISTS(mmap HAVE_MMAP)
    if(HAVE_MMAP)
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
