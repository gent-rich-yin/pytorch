cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(c10 CXX)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard whose features are requested to build this target.")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Main build file for the C10 library.
#
# Note that the C10 library should maintain minimal dependencies - especially,
# it should not depend on any library that is implementation specific or
# backend specific. It should in particular NOT be dependent on any generated
# protobuf header files, because protobuf header files will transitively force
# one to link against a specific protobuf version.

set(C10_LIB c10)
set(C10_USE_GFLAGS ${USE_GFLAGS})  # also used in torch/headeronly  # OFF
set(C10_USE_GLOG ${USE_GLOG})  # also used in torch/headeronly  # OFF
set(C10_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})  # also used in torch/headeronly  # ON
set(C10_USE_NUMA ${USE_NUMA})  # also used in torch/headeronly  # OFF
set(C10_USE_MSVC_STATIC_RUNTIME ${CAFFE2_USE_MSVC_STATIC_RUNTIME})  # also used in torch/headeronly  # OFF
set(C10_USE_ROCM_KERNEL_ASSERT ${USE_ROCM_KERNEL_ASSERT})  # also used in torch/headeronly  # OFF

# Note: if you want to add ANY dependency to the c10 library, make sure you
# check with the core PyTorch developers as the dependency will be
# transitively passed on to all libraries dependent on PyTorch.
# Note by Richard: these folders are not included:
# benchmark, cuda, hip, metal, test, xpu
file(GLOB C10_SRCS
        *.cpp
        core/*.cpp
        core/impl/*.cpp
        mobile/*.cpp
        macros/*.cpp
        util/*.cpp
      )
file(GLOB C10_HEADERS
        *.h
        core/*.h
        core/impl/*.h
        mobile/*.h
        macros/*.h
        util/*.h
      )

option(C10_USE_IWYU "Use include-what-you-use to clean up header inclusion" OFF)  # OFF

find_package(Backtrace)  # Backtrace_FOUND: ON

add_library(c10 ${C10_SRCS} ${C10_HEADERS})
torch_compile_options(c10)  # torch_compile_options is defined in <CMake_SOURCE_DIR>/cmake/public/utils.cmake
# If building shared library, set dllimport/dllexport proper.
target_compile_options(c10 PRIVATE "-DC10_BUILD_MAIN_LIB")
# Enable hidden visibility if compiler supports it.
target_compile_options(c10 PRIVATE "-fvisibility=hidden")  # already set in torch_compile_options(c10) above
# Assume Backtrace_FOUND is ON
target_compile_definitions(c10 PRIVATE SUPPORTS_BACKTRACE=1)

target_link_libraries(c10 PUBLIC headeronly)
target_link_libraries(c10 PRIVATE fmt::fmt-header-only)
target_link_libraries(c10 PRIVATE nlohmann)
target_link_libraries(c10 PRIVATE moodycamel)
target_link_libraries(c10 PRIVATE cpuinfo)
target_link_libraries(c10 PRIVATE ${Backtrace_LIBRARIES})
target_link_libraries(c10 PRIVATE Threads::Threads)
target_link_libraries(c10 PRIVATE dl)

# Assume Backtrace_FOUND is ON
target_include_directories(c10 PRIVATE ${Backtrace_INCLUDE_DIRS})
target_include_directories(
    c10 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>)

add_subdirectory(test)
add_subdirectory(benchmark)

# ---[ Installation
# Note: for now, we will put all export path into one single Caffe2Targets group
# to deal with the cmake deployment need. Inside the Caffe2Targets set, the
# individual libraries like libc10.so and libcaffe2.so are still self-contained.
install(TARGETS c10 EXPORT Caffe2Targets DESTINATION lib)

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")
