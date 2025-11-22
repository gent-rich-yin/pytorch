CXX = g++

CXXFLAGS = -Wall -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1

MY_TORCH_INCLUDE := -I../torch/csrc/api/include \
					-I.. \
					-I../build \
					-I../aten/src \
					-I../build/aten/src

MY_TORCH_LIB_DIR := ../build/lib
MY_TEST_SRC_DIR := ./src
MY_TEST_BUILD_DIR := ./build

prepare:
	mkdir -p ${MY_TEST_BUILD_DIR}

clean:
	rm -fr ${MY_TEST_BUILD_DIR}

create_tensor: $(MY_TEST_SRC_DIR)/create_tensor.cpp
	$(CXX) $(CXXFLAGS) $(MY_TORCH_INCLUDE) \
		-L$(MY_TORCH_LIB_DIR) \
		-lbackend_with_compiler \
		-lc10 \
		-lclog \
		-lcpuinfo \
		-lpthreadpool \
		-ltensorpipe \
		-ltensorpipe_uv \
		-ltorch \
		-ltorch_cpu \
		-ltorch_global_deps \
		-ltorch_python \
		-o $(MY_TEST_BUILD_DIR)/create_tensor \
		$(MY_TEST_SRC_DIR)/create_tensor.cpp
