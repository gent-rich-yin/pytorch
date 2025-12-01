unset CONDA_PREFIX
export USE_CUDA=0
export USE_NINJA=0

cd ~/pytorch
rm -fr build
mkdir build
cd build
export ts=$(date +%s)
cmake -DBUILD_PYTHON=True -DBUILD_TEST=True \
    -DCMAKE_INSTALL_PREFIX=/home/user/pytorch/torch \
    -DCMAKE_PREFIX_PATH=/home/user/anaconda3/lib/python3.13/site-packages:/home/user/anaconda3/envs/torch-dev \
    -DPython_EXECUTABLE=/home/user/anaconda3/bin/python3.13 \
    -DPython_NumPy_INCLUDE_DIR=/home/user/anaconda3/lib/python3.13/site-packages/numpy/_core/include \
    -DTORCH_BUILD_VERSION=2.10.0a0 -DUSE_NUMPY=True -DUSE_CUDA=0 \
    /home/user/pytorch > /tmp/build-${ts}.txt 2>&1

cmake --build . --target install --config Release -j $(cat /proc/cpuinfo | grep "processor" | wc -l) >> /tmp/build-${ts}.txt 2>&1
