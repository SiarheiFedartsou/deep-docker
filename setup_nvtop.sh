git clone https://github.com/Syllo/nvtop.git

mkdir -p nvtop/build
pushd nvtop/build

cmake ..
cmake .. -DNVML_RETRIEVE_HEADER_ONLINE=True

make
make install

popd

rm -rf nvtop
