

install abseil
https://abseil.io/docs/cpp/quickstart-cmake


install protobuf
https://github.com/protocolbuffers/protobuf/blob/main/cmake/README.md


git clone protobuf
cmake . -Dprotobuf_FORCE_FETCH_DEPENDENCIES=ON
cmake --build . --parallel 10
sudo cmake --install .

