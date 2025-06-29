


sudo apt-get update && sudo apt-get install -y libzmq3-dev

git clone protobuf
cmake . -Dprotobuf_FORCE_FETCH_DEPENDENCIES=ON
cmake --build . --parallel 10
sudo cmake --install .

sudo apt-get install libcbor-dev libzstd-dev
