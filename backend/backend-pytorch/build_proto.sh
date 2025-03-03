protoc --pyi_out=. --proto_path=../../backend-api l4m.proto
protoc --python_out=. --proto_path=../../backend-api l4m.proto
protoc --pyi_out=. --proto_path=../../backend-api l4m_vision.proto
protoc --python_out=. --proto_path=../../backend-api l4m_vision.proto
protoc --pyi_out=. --proto_path=../../backend-api ping.proto
protoc --python_out=. --proto_path=../../backend-api ping.proto

