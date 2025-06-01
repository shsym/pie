protoc --python_out=. --proto_path=../../api/backend l4m.proto
protoc --python_out=. --proto_path=../../api/backend l4m_vision.proto
protoc --python_out=. --proto_path=../../api/backend ping.proto
protoc --python_out=. --proto_path=../../api/backend handshake.proto

protoc --pyi_out=. --proto_path=../../api/backend l4m.proto
protoc --pyi_out=. --proto_path=../../api/backend l4m_vision.proto
protoc --pyi_out=. --proto_path=../../api/backend ping.proto
protoc --pyi_out=. --proto_path=../../api/backend handshake.proto