protoc --python_out=. --proto_path=../proto l4m.proto
protoc --python_out=. --proto_path=../proto l4m_vision.proto
protoc --python_out=. --proto_path=../proto ping.proto
protoc --python_out=. --proto_path=../proto handshake.proto

protoc --pyi_out=. --proto_path=../proto l4m.proto
protoc --pyi_out=. --proto_path=../proto l4m_vision.proto
protoc --pyi_out=. --proto_path=../proto ping.proto
protoc --pyi_out=. --proto_path=../proto handshake.proto