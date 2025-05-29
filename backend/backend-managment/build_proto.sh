#!/bin/bash
# Build protobuf files for the management service

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Path to the proto files
PROTO_DIR="$DIR/../../api/backend"

# Generate Python stubs from proto files
echo "Generating protobuf files..."

protoc --python_out="$DIR" --proto_path="$PROTO_DIR" "$PROTO_DIR/handshake.proto"

if [ $? -eq 0 ]; then
    echo "Protobuf files generated successfully"
else
    echo "Error generating protobuf files"
    exit 1
fi
