
# PIE Torch Backend



## Installation

First, build proto files:
```bash
./build_proto.sh
```
This script generates the necessary Python files from the protobuf definitions.

We recommend using `uv` for installing and managing dependencies.
Follow the [official instructions](https://docs.astral.sh/uv/getting-started/installation/) to install uv.


Once uv is installed, run:
```bash
uv sync
```
This will install all required and nested dependencies based on the project’s configuration.

To start the backend server, run:

```
uv run server.py --config dev.toml
```
This command uses the specified development configuration (dev.toml).