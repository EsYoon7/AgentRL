# Deployment

To get started, you should deploy the controller and at least one task worker.

With the support of gRPC transport, the controller can be deployed only once,
and multiple task workers on different machines, even different clusters,
can connect to that one controller.

## Table of Contents

- [Controller](#controller)
  - [Prebuilt Binaries](#prebuilt-binaries)
  - [Running with Docker](#running-with-docker)
  - [Building Standalone Binary from Source](#building-standalone-binary-from-source)
- [Task Worker](#task-worker)
  - [Installing as a Python Package](#installing-as-a-python-package)
  - [Running with Docker](#running-with-docker-1)
  - [Configuration](#configuration)

## Controller

The controller is implemented in Go, and can be run as a standalone binary.
To deploy the controller, you can choose one of the following methods:

### Prebuilt Binaries

Prebuilt binaries can be found at the [Releases](https://github.com/thudm/agentrl/releases) page.
Download the latest version corresponding to your OS and architecture, extract the archive,
and run the controller with:

```shell
./agentrl controller
```

### Running with Docker

```shell
docker run --name agentrl-controller -p 5020:5020 jingbh/agentrl-controller:latest controller
```

Source of the Dockerfile to build the controller can be found at [`controller/Dockerfile`](controller/Dockerfile).

> [!NOTE]  
> It is known that in certain setups, connections to the controller may be killed if it takes too long.
> If this happens, try re-creating the container using host network, or run the controller binary on the host directly.

### Building Standalone Binary from Source

The overall building process of the controller can be found in the [Dockerfile](controller/Dockerfile)

1. **Prepare Dependencies**:

   To successfully build the controller, you should have [`Go`](https://go.dev/dl)
   and [`protoc`](https://github.com/protocolbuffers/protobuf/releases) (at least v27.0) installed.

   For macOS users, you can quickly install both with:
   ```shell
   brew install go protobuf
   ```

   Then, install gRPC compilers for Go with:
   ```shell
   go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
   go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
   ```

2. **Build gRPC protos**:

   ```shell
   export PATH="$PATH:$(go env GOPATH)/bin"
   mkdir -p controller/internal/pb
   protoc --proto_path=proto \
      --go_out=controller/internal/pb --go_opt=paths=source_relative \
      --go-grpc_out=controller/internal/pb --go-grpc_opt=paths=source_relative \
      proto/*.proto
   ```

3. **Build Dashboard**:

   > This step is optional.
   > If you do not need the dashboard, create an empty directory at `controller/dashboard/dist`
   > so that the Go compiler does not complain about missing assets.
   > Then you can safely skip this step.

   Building the dashboard additionally requires `node` and `yarn` to be installed.

   ```shell
   cd controller/dashboard
   yarn install --frozen-lockfile
   yarn build
   ```

4. **Build the Controller**:

   ```shell
   cd controller
   go build -o agentrl .
   ```
   
   The binary `agentrl` should be generated in the `controller` directory.

5. **Run the Controller**:

   ```shell
   ./agentrl controller
   ```

## Task Worker

The task worker can be run directly as a Python script,
but it is more recommended to run it containerized for easier scaling.

### Installing as a Python Package

The task worker can be installed as a Python package from PyPI:

```shell
pip install agentrl-worker
```

Then run the task worker with:

```shell
python -m agentrl.worker --config CONFIG [--controller CONTROLLER] [--self SELF] name
```

- `--config`: Path to the YAML configuration file of the task worker.
- `--controller`: URL of the controller API, default to `http://localhost:5020/api`  
  To use the gRPC transport, specify this address in the form of `grpc://host:port`.
- `--self`: URL of the task worker API, default to `http://localhost:5021/api`  
  This address must be accessible by the controller, unless gRPC transport is used.
- `name`: Name of the task to run, should match a top-level entry in the configuration file.

### Running with Docker

Since each task's task workers may have different dependencies,
we do not provide pre-built Docker images for the task worker.

Refer to the template Dockerfile at `extra/docker/task-worker-template.Dockerfile`
for building a Docker image for the task worker.

### Configuration

The task worker can be configured with a YAML file, consider this example:

```yaml
default:
  module: tasks.demo.DemoTask
  parameters:
    concurrency: 32

demo-std:
  parameters:
    name: demo-std
    data_file: "data/demo/standard.jsonl"

demo-env_train:
  parameters:
    name: demo-env_train
    data_file: "data/demo/train.jsonl"
```

Two tasks are defined in this configuration, extending the default configuration.

On start, the task worker loads each top-level entry as a task, with sub-entries merged into the default configuration.
Then, the task worker will import the Task class specified in the `module` field, and instantiate it with the parameters defined in the configuration.

Runtime configuration can be done by either modifying the yaml configuration files of the task or by setting environment variables.
The environment variables should be named as the configuration keys in uppercase, replacing dots, dashes with underscores.

For example, to override the `data_file` parameter of the `demo-env_train` task,
you can set the environment variable `DEMO_ENV_TRAIN_PARAMETERS_DATA_FILE` to the desired value.

Note that the `default` part is merged into each part and is not present at runtime
so you cannot use `DEFAULT_` environment variables to override a parameter on all tasks at once.

