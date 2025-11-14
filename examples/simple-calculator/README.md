# simple-calculator

A minimal example task showing how the framework works.

The task definition `task.py` contains a simple task implemented with the framework
which requires the agent to call a calculator tool to compute the result of simple arithmetic expressions.

The evaluation script `eval.py` contains the basic flow of how to call the framework to run an evaluation.

## Setup

### 1. Start a Controller

```shell
docker run --name agentrl-controller -p 5020:5020 jingbh/agentrl-controller:latest controller
```

Keep the controller running in the background.

### 2. Start a Worker

Install dependencies in a Python virtual environment if you haven't already:

```shell
(venv) pip install agentrl-worker
```

Or from a local clone:

```shell
(venv) pip install -e ../../worker
```

Then:

```shell
(venv) python -m agentrl.worker --config ./config.yaml --controller grpc://localhost:5020 simple-calculator
```

Keep the worker running in the background.

### 3. Run Evaluation

Install the agentrl-eval package if you haven't already:

```shell
(venv) pip install -e ../../eval[tui]
```

Set your OpenAI API key (and optionally a base URL) via environment variables:

```shell
export OPENAI_API_KEY="your_openai_api_key"
export OPENAI_BASE_URL="your_openai_api_base_url"  # Optional
```

Then run the evaluation script:

```shell
(venv) agentrl-eval -m gpt-5-nano-2025-08-07 -n 1 -j 1 -c http://localhost:5020/api simple-calculator
```
