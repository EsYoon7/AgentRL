### Note:
# This is a template Dockerfile for creating a task worker image based on custom task code.
# The file cannot be built directly, and should be copied to a specific task folder and modified as needed.

# 1. base image: set to the python version that your task requires
FROM python:3.13

# 2. set global environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TQDM_DISABLE=1

WORKDIR /app

# 3. (optional) install additional system packages that your task requires
# RUN apt-get update && \
#     apt-get install -y build-essential && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# 4. (optional) if your task is an adaptation of other existing tasks, clone their repository and install the code
# ADD https://github.com/some-org/some-repo.git /usr/src/some-repo

# 5. add your task-specific python requirements here
COPY ./requirements.txt /app/requirements.txt

# 6. install python dependencies
# add cloned third-party repos or other requirements files if needed
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    python -m pip install agentrl-worker -r ./requirements.txt

# 7. (optional) additional setup after installing dependencies, e.g., downloading datasets
# RUN some-download-script

# 8. copy your task code, data and configuration files (modify the paths as needed)
COPY ./data /app/data
COPY ./src /app/src
COPY ./configs /app/configs

# 9. add entrypoint script
COPY --chmod=0755 ./extra/docker/worker-entrypoint.sh /entrypoint.sh

# 10. set entrypoint to the script and point default config to your task config file
ENTRYPOINT ["/entrypoint.sh", "-c", "configs/your-config.yaml"]
