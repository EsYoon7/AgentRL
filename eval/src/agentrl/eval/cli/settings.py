from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import Field, SecretStr, AliasChoices
from pydantic_core import Url
from pydantic_settings import BaseSettings, CliImplicitFlag, CliPositionalArg, SettingsConfigDict

from .main import main
from ..session.types import MetricType, TaskIndex


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        cli_avoid_json=True,
        cli_hide_none_type=True,
        cli_ignore_unknown_args=False,
        cli_kebab_case=True,
        cli_parse_args=True,
        cli_shortcuts={
            'api-key': 'k',
            'base-url': 'u',
            'concurrency': ['j', 'jobs'],
            'controller': 'c',
            'indices': ['i', 'index'],
            'model': 'm',
            'output': 'o',
            'resume': 'r',
            'runs': 'n',
            'tasks': 'task',
            'temperature': 't',
            'verbose': ['v', 'debug']
        },
        env_file='.env',
        env_nested_delimiter='_',
        env_prefix='AGENTRL_EVAL_',
        nested_model_default_partial_update=True
    )

    # model settings
    client: Literal['anthropic', 'bedrock', 'dummy', 'openai'] = Field(
        default='openai',
        description='type of model client to use'
    )
    model: Optional[set[str]] = Field(
        default=None,
        description='name of the model to use. '
                    'specify more than one to use multiple models from the same provider. '
                    'for multiple models with different configurations, '
                    'use `--models` and pass a JSON list instead. '
                    '(omit for the first model available in the API)'
    )
    api_key: Optional[SecretStr] = Field(
        default=None,
        description='API key for the model service',
        validation_alias=AliasChoices(
            'api_key',
            'anthropic_api_key',
            'openai_api_key'
        )
    )
    base_url: Optional[Url] = Field(
        default=None,
        description='base URL for the model API',
        validation_alias=AliasChoices(
            'base_url',
            'anthropic_base_url',
            'anthropic_bedrock_base_url',
            'openai_base_url'
        )
    )
    proxy_url: Optional[Url] = Field(
        default=None,
        description='URL of the proxy to use to access the model API'
    )
    override_url: Optional[Url] = Field(
        default=None,
        description='override URL for the model messages API endpoint (for Anthropic clients only)'
    )
    thinking: CliImplicitFlag[bool] = Field(
        default=True,
        description='enable thinking mode for the model if supported (conflicts with temperature)'
    )
    computer_use: CliImplicitFlag[bool] = Field(
        default=False,
        description='enable computer use mode for the model (for Anthropic clients only)'
    )
    parallel_tool_calls: CliImplicitFlag[bool] = Field(
        default=True,
        description='permit parallel tool calls for the model'
    )
    temperature: Optional[float] = Field(
        default=0.8,
        description='temperature of the model to use (automatically disabled for thinking models)',
        ge=0.0,
        le=1.0
    )
    max_thinking_tokens: int = Field(
        default=10000,
        description='max thinking tokens for each round of interaction '
                    '(for Anthropic clients with thinking enabled only)',
        ge=1024
    )
    max_output_tokens: Optional[int] = Field(
        default=16000,
        description='max output tokens for each round of interaction',
        ge=1024
    )
    max_retries: Optional[int] = Field(
        default=2,
        description='max times of retries allowed for each API request (0 to disable retries)',
        ge=0
    )
    max_images: Optional[int] = Field(
        default=None,
        description='max number of images to include in each model query, omit for unlimited',
        ge=0
    )
    image_size: Optional[str] = Field(
        default=None,
        description='if specified, resize images to the specified size before including them in model queries. '
                    'supports resize based on minimum dimension, maximum dimension, or exact size. '
                    'examples: 512+ (min dimension), 1024- (max dimension), 800x600 (exact size)'
    )
    chat_completions: CliImplicitFlag[bool] = Field(
        default=False,
        description='use chat completions api only. '
                    'if not specified, the system will try to use responses api first, then fallback. '
                    '(for OpenAI clients only)'
    )
    extra_body: Optional[dict[str, Any]] = Field(
        default=None,
        description='extra body parameters to include in each model API request'
    )
    extra_headers: Optional[dict[str, str]] = Field(
        default=None,
        description='extra headers to include in each model API request'
    )
    aws_access_key: Optional[SecretStr] = Field(
        default=None,
        description='AWS access key (for Bedrock client only)'
    )
    aws_secret_key: Optional[SecretStr] = Field(
        default=None,
        description='AWS secret key (for Bedrock client only)'
    )
    aws_region: Optional[str] = Field(
        default=None,
        description='AWS region (for Bedrock client only)'
    )
    dummy_interval: Optional[float] = Field(
        default=1.0,
        description='interval between dummy model responses in seconds (for dummy client only)',
        ge=0.0
    )
    models: list[dict] = Field(
        default_factory=list,
        description='optionally provide a list of model configurations for cross-sampling evaluation'
    )

    # controller settings
    controller: Optional[Url] = Field(
        default=None,
        description='URL of the AgentRL controller API '
                    '(omit for view-only mode when interactive is enabled)'
    )
    controller_proxy: Optional[Url] = Field(
        default=None,
        description='URL of the proxy to use to access the controller API'
    )
    indices: set[TaskIndex] = Field(
        default_factory=set,
        description='a set of task indices to run (not recommended with multiple tasks)'
    )
    indices_range: Optional[str] = Field(
        default=None,
        description='range of task indices to run in the format "a-b"'
    )
    tasks: CliPositionalArg[set[str]] = Field(
        default_factory=set,
        description='one or more task names to run '
                    '(omit for view-only mode when interactive is enabled)'
    )
    custom_params: Optional[dict[str, Any]] = Field(
        default=None,
        description='custom parameters to use for all tasks'
    )
    controller_renew: CliImplicitFlag[bool] = Field(
        default=False,
        description='renew controller session periodically to avoid session expiration '
                    '(only valid when the task supports so, use with caution)'
    )

    # run settings
    runs: int = Field(
        default=1,
        description='number of times to run each task',
        ge=1
    )
    concurrency: int = Field(
        default=32,
        description='number of concurrent tasks to run',
        ge=1
    )
    start_sample: CliImplicitFlag[bool] = Field(
        default=False,
        description='call start_sample only, do not use models'
    )
    cross_sample: CliImplicitFlag[bool] = Field(
        default=False,
        description='enable cross-sampling evaluation when multiple models are provided'
    )

    # results settings
    output: Path = Field(
        default='results',
        description='directory to store evaluation results'
    )
    resume: Optional[Path] = Field(
        default=None,
        description='path to existing results to resume evaluation from'
    )
    metric: MetricType = Field(
        default=MetricType.SUCCESS_RATE,
        description='metric to use to evaluate model performance'
    )

    # generic settings
    insecure: CliImplicitFlag[bool] = Field(
        default=False,
        description='disable SSL verification for controller and model API requests'
    )
    verbose: CliImplicitFlag[bool] = Field(
        default=False,
        description='enable debug logging'
    )
    interactive: CliImplicitFlag[bool] = Field(
        default=sys.stdin.isatty() and sys.stdout.isatty(),
        description='enable interactive mode'
    )

    async def cli_cmd(self):
        await main(self)
