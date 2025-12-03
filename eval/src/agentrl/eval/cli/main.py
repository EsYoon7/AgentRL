from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from ..client import create_client
from ..event.bus import EventBus
from ..event.types import InitializedEvent
from ..session.controller import ControllerClient
from ..session.metric import Metric
from ..session.runner import EvaluationRunner
from ..session.tokens import TokenCounter
from ..utils import setup_logging

if TYPE_CHECKING:
    from .settings import Settings

LOG = logging.getLogger(__name__)


async def main(settings: Settings):
    ### setup logging
    setup_logging(settings.verbose)

    ### initialize event bus
    event_bus = EventBus()

    ### initialize tui
    if settings.interactive:
        try:
            from .tui import EvalApp
        except ImportError:
            logging.error('please reinstall with the "tui" extra to use interactive mode\n'
                          'pip install agentrl-eval[tui]')
            return

        _init_task = asyncio.create_task(event_bus.wait_for(InitializedEvent))

        app = EvalApp(event_bus=event_bus)
        _app_task = asyncio.create_task(app.run_async())

        async def _monitor(main_task: asyncio.Task):
            try:
                await _app_task
            finally:
                if not main_task.done():
                    main_task.cancel()

        asyncio.create_task(_monitor(asyncio.current_task()))

        await _init_task
    else:
        _app_task = None

    ### check: whether to enter view-only mode
    if not settings.controller or not settings.tasks:
        if _app_task is None:
            LOG.error('either controller URL or tasks must be provided (or enable interactive for view-only mode)')
            return
        view_only = True
    else:
        view_only = False

    ### build controller client
    if not view_only:
        controller = ControllerClient(
            base_url=str(settings.controller),
            proxy_url=str(settings.controller_proxy) if settings.controller_proxy is not None else None,
            insecure=settings.insecure
        )
    else:
        controller = None

    ### build model clients
    token_counter = TokenCounter(event_bus=event_bus)

    model_options = []
    if settings.models:
        for model in settings.models:
            settings_model = settings.model_dump()
            settings_model.update(model)
            model_options.append(settings_model)
    elif isinstance(settings.model, set) and len(settings.model) > 0:
        for model_name in settings.model:
            settings_model = settings.model_dump()
            settings_model['model'] = model_name
            model_options.append(settings_model)
    elif not view_only and not settings.start_sample:
        model_options.append(settings.model_dump())
    models = await asyncio.gather(*(
        create_client(settings.client, options, token_counter)
        for options in model_options
    ))

    ### run evaluation
    if settings.indices_range is not None:
        a, b = settings.indices_range.split('-', 1)
        settings.indices = set(range(int(a.strip()), int(b.strip()) + 1))

    if settings.resume is not None:
        if not settings.resume.is_dir():
            settings.resume = settings.output / settings.resume
        if not settings.resume.is_dir():
            LOG.error(f'resume directory {settings.resume} does not exist or is not a directory')
            return

    try:
        runner = EvaluationRunner(
            concurrency=settings.concurrency,
            controller=controller,
            controller_renew=settings.controller_renew,
            cross_sample=settings.cross_sample,
            custom_params=settings.custom_params,
            event_bus=event_bus,
            indices=settings.indices,
            interactive=settings.interactive,
            metric=Metric(settings.metric),
            models=models,
            output_dir=settings.output,
            resume=settings.resume,
            runs=settings.runs,
            start_sample=settings.start_sample,
            tasks=settings.tasks,
            token_counter=token_counter,
            view_only=view_only
        )

        await runner.run()
    except Exception:
        LOG.exception('unhandled error during evaluation:')
    finally:
        if controller is not None:
            await controller.close()
        for model_client in models:
            await model_client.close()

    # wait for tui to exit
    if _app_task is not None:
        await _app_task
