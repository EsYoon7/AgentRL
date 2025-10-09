"""Runtime monkey patches for :mod:`sglang`."""

from __future__ import annotations

import asyncio
import inspect
import logging
import signal
from typing import Any, Callable, Optional

_LOGGER = logging.getLogger(__name__)


def _method_source_contains(method: Callable[..., Any], needle: str) -> bool:
    try:
        source = inspect.getsource(method)
    except (OSError, TypeError):
        return False
    return needle in source


def _patch_engine_async_methods(engine_mod: Any) -> bool:
    patched = False
    engine_cls = getattr(engine_mod, "SGLangEngine", None)
    if engine_cls is None:
        _LOGGER.debug("SGLangEngine is missing; skip engine async patches")
        return False

    if not hasattr(engine_cls, "async_flush_cache"):
        async def async_flush_cache(self: Any) -> Any:
            print("sglang async flush cache")
            return await self.tokenizer_manager.flush_cache()

        engine_cls.async_flush_cache = async_flush_cache  # type: ignore[attr-defined]
        patched = True

    if not hasattr(engine_cls, "async_update_weights_from_tensor"):
        UpdateWeightsFromTensorReqInput = getattr(
            engine_mod, "UpdateWeightsFromTensorReqInput", None
        )
        MultiprocessingSerializer = getattr(
            engine_mod, "MultiprocessingSerializer", None
        )
        if UpdateWeightsFromTensorReqInput is None or MultiprocessingSerializer is None:
            _LOGGER.warning(
                "sglang async_update_weights_from_tensor patch skipped: missing helpers"
            )
        else:
            async def async_update_weights_from_tensor(
                self: Any,
                named_tensors: Any,
                load_format: Optional[str] = None,
                flush_cache: bool = True,
            ) -> Any:
                obj = UpdateWeightsFromTensorReqInput(
                    serialized_named_tensors=[
                        MultiprocessingSerializer.serialize(named_tensors)
                        for _ in range(self.server_args.tp_size)
                    ],
                    load_format=load_format,
                    flush_cache=flush_cache,
                )
                return await self.tokenizer_manager.update_weights_from_tensor(obj, None)

            engine_cls.async_update_weights_from_tensor = (  # type: ignore[attr-defined]
                async_update_weights_from_tensor
            )
            patched = True

    return patched


def _reset_engine_signal_handlers(engine_mod: Any) -> bool:
    signal_module = getattr(engine_mod, "signal", None)
    if signal_module is None:
        _LOGGER.debug("Engine module has no signal reference; skip signal patch")
        return False

    if getattr(signal_module, "_agentrl_signal_patch_applied", False):
        return False

    original_signal = getattr(signal_module, "signal", None)
    if original_signal is None:
        _LOGGER.debug("Engine signal module exposes no signal() function; skipping")
        return False

    def _noop_signal(signum: int, handler: Any) -> Any:
        # Upstream stops registering handlers because doing so from threads fails; emulate that.
        _LOGGER.debug("Skipping SGLang signal handler registration for %s", signum)
        try:
            return signal.getsignal(signum)
        except (OSError, RuntimeError, ValueError):
            return signal.SIG_DFL

    signal_module._agentrl_signal_patch_applied = True
    signal_module._agentrl_original_signal = original_signal  # type: ignore[attr-defined]
    signal_module.signal = _noop_signal  # type: ignore[assignment]
    return True


def _patch_tokenizer_manager(tokenizer_manager_mod: Any) -> bool:
    patched = False
    manager_cls = getattr(tokenizer_manager_mod, "TokenizerManager", None)
    if manager_cls is None:
        _LOGGER.debug("TokenizerManager is missing; skip tokenizer patches")
        return False

    # Ensure flush_cache auto-creates the communication loop before dispatch.
    if not _method_source_contains(manager_cls.flush_cache, "auto_create_handle_loop"):
        original_flush_cache = manager_cls.flush_cache

        async def patched_flush_cache(self: Any, *args: Any, **kwargs: Any) -> Any:
            self.auto_create_handle_loop()
            return await original_flush_cache(self, *args, **kwargs)

        manager_cls.flush_cache = patched_flush_cache  # type: ignore[assignment]
        patched = True

    # Ensure cancellations abort the outstanding request.
    if not _method_source_contains(manager_cls._wait_one_response, "asyncio.CancelledError"):
        dataclass_to_string_truncated = getattr(
            tokenizer_manager_mod, "dataclass_to_string_truncated", None
        )
        HTTPStatus = getattr(tokenizer_manager_mod, "HTTPStatus", None)
        logger = getattr(tokenizer_manager_mod, "logger", None)

        if dataclass_to_string_truncated is None or HTTPStatus is None or logger is None:
            _LOGGER.warning(
                "TokenizerManager cancellation patch skipped: missing helper symbols"
            )
        else:
            async def patched_wait_one_response(self: Any, obj: Any, state: Any, request: Any = None):
                while True:
                    try:
                        await asyncio.wait_for(state.event.wait(), timeout=4)
                    except asyncio.TimeoutError:
                        if request is not None and await request.is_disconnected():
                            self.abort_request(obj.rid)
                            raise ValueError(
                                f"Request is disconnected from the client side (type 1). Abort request {obj.rid=}"
                            )
                        continue
                    except asyncio.CancelledError:
                        self.abort_request(obj.rid)
                        print(f"aborting request {obj.rid=}")
                        raise

                    out = state.out_list[-1]

                    state.out_list = []
                    if state.finished:
                        if self.log_requests:
                            max_length, skip_names, out_skip_names = self.log_request_metadata
                            if self.model_config.is_multimodal_gen:
                                msg = (
                                    "Finish: obj="
                                    f"{dataclass_to_string_truncated(obj, max_length, skip_names=skip_names)}"
                                )
                            else:
                                msg = (
                                    "Finish: obj="
                                    f"{dataclass_to_string_truncated(obj, max_length, skip_names=skip_names)}"
                                    ", out="
                                    f"{dataclass_to_string_truncated(out, max_length, skip_names=out_skip_names)}"
                                )
                            logger.info(msg)

                        if isinstance(out["meta_info"].get("finish_reason"), dict):
                            finish_reason = out["meta_info"]["finish_reason"]
                            if (
                                finish_reason.get("type") == "abort"
                                and finish_reason.get("status_code") == HTTPStatus.BAD_REQUEST
                            ):
                                raise ValueError(finish_reason["message"])

                        yield out
                        break

                    state.event.clear()

                    if obj.stream:
                        yield out
                    else:
                        continue

            manager_cls._wait_one_response = patched_wait_one_response  # type: ignore[assignment]
            patched = True

    return patched


def apply_patch() -> bool:
    try:
        import sglang.srt.entrypoints.engine as engine_mod
        import sglang.srt.managers.tokenizer_manager as tokenizer_manager_mod
    except Exception as exc:  # pragma: no cover - defensive logging only
        _LOGGER.debug("sglang patch skipped; import failed: %s", exc)
        return False

    applied = False
    applied |= _patch_engine_async_methods(engine_mod)
    applied |= _reset_engine_signal_handlers(engine_mod)
    applied |= _patch_tokenizer_manager(tokenizer_manager_mod)

    return applied
