"""Sampler lifetime failures with controlled worker and logging boundaries."""
from contextlib import contextmanager
from contextvars import ContextVar
from types import ModuleType, SimpleNamespace
import sys

import pytest

from provada.sampler.sampler import MADA
from provada.utils.cache import cache_to_csv, shutdown_cache

import pandas as pd


def cleanup_sampler_type(events, fail_at=None, failure=None, close_failure=None):
    class CleanupSampler(MADA):
        def _compute_context(self):
            @contextmanager
            def context():
                events.append("enter")
                try:
                    yield
                finally:
                    events.append("close")
                    if close_failure is not None:
                        raise close_failure
            return context()
        def step(self, name):
            events.append(name)
            if name == fail_at:
                raise failure
        def initialize_generator(self, *args):
            self.step("generator")
            self.generator = SimpleNamespace()
        def initialize_evaluators(self):
            self.step("evaluator")
        def initialize_masking_schedule(self, *args):
            self.step("masking")
        def initialize_temperature_schedule(self, *args):
            self.step("temperature")
        def _run(self):
            self.step("sampling")
    return CleanupSampler


def construct_cleanup_sampler(cls, tmp_path, events, config=None, wandb=None):
    base = SimpleNamespace(score_dict={"dummy": 0.5}, num_designable_positions=1,
                           save_yaml=lambda path: events.append("reference_save"))
    obj = cls.__new__(cls)
    obj.__init__(base, num_iters=1, population_size=1, top_k_fraction=1.,
                 score_weights={"dummy": 1.}, output_dir=tmp_path,
                 run_name="cleanup", wandb_run=wandb, config=config)
    return obj


def cleanup_boundaries(monkeypatch, events, cache_failure=None, log_failure=None):
    def flush():
        events.append("flush")
        if cache_failure is not None:
            raise cache_failure
    def copy_log(**kwargs):
        events.append("log_copy")
        if log_failure is not None:
            raise log_failure
    namespace = MADA.end_run.__globals__
    monkeypatch.setitem(namespace, "shutdown_cache", flush)
    monkeypatch.setitem(namespace, "logger", SimpleNamespace(
        info=lambda *a, **k: None, error=lambda *a, **k: None, copy_log_file=copy_log))


@pytest.mark.parametrize("stage", ["generator", "evaluator", "masking", "temperature"])
@pytest.mark.parametrize("kind", [ValueError, KeyboardInterrupt])
def test_startup_failure_closes_owned_context_and_preserves_error(tmp_path, monkeypatch, stage, kind):
    events = []
    cleanup_boundaries(monkeypatch, events)
    failure = kind("startup failed")
    cls = cleanup_sampler_type(events, fail_at=stage, failure=failure)
    with pytest.raises(kind) as caught:
        construct_cleanup_sampler(cls, tmp_path, events)
    assert caught.value is failure
    assert events.count("enter") == events.count("close") == 1
    assert "flush" not in events  # Startup never created sampler CSV writers.


def test_config_export_failure_closes_context_and_finishes_wandb(tmp_path, monkeypatch):
    events = []
    cleanup_boundaries(monkeypatch, events)
    error = OSError("config export failed")
    def save(path):
        raise error
    wandb = SimpleNamespace(summary={}, finish=lambda exit_code: events.append(("wandb", exit_code)))
    with pytest.raises(OSError) as caught:
        construct_cleanup_sampler(cleanup_sampler_type(events), tmp_path, events,
                                  config=SimpleNamespace(save_yaml=save), wandb=wandb)
    assert caught.value is error
    assert events.count("close") == 1
    assert ("wandb", 1) in events


def test_startup_preserves_error_when_context_exit_also_fails(tmp_path, monkeypatch):
    events = []
    cleanup_boundaries(monkeypatch, events)
    original, secondary = ValueError("bad evaluator"), OSError("worker shutdown failed")
    wandb = SimpleNamespace(summary={}, finish=lambda exit_code: events.append(("wandb", exit_code)))
    cls = cleanup_sampler_type(events, fail_at="evaluator", failure=original, close_failure=secondary)
    with pytest.raises(ValueError) as caught:
        construct_cleanup_sampler(cls, tmp_path, events, wandb=wandb)
    assert caught.value is original
    assert any("worker shutdown failed" in note for note in original.__notes__)
    assert ("wandb", 1) in events


def test_successful_run_closes_once_and_cleanup_is_idempotent(tmp_path, monkeypatch):
    events = []
    cleanup_boundaries(monkeypatch, events)
    wandb = SimpleNamespace(summary={}, finish=lambda exit_code: events.append(("wandb", exit_code)))
    sampler = construct_cleanup_sampler(cleanup_sampler_type(events), tmp_path, events, wandb=wandb)
    sampler.run()
    sampler.end_run()
    assert events.count("flush") == events.count("close") == events.count("log_copy") == events.count("reference_save") == 1
    assert events.count(("wandb", 0)) == 1
    assert events.index("flush") < events.index("close") < events.index("log_copy")
    assert events.index("log_copy") < events.index("reference_save") < events.index(("wandb", 0))
    with pytest.raises(RuntimeError, match="already been closed"):
        sampler.run()


@pytest.mark.parametrize("kind", [ValueError, KeyboardInterrupt, SystemExit])
def test_run_failure_releases_resources_and_preserves_original(tmp_path, monkeypatch, kind):
    events = []
    cleanup_boundaries(monkeypatch, events)
    original = kind("sampling failed")
    sampler = construct_cleanup_sampler(cleanup_sampler_type(events, fail_at="sampling", failure=original),
                                        tmp_path, events)
    with pytest.raises(kind) as caught:
        sampler.run()
    assert caught.value is original
    assert events.count("close") == events.count("flush") == 1


@pytest.mark.parametrize("kind", [OSError, KeyboardInterrupt])
def test_cache_failure_does_not_skip_worker_or_wandb_shutdown(tmp_path, monkeypatch, kind):
    events = []
    original = kind("cache shutdown failed")
    cleanup_boundaries(monkeypatch, events, cache_failure=original)
    wandb = SimpleNamespace(summary={}, finish=lambda exit_code: events.append(("wandb", exit_code)))
    sampler = construct_cleanup_sampler(cleanup_sampler_type(events), tmp_path, events, wandb=wandb)
    with pytest.raises(kind) as caught:
        sampler.run()
    assert caught.value is original
    assert events.count("close") == 1
    assert ("wandb", 1) in events
    assert "log_copy" in events and "reference_save" in events


def test_sampling_error_wins_over_multiple_cleanup_failures(tmp_path, monkeypatch):
    events = []
    original = ValueError("sampling failed")
    cleanup_boundaries(monkeypatch, events, cache_failure=OSError("cache failed"))
    sampler = construct_cleanup_sampler(cleanup_sampler_type(events, fail_at="sampling", failure=original,
                                                               close_failure=OSError("worker failed")),
                                        tmp_path, events)
    with pytest.raises(ValueError) as caught:
        sampler.run()
    assert caught.value is original
    assert any("cache failed" in note for note in original.__notes__)
    assert any("worker failed" in note for note in original.__notes__)
    assert "reference_save" in events


def test_log_copy_failure_still_exports_reference_and_releases_resources(tmp_path, monkeypatch):
    events = []
    cleanup_boundaries(monkeypatch, events, log_failure=OSError("log copy failed"))
    sampler = construct_cleanup_sampler(cleanup_sampler_type(events), tmp_path, events)
    with pytest.raises(OSError, match="log copy failed"):
        sampler.run()
    assert events.count("close") == events.count("reference_save") == 1


@pytest.mark.parametrize("failed_export", ["log", "reference"])
@pytest.mark.parametrize("kind", [OSError, KeyboardInterrupt])
def test_final_export_failure_marks_wandb_failed_after_both_exports(tmp_path, monkeypatch, failed_export, kind):
    events = []
    error = kind("final export failed")
    cleanup_boundaries(monkeypatch, events, log_failure=error if failed_export == "log" else None)
    wandb = SimpleNamespace(summary={}, finish=lambda exit_code: events.append(("wandb", exit_code)))
    sampler = construct_cleanup_sampler(cleanup_sampler_type(events), tmp_path, events, wandb=wandb)
    if failed_export == "reference":
        def save(path):
            events.append("reference_save")
            raise error
        sampler.base_variant.save_yaml = save
    with pytest.raises(kind) as caught:
        sampler.run()
    assert caught.value is error
    assert events.count("close") == 1
    assert events.index("log_copy") < events.index(("wandb", 1))
    assert events.index("reference_save") < events.index(("wandb", 1))
    assert ("wandb", 0) not in events
    sampler.end_run()
    assert events.count(("wandb", 1)) == 1


def test_export_errors_remain_primary_when_wandb_finalization_also_fails(tmp_path, monkeypatch):
    events = []
    log_error, reference_error = OSError("log copy failed"), ValueError("reference export failed")
    tracking_error = RuntimeError("W&B finish failed")
    cleanup_boundaries(monkeypatch, events, log_failure=log_error)
    def finish(exit_code):
        events.append(("wandb", exit_code))
        raise tracking_error
    wandb = SimpleNamespace(summary={}, finish=finish)
    sampler = construct_cleanup_sampler(cleanup_sampler_type(events), tmp_path, events, wandb=wandb)
    def save(path):
        events.append("reference_save")
        raise reference_error
    sampler.base_variant.save_yaml = save
    with pytest.raises(OSError) as caught:
        sampler.run()
    assert caught.value is log_error
    assert any("reference export failed" in note for note in log_error.__notes__)
    assert any("W&B finish failed" in note for note in log_error.__notes__)
    assert events.index("reference_save") < events.index(("wandb", 1))
    assert events.count("close") == events.count(("wandb", 1)) == 1


@pytest.mark.parametrize("fail_startup", [False, True])
def test_external_worker_pool_remains_owned_by_the_caller(tmp_path, monkeypatch, fail_startup):
    events = []
    cleanup_boundaries(monkeypatch, events)
    active = ContextVar("active_pool", default=None)
    caller_pool = object()
    token = active.set(caller_pool)
    utils = ModuleType("proto_tools.utils")
    def forbidden_pool():
        pytest.fail("Sampler tried to acquire a pool while caller owns one")
    utils.ToolPool = forbidden_pool
    pool_module = ModuleType("proto_tools.utils.tool_pool")
    pool_module._active_pool = active
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    monkeypatch.setitem(sys.modules, pool_module.__name__, pool_module)
    cls = cleanup_sampler_type(events, fail_at="evaluator" if fail_startup else None,
                               failure=ValueError("invalid evaluator"))
    cls._compute_context = MADA._compute_context
    try:
        if fail_startup:
            with pytest.raises(ValueError, match="invalid evaluator"):
                construct_cleanup_sampler(cls, tmp_path, events)
        else:
            sampler = construct_cleanup_sampler(cls, tmp_path, events)
            sampler.run()
        assert active.get() is caller_pool
        assert "enter" not in events and "close" not in events
    finally:
        active.reset(token)


def test_cache_shutdown_attempts_all_writers_after_failure(monkeypatch):
    events = []
    original = OSError("first cache failed")
    def fail():
        events.append("first")
        raise original
    registry = {"first": SimpleNamespace(shutdown=fail),
                "second": SimpleNamespace(shutdown=lambda: events.append("second"))}
    monkeypatch.setitem(shutdown_cache.__globals__, "_path_caches", registry)
    with pytest.raises(OSError) as caught:
        shutdown_cache()
    assert caught.value is original
    assert events == ["first", "second"]
    assert registry == {}


@pytest.mark.parametrize("sampling_fails", [False, True])
def test_real_write_failure_still_releases_workers_and_finishes_tracking(tmp_path, monkeypatch, sampling_fails):
    events = []
    real_shutdown = shutdown_cache
    cleanup_boundaries(monkeypatch, events)
    monkeypatch.setitem(MADA.end_run.__globals__, "shutdown_cache", real_shutdown)
    registry = {}
    monkeypatch.setitem(cache_to_csv.__globals__, "_path_caches", registry)
    write_error = OSError("background trajectory write failed")
    def failed_write(*args, **kwargs):
        raise write_error
    monkeypatch.setattr(pd.DataFrame, "to_csv", failed_write)
    wandb = SimpleNamespace(summary={}, finish=lambda exit_code: events.append(("wandb", exit_code)))
    sampler = construct_cleanup_sampler(cleanup_sampler_type(events), tmp_path, events, wandb=wandb)
    sampling_error = ValueError("sampling failed first")
    def sample():
        cache_to_csv(pd.DataFrame({"value": [1]}), tmp_path / "trajectory.csv")
        if sampling_fails:
            raise sampling_error
    sampler._run = sample
    expected = sampling_error if sampling_fails else write_error
    with pytest.raises(type(expected)) as caught:
        sampler.run()
    assert caught.value is expected
    assert events.count("close") == 1
    assert ("wandb", 1) in events
    assert "log_copy" in events and "reference_save" in events
    assert registry == {}
