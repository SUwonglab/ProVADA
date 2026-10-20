"""Real asynchronous CSV writes and failure propagation."""
import pandas as pd
import pytest

from provada.utils.cache import CSVCache, cache_to_csv, shutdown_cache


def test_csv_cache_appends_snapshots_and_drains_writes(tmp_path):
    path = tmp_path / "rows.csv"
    cache = CSVCache(path)
    frame = pd.DataFrame({"value": [1]})
    try:
        cache.cache_df(frame)
        frame.loc[0, "value"] = 2
        cache.cache_df(frame)
    finally:
        cache.shutdown()
    assert pd.read_csv(path).value.tolist() == [1, 2]


@pytest.mark.parametrize("kind", [OSError, KeyboardInterrupt])
def test_csv_cache_surfaces_worker_error_and_stops_later_enqueues(tmp_path, monkeypatch, kind):
    error = kind("simulated write failure")
    def failed_write(*args, **kwargs):
        raise error
    monkeypatch.setattr(pd.DataFrame, "to_csv", failed_write)
    path = tmp_path / "trajectory.csv"
    cache = CSVCache(path, max_pending_writes=1)
    cache.cache_df(pd.DataFrame({"value": [1]}))
    with pytest.raises(kind) as caught:
        cache.shutdown()
    assert caught.value is error
    assert any(str(path) in note for note in error.__notes__)
    with pytest.raises(kind) as later:
        cache.cache_df(pd.DataFrame({"value": [2]}))
    assert later.value is error
    assert cache.semaphore.acquire(blocking=False)
    cache.semaphore.release()


def test_csv_submission_failure_releases_the_pending_slot(tmp_path):
    cache = CSVCache(tmp_path / "rows.csv", max_pending_writes=1)
    cache.shutdown()
    with pytest.raises(RuntimeError, match="shutdown"):
        cache.cache_df(pd.DataFrame({"value": [1]}))
    assert cache.semaphore.acquire(blocking=False)
    cache.semaphore.release()


def test_waiting_csv_producer_wakes_and_receives_worker_failure(tmp_path, monkeypatch):
    from threading import Event, Thread
    error = OSError("write failed while another producer was waiting")
    release_writer, waiting_producer = Event(), Event()
    cache = CSVCache(tmp_path / "rows.csv", max_pending_writes=1)
    semaphore = cache.semaphore
    class ObservedSemaphore:
        calls = 0
        def acquire(self):
            self.calls += 1
            if self.calls == 2:
                waiting_producer.set()
            return semaphore.acquire()
        def release(self):
            return semaphore.release()
    cache.semaphore = ObservedSemaphore()
    def failed_write(*args, **kwargs):
        assert release_writer.wait(timeout=5)
        raise error
    monkeypatch.setattr(pd.DataFrame, "to_csv", failed_write)
    errors = []
    def produce():
        try:
            cache.cache_df(pd.DataFrame({"value": [2]}))
        except BaseException as caught:
            errors.append(caught)
    producer = Thread(target=produce, daemon=True)
    try:
        cache.cache_df(pd.DataFrame({"value": [1]}))
        producer.start()
        assert waiting_producer.wait(timeout=5)
    finally:
        release_writer.set()
        if producer.ident is not None:
            producer.join(timeout=5)
        with pytest.raises(OSError):
            cache.shutdown()
    assert not producer.is_alive()
    assert errors == [error]
    assert semaphore.acquire(blocking=False)
    semaphore.release()


def test_real_cache_error_does_not_skip_other_paths(tmp_path, monkeypatch):
    registry = {}
    monkeypatch.setitem(cache_to_csv.__globals__, "_path_caches", registry)
    original_to_csv = pd.DataFrame.to_csv
    error = OSError("trajectory disk failure")
    bad = tmp_path / "bad.csv"
    good = tmp_path / "good.csv"
    def write(self, path, *args, **kwargs):
        if path == bad:
            raise error
        return original_to_csv(self, path, *args, **kwargs)
    monkeypatch.setattr(pd.DataFrame, "to_csv", write)
    cache_to_csv(pd.DataFrame({"value": [1]}), bad)
    cache_to_csv(pd.DataFrame({"value": [2]}), good)
    with pytest.raises(OSError) as caught:
        shutdown_cache()
    assert caught.value is error
    assert registry == {}
    assert pd.read_csv(good).value.tolist() == [2]
