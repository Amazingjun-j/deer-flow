from __future__ import annotations

import errno
import os
import threading
from types import SimpleNamespace

import pytest

from deerflow.sandbox.local.local_sandbox import LocalSandbox, PathMapping
from deerflow.sandbox.sandbox import Sandbox


class _LegacySandbox(Sandbox):
    """Third-party-style provider implementing only the historical contract."""

    def execute_command(self, command, env=None, timeout=None):
        return ""

    def read_file(self, path, start_line=None, end_line=None):
        return ""

    def download_file(self, path):
        return b"legacy"

    def list_dir(self, path, max_depth=2):
        return []

    def write_file(self, path, content, append=False):
        return None

    def glob(self, path, pattern, *, include_dirs=False, max_results=200):
        return [], False

    def grep(self, path, pattern, *, glob=None, literal=False, case_sensitive=False, max_results=100):
        return [], False

    def update_file(self, path, content):
        return None


def test_bounded_download_is_additive_for_legacy_subclasses() -> None:
    sandbox = _LegacySandbox("legacy")

    assert sandbox.download_file("/mnt/user-data/outputs/a.bin") == b"legacy"
    with pytest.raises(NotImplementedError, match="does not support bounded file downloads"):
        sandbox.download_file_bounded("/mnt/user-data/outputs/a.bin", max_bytes=1)


@pytest.mark.parametrize("value", [-1, -100])
def test_bounded_download_rejects_negative_limits(value: int) -> None:
    sandbox = _LegacySandbox("legacy")
    with pytest.raises(ValueError, match="non-negative"):
        sandbox.download_file_bounded("/mnt/user-data/outputs/a.bin", max_bytes=value)


@pytest.mark.parametrize("value", [True, False, 1.5, "1", None])
def test_bounded_download_rejects_non_integer_limits(value) -> None:
    sandbox = _LegacySandbox("legacy")
    with pytest.raises(TypeError, match="integer"):
        sandbox.download_file_bounded("/mnt/user-data/outputs/a.bin", max_bytes=value)


def test_local_bounded_download_allows_exact_limit_and_zero_empty(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "exact.bin").write_bytes(b"abcd")
    (outputs / "empty.bin").write_bytes(b"")
    sandbox = LocalSandbox(
        "local",
        [PathMapping("/mnt/user-data/outputs", str(outputs))],
    )

    assert sandbox.download_file_bounded("/mnt/user-data/outputs/exact.bin", max_bytes=4) == b"abcd"
    assert sandbox.download_file_bounded("/mnt/user-data/outputs/empty.bin", max_bytes=0) == b""


def test_local_bounded_download_rejects_one_byte_over_limit(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "large.bin").write_bytes(b"abcde")
    sandbox = LocalSandbox(
        "local",
        [PathMapping("/mnt/user-data/outputs", str(outputs))],
    )

    with pytest.raises(OSError) as exc_info:
        sandbox.download_file_bounded("/mnt/user-data/outputs/large.bin", max_bytes=4)
    assert exc_info.value.errno == errno.EFBIG


def test_local_bounded_download_caps_read_after_stale_size_probe(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    payload_path = outputs / "growing.bin"
    payload_path.write_bytes(b"0123456789abcdef")
    sandbox = LocalSandbox(
        "local",
        [PathMapping("/mnt/user-data/outputs", str(outputs))],
    )

    real_getsize = os.path.getsize
    read_sizes: list[int] = []
    real_open = open

    def stale_getsize(path):
        if os.fspath(path) == os.fspath(payload_path):
            return 4
        return real_getsize(path)

    class _TrackedReader:
        def __init__(self, file_obj):
            self._file_obj = file_obj

        def __enter__(self):
            self._file_obj.__enter__()
            return self

        def __exit__(self, *args):
            return self._file_obj.__exit__(*args)

        def read(self, size=-1):
            read_sizes.append(size)
            return self._file_obj.read(size)

    def tracked_open(path, mode="r", *args, **kwargs):
        file_obj = real_open(path, mode, *args, **kwargs)
        if os.fspath(path) == os.fspath(payload_path) and mode == "rb":
            return _TrackedReader(file_obj)
        return file_obj

    monkeypatch.setattr(os.path, "getsize", stale_getsize)
    monkeypatch.setattr("builtins.open", tracked_open)

    with pytest.raises(OSError) as exc_info:
        sandbox.download_file_bounded("/mnt/user-data/outputs/growing.bin", max_bytes=8)
    assert exc_info.value.errno == errno.EFBIG
    assert read_sizes == [9]


class _TrackedStream:
    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks
        self.consumed = 0
        self.closed = False

    def __iter__(self):
        for chunk in self._chunks:
            self.consumed += 1
            yield chunk

    def close(self):
        self.closed = True


def _new_aio_sandbox(stream: _TrackedStream):
    aio_mod = pytest.importorskip("deerflow.community.aio_sandbox.aio_sandbox")
    sandbox = object.__new__(aio_mod.AioSandbox)
    Sandbox.__init__(sandbox, "aio")
    sandbox._lock = threading.Lock()
    sandbox._client = SimpleNamespace(
        file=SimpleNamespace(download_file=lambda *, path: stream),
    )
    return sandbox


def test_aio_bounded_download_stops_after_first_overflow_chunk() -> None:
    stream = _TrackedStream([b"ab", b"cd", b"ef", b"gh"])
    sandbox = _new_aio_sandbox(stream)

    with pytest.raises(OSError) as exc_info:
        sandbox.download_file_bounded("/mnt/user-data/outputs/a.bin", max_bytes=4)
    assert exc_info.value.errno == errno.EFBIG
    assert stream.consumed == 3


def _new_e2b_sandbox(files):
    e2b_mod = pytest.importorskip("deerflow.community.e2b_sandbox.e2b_sandbox")
    sandbox = object.__new__(e2b_mod.E2BSandbox)
    Sandbox.__init__(sandbox, "e2b")
    sandbox._lock = threading.Lock()
    sandbox._client = SimpleNamespace(files=files)
    sandbox._home_dir = "/home/user"
    return sandbox


def test_e2b_bounded_download_closes_stream_on_overflow() -> None:
    stream = _TrackedStream([b"ab", b"cd", b"ef", b"gh"])
    files = SimpleNamespace(read=lambda path, format=None: stream)
    sandbox = _new_e2b_sandbox(files)

    with pytest.raises(OSError) as exc_info:
        sandbox.download_file_bounded("/mnt/user-data/outputs/a.bin", max_bytes=4)
    assert exc_info.value.errno == errno.EFBIG
    assert stream.consumed == 3
    assert stream.closed is True


def test_e2b_strict_bounded_download_does_not_use_buffered_sdk_fallback() -> None:
    formats: list[str | None] = []

    def read(path, format=None):
        formats.append(format)
        if format == "stream":
            raise TypeError("format is not supported")
        return b"small"

    sandbox = _new_e2b_sandbox(SimpleNamespace(read=read))

    with pytest.raises(OSError) as exc_info:
        sandbox.download_file_bounded("/mnt/user-data/outputs/a.bin", max_bytes=10)
    assert exc_info.value.errno == errno.ENOTSUP
    assert formats == ["stream"]

    assert sandbox.download_file("/mnt/user-data/outputs/a.bin") == b"small"
    assert formats == ["stream", "stream", "bytes"]


def test_opensandbox_bounded_download_closes_stream_on_overflow() -> None:
    opensandbox_mod = pytest.importorskip("deerflow.community.opensandbox.sandbox")
    stream = _TrackedStream([b"ab", b"cd", b"ef", b"gh"])
    files = SimpleNamespace(read_bytes_stream=lambda path: stream)
    sandbox = object.__new__(opensandbox_mod.OpenSandboxSandbox)
    Sandbox.__init__(sandbox, "opensandbox")
    sandbox._file_op = lambda operation: operation(files)

    with pytest.raises(OSError) as exc_info:
        sandbox.download_file_bounded("/mnt/user-data/outputs/a.bin", max_bytes=4)
    assert exc_info.value.errno == errno.EFBIG
    assert stream.consumed == 3
    assert stream.closed is True


def test_tenki_bounded_download_preflights_and_stops_stream() -> None:
    tenki_mod = pytest.importorskip("deerflow.community.tenki.sandbox")
    stream = _TrackedStream([b"ab", b"cd", b"ef", b"gh"])
    sandbox = object.__new__(tenki_mod.TenkiSandbox)
    Sandbox.__init__(sandbox, "tenki")
    sandbox._lock = threading.Lock()
    sandbox._closed = False
    sandbox._home_dir = "/home/tenki"
    sandbox._sandbox = SimpleNamespace(fs=SimpleNamespace(read_stream=lambda path: stream))
    sandbox._sh = lambda script: SimpleNamespace(exit_code=0, stdout_text="4\n", stderr_text="")
    sandbox._note_failure = lambda error: None

    with pytest.raises(OSError) as exc_info:
        sandbox.download_file_bounded("/mnt/user-data/outputs/a.bin", max_bytes=4)
    assert exc_info.value.errno == errno.EFBIG
    assert stream.consumed == 3


def test_tenki_bounded_download_rejects_preflight_oversize_without_opening_stream() -> None:
    tenki_mod = pytest.importorskip("deerflow.community.tenki.sandbox")
    opened = False

    def read_stream(path):
        nonlocal opened
        opened = True
        return iter([b"payload"])

    sandbox = object.__new__(tenki_mod.TenkiSandbox)
    Sandbox.__init__(sandbox, "tenki")
    sandbox._lock = threading.Lock()
    sandbox._closed = False
    sandbox._home_dir = "/home/tenki"
    sandbox._sandbox = SimpleNamespace(fs=SimpleNamespace(read_stream=read_stream))
    sandbox._sh = lambda script: SimpleNamespace(exit_code=0, stdout_text="100\n", stderr_text="")
    sandbox._note_failure = lambda error: None

    with pytest.raises(OSError) as exc_info:
        sandbox.download_file_bounded("/mnt/user-data/outputs/a.bin", max_bytes=4)
    assert exc_info.value.errno == errno.EFBIG
    assert opened is False


def test_boxlite_bounded_download_uses_limit_plus_one_payload_read() -> None:
    boxlite_mod = pytest.importorskip("deerflow.community.boxlite.box")
    commands: list[str] = []
    sandbox = object.__new__(boxlite_mod.BoxliteBox)
    Sandbox.__init__(sandbox, "boxlite")

    def sh(script):
        commands.append(script)
        if script.startswith("wc -c"):
            return SimpleNamespace(exit_code=0, stdout="4\n", stderr="")
        if script.startswith("head -c 5"):
            return SimpleNamespace(exit_code=0, stdout="YWJjZA==\n", stderr="")
        raise AssertionError(script)

    sandbox._sh = sh

    assert sandbox.download_file_bounded("/mnt/user-data/outputs/a.bin", max_bytes=4) == b"abcd"
    assert any(command.startswith("head -c 5") for command in commands)


def test_boxlite_bounded_download_rejects_preflight_oversize_without_payload_read() -> None:
    boxlite_mod = pytest.importorskip("deerflow.community.boxlite.box")
    commands: list[str] = []
    sandbox = object.__new__(boxlite_mod.BoxliteBox)
    Sandbox.__init__(sandbox, "boxlite")

    def sh(script):
        commands.append(script)
        return SimpleNamespace(exit_code=0, stdout="100\n", stderr="")

    sandbox._sh = sh

    with pytest.raises(OSError) as exc_info:
        sandbox.download_file_bounded("/mnt/user-data/outputs/a.bin", max_bytes=4)
    assert exc_info.value.errno == errno.EFBIG
    assert len(commands) == 1
