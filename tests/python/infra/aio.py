from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

# Public API expected by tests:
# - AioHTTPMock
# - ResponseContext
# - FakeAsyncIter
# - FakeWebSocket
# - patch_websockets_connect
# - _RespCtx (internal helper for tests)
# - _RouteSpec (internal helper for tests)


# -----------------------------
# Core response/context classes
# -----------------------------


@dataclass
class _RouteSpec:
    method: str
    url: str
    kind: str  # "text" or "json"
    payload: Any
    status: int = 200
    headers: Mapping[str, str] | None = None


class ResponseContext:
    """
    Lightweight aiohttp-like response used in 'async with ... as resp'.

    Implements subset used by tests:
      - attributes: status, headers
      - methods: text(), json(), read(), raise_for_status()
      - property: content (with async iter_chunked())
    """

    def __init__(self, spec: _RouteSpec | None = None, *, _bytes_iter: list[bytes] | None = None):
        if _bytes_iter is not None:
            # Alternative constructor for streaming bytes
            self._spec = None
            self.status: int = 200
            self.headers: Mapping[str, str] = {}
            self._bytes = b"".join(_bytes_iter)
            self._bytes_iter = _bytes_iter
        else:
            self._spec = spec
            self.status: int = int(spec.status)
            self.headers: Mapping[str, str] = dict(spec.headers or {})
            self._bytes_iter = None

            # Pre-encode bytes for streaming/read
            if spec.kind == "text":
                self._bytes = (spec.payload or "").encode("utf-8")
            else:
                self._bytes = json.dumps(spec.payload).encode("utf-8")

        class _Content:
            def __init__(self, data: bytes, bytes_iter: list[bytes] | None = None):
                self._data = data
                self._bytes_iter = bytes_iter

            async def iter_chunked(self, n: int):
                i = 0
                while i < len(self._data):
                    chunk = self._data[i : i + n]
                    i += n
                    yield chunk

            def __aiter__(self):
                return self._async_iter()

            async def _async_iter(self):
                # If we have original byte chunks, yield them as is
                if self._bytes_iter is not None:
                    for chunk in self._bytes_iter:
                        yield chunk
                else:
                    # Yield the data in small chunks to simulate streaming
                    chunk_size = 1024
                    i = 0
                    while i < len(self._data):
                        chunk = self._data[i : i + chunk_size]
                        yield chunk
                        i += chunk_size

        self.content = _Content(self._bytes, self._bytes_iter)

    # ----- async context manager -----
    async def __aenter__(self) -> ResponseContext:
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    # ----- body infra -----
    async def text(self) -> str:
        if self._spec is None:
            return self._bytes.decode("utf-8")
        if self._spec.kind == "text":
            return str(self._spec.payload or "")
        return json.dumps(self._spec.payload)

    async def json(self) -> Any:
        if self._spec is None:
            try:
                return json.loads(self._bytes.decode("utf-8"))
            except Exception as e:
                raise ValueError("Response is not JSON") from e
        if self._spec.kind == "json":
            return self._spec.payload
        try:
            return json.loads(self._spec.payload or "")
        except Exception as e:
            raise ValueError("Response is not JSON") from e

    async def read(self) -> bytes:
        return self._bytes

    def raise_for_status(self) -> None:
        if not (200 <= int(self.status) < 400):
            from aiohttp.client_exceptions import ClientResponseError

            raise ClientResponseError(
                request_info=None,
                history=(),
                status=int(self.status),
                message="mock error",
                headers=self.headers,
            )


class FakeAsyncIter:
    """Simple async iterator used by a few tests."""

    def __init__(self, items: Iterable[Any]):
        self._it = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


# ----------------------
# AioHTTPMock & session
# ----------------------


class _MockSession:
    """
    Tiny subset of aiohttp.ClientSession: get/post return a ResponseContext
    that supports 'async with'.
    """

    def __init__(self, routes: AioHTTPMock):
        self._routes = routes
        self._closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()
        return False

    def _response_for(self, method: str, url: str) -> ResponseContext:
        spec = self._routes._next_spec(method, url)
        return ResponseContext(spec)

    # Signature parity (we ignore params/timeout/etc; they’re allowed as kwargs)
    def get(self, url: str, *args, **kwargs) -> ResponseContext:
        return self._response_for("GET", url)

    def post(self, url: str, *args, **kwargs) -> ResponseContext:
        return self._response_for("POST", url)

    async def close(self):
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed


class AioHTTPMock:
    """
    Register per-(method,url) responses and hand out a mock session.
    Supports multiple queued responses for the same route (for retry/backoff tests).
    """

    def __init__(self):
        self._queue: dict[tuple[str, str], list[_RouteSpec]] = {}

    # Registration infra — match the tests' call shape exactly:
    # http.register_text("GET", "https://x/api", "boom", status=500)
    def register_text(
        self,
        method: str,
        url: str,
        text: str,
        *,
        status: int = 200,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        spec = _RouteSpec(
            method=method.upper(),
            url=url,
            kind="text",
            payload=text,
            status=status,
            headers=headers,
        )
        self._queue.setdefault((spec.method, spec.url), []).append(spec)

    def register_json(
        self,
        method: str,
        url: str,
        obj: Any,
        *,
        status: int = 200,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        spec = _RouteSpec(
            method=method.upper(), url=url, kind="json", payload=obj, status=status, headers=headers
        )
        self._queue.setdefault((spec.method, spec.url), []).append(spec)

    def _next_spec(self, method: str, url: str) -> _RouteSpec:
        key = (method.upper(), url)
        if key not in self._queue or not self._queue[key]:
            # default 404 text if unregistered or exhausted
            return _RouteSpec(
                method=method.upper(),
                url=url,
                kind="text",
                payload="not found",
                status=404,
                headers={"X-Mock": "1"},
            )
        return self._queue[key].pop(0)

    # Session factory used by tests
    def session(self) -> _MockSession:
        return _MockSession(self)

    # Optional context manager API
    def __enter__(self) -> AioHTTPMock:
        return self

    def __exit__(self, exc_type, exc, tb):
        self._queue.clear()
        return False

    # Monkeypatch helper for tests
    def patch(self, monkeypatch, target: str) -> None:
        """Patch the target with our mock session factory."""

        def mock_session_class(*args, **kwargs):
            return self.session()

        monkeypatch.setattr(target, mock_session_class)


# ----------------------
# WebSocket test infra
# ----------------------


class FakeWebSocket:
    def __init__(self, messages: Iterable[str] | None = None):
        self._queue = list(messages or [])
        self._sent: list[str] = []
        self.closed = False
        self.messages = messages or []
        self._iter = iter(self.messages)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration

    async def __aenter__(self) -> FakeWebSocket:
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()
        return False

    async def recv(self) -> str:
        await asyncio.sleep(0)  # yield control
        if not self._queue:
            self.closed = True
            # many websocket libs would signal closure; CancelledError is fine for tests
            raise asyncio.CancelledError("No more messages")
        return self._queue.pop(0)

    async def send_str(self, data: str) -> None:
        await asyncio.sleep(0)
        self._sent.append(str(data))

    async def close(self):
        self.closed = True


def patch_websockets_connect(messages: Iterable[str] | None = None):
    import types

    try:
        import websockets  # type: ignore
    except Exception:
        websockets = types.SimpleNamespace()

    original = getattr(websockets, "connect", None)

    async def _connect(*args, **kwargs):
        return FakeWebSocket(messages)

    class _Ctx:
        # sync enter/exit
        def __enter__(self):
            websockets.connect = _connect  # type: ignore[attr-defined]
            return self

        def __exit__(self, exc_type, exc, tb):
            if original is None:
                with contextlib.suppress(Exception):
                    delattr(websockets, "connect")  # type: ignore[attr-defined]
            else:
                websockets.connect = original  # type: ignore[attr-defined]
            return False

        # async enter/exit (so tests can use `async with` as well)
        async def __aenter__(self):
            self.__enter__()
            return FakeWebSocket(messages)

        async def __aexit__(self, exc_type, exc, tb):
            self.__exit__(exc_type, exc, tb)
            return False

    return _Ctx()


# Internal helper alias for tests
_RespCtx = ResponseContext
