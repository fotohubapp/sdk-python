"""Shared fixtures: a client wired to an in-memory transport, so no test touches the network."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fotohub import AsyncFotoHub, FotoHub  # noqa: E402

Responder = Callable[[httpx.Request], httpx.Response]


class Recorder:
    """Records every request and answers from a queue of (status, json[, headers]) tuples."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.answers: list[tuple] = []
        self.default: tuple = (200, {})

    def queue(self, *answers: tuple) -> "Recorder":
        self.answers.extend(answers)
        return self

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.answers.pop(0) if self.answers else self.default
        status, payload = answer[0], answer[1]
        headers = answer[2] if len(answer) > 2 else None
        return httpx.Response(status, json=payload, headers=headers)

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]

    def body(self, index: int = -1) -> Any:
        raw = self.requests[index].content
        return json.loads(raw) if raw else None


@pytest.fixture()
def rec() -> Recorder:
    return Recorder()


@pytest.fixture()
def client(rec: Recorder) -> FotoHub:
    c = FotoHub(api_key="fh_test_key", base_url="https://api.test", max_retries=1)
    c._client = httpx.Client(base_url=c.base_url, headers=c._headers(), transport=httpx.MockTransport(rec.handler))
    return c


@pytest.fixture()
def aclient(rec: Recorder) -> AsyncFotoHub:
    c = AsyncFotoHub(api_key="fh_test_key", base_url="https://api.test", max_retries=1)
    c._client = httpx.AsyncClient(
        base_url=c.base_url, headers=c._headers(), transport=httpx.MockTransport(rec.handler)
    )
    return c
