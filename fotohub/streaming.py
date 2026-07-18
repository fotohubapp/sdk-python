"""SSE streaming support for FOTOhub chat completions."""

from __future__ import annotations

import json
from typing import AsyncIterator, Iterator

import httpx

from .models import ChatChunk


def _parse_sse_line(line: str) -> ChatChunk | None:
    """Parse a single SSE data line into a ChatChunk.

    Returns None for non-data lines, empty lines, or [DONE] signals.
    """
    line = line.strip()

    if not line or not line.startswith("data:"):
        return None

    data = line[5:].strip()

    if data == "[DONE]":
        return None

    try:
        payload = json.loads(data)
        return ChatChunk(**payload)
    except (json.JSONDecodeError, Exception):
        return None


class ChatStream:
    """Synchronous iterator over SSE chat completion chunks.

    Usage:
        stream = client.chat(messages=[...], stream=True)
        for chunk in stream:
            if chunk.delta_content:
                print(chunk.delta_content, end="")
    """

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self._iterator = response.iter_lines()

    def __iter__(self) -> Iterator[ChatChunk]:
        return self

    def __next__(self) -> ChatChunk:
        while True:
            try:
                line = next(self._iterator)
            except StopIteration:
                self.close()
                raise

            chunk = _parse_sse_line(line)
            if chunk is not None:
                return chunk

    def close(self) -> None:
        """Close the underlying response."""
        self._response.close()

    def __enter__(self) -> "ChatStream":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def collect(self) -> str:
        """Consume the entire stream and return the full text content."""
        parts: list[str] = []
        for chunk in self:
            content = chunk.delta_content
            if content:
                parts.append(content)
        return "".join(parts)


class AsyncChatStream:
    """Asynchronous iterator over SSE chat completion chunks.

    Usage:
        stream = await client.chat(messages=[...], stream=True)
        async for chunk in stream:
            if chunk.delta_content:
                print(chunk.delta_content, end="")
    """

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self._iterator = response.aiter_lines()

    def __aiter__(self) -> AsyncIterator[ChatChunk]:
        return self

    async def __anext__(self) -> ChatChunk:
        while True:
            try:
                line = await self._iterator.__anext__()
            except StopAsyncIteration:
                await self.close()
                raise

            chunk = _parse_sse_line(line)
            if chunk is not None:
                return chunk

    async def close(self) -> None:
        """Close the underlying response."""
        await self._response.aclose()

    async def __aenter__(self) -> "AsyncChatStream":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def collect(self) -> str:
        """Consume the entire stream and return the full text content."""
        parts: list[str] = []
        async for chunk in self:
            content = chunk.delta_content
            if content:
                parts.append(content)
        return "".join(parts)
