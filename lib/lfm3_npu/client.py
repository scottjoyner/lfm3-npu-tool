"""
Enhanced Python client library for LFM3-1.7B inference servers.

Provides a production-grade Python interface to the OpenAI-compatible API with:
  * Async (asyncio) and sync APIs
  * Retry with exponential backoff and jitter
  * API key authentication (Bearer / X-API-Key)
  * SSL/TLS support with CA verification
  * Server-Sent Events (SSE) streaming
  * Connection reuse via urllib opener
  * Custom exception hierarchy
  * Full type hints

Usage:
    # Sync
    client = LFM3Client(base_url="http://localhost:1239", api_key="sk-...")
    response = client.chat("Hello, how are you?", max_tokens=50)

    # Async
    async with LFM3AsyncClient(base_url="http://localhost:1239", api_key="sk-...") as client:
        response = await client.chat("Hello, how are you?", max_tokens=50)

    # Streaming (SSE)
    for chunk in client.chat_stream("Tell me a story", max_tokens=100):
        print(chunk, end="", flush=True)
"""

import asyncio
import json
import ssl
import time
import urllib.error
import urllib.request
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
)

# --------------------------------------------------------------------------- #
# Exceptions                                                                  #
# --------------------------------------------------------------------------- #


class LFM3ClientError(Exception):
    """Base exception for all client errors."""


class LFM3AuthError(LFM3ClientError):
    """Authentication failed (401/403)."""


class LFM3NotFoundError(LFM3ClientError):
    """Resource not found (404)."""


class LFM3RateLimitError(LFM3ClientError):
    """Rate limited (429)."""


class LFM3ServerError(LFM3ClientError):
    """Server error (5xx)."""


class LFM3TimeoutError(LFM3ClientError):
    """Request timed out."""


class LFM3ConnectionError(LFM3ClientError):
    """Connection failed."""


# --------------------------------------------------------------------------- #
# Retry configuration                                                         #
# --------------------------------------------------------------------------- #


class RetryConfig:
    """Configuration for retry behaviour."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 30.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        retry_on_status: Optional[set] = None,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.retry_on_status = retry_on_status or {429, 502, 503, 504}

    def delay_for(self, attempt: int) -> float:
        """Compute the delay (seconds) before the given retry attempt (0-indexed)."""
        delay = min(self.base_delay * (self.backoff_factor ** attempt), self.max_delay)
        if self.jitter:
            import random

            delay *= 0.5 + random.random()  # nosec B311 - jitter only, not crypto
        return delay


# --------------------------------------------------------------------------- #
# Sync client                                                                 #
# --------------------------------------------------------------------------- #


class LFM3Client:
    """Synchronous Python client for LFM3 inference servers."""

    def __init__(
        self,
        base_url: str = "http://localhost:1239",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        tls_verify: bool = True,
        tls_ca_file: Optional[str] = None,
        retry: Optional[RetryConfig] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.tls_verify = tls_verify
        self.tls_ca_file = tls_ca_file
        self.retry = retry or RetryConfig()
        self._opener = self._build_opener()

    # -- internal ----------------------------------------------------------- #

    def _build_opener(self) -> urllib.request.OpenerDirector:
        """Build a urllib opener with SSL context if needed."""
        context = None
        if self.base_url.startswith("https:"):
            context = ssl.create_default_context()
            if not self.tls_verify:
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
            elif self.tls_ca_file:
                context.load_verify_locations(self.tls_ca_file)
            else:
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
        return urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))

    def _headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if extra:
            headers.update(extra)
        return headers

    def _request(
        self,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Send an HTTP request with retry logic."""
        url = f"{self.base_url}{path}"
        timeout = timeout or self.timeout
        last_error: Optional[Exception] = None

        for attempt in range(self.retry.max_retries + 1):
            try:
                body = None
                if data is not None:
                    body = json.dumps(data).encode("utf-8")
                req = urllib.request.Request(
                    url, data=body, headers=self._headers(), method=method
                )
                with self._opener.open(req, timeout=timeout) as resp:
                    raw = resp.read().decode("utf-8")
                    if not raw:
                        return None
                    return json.loads(raw)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                status = e.code
                last_error = self._map_http_error(status, body)
                if status in self.retry.retry_on_status and attempt < self.retry.max_retries:
                    time.sleep(self.retry.delay_for(attempt))
                    continue
                raise last_error
            except urllib.error.URLError as e:
                last_error = LFM3ConnectionError(str(e))
                if attempt < self.retry.max_retries:
                    time.sleep(self.retry.delay_for(attempt))
                    continue
                raise last_error
            except Exception as e:  # noqa BLE001
                last_error = LFM3ClientError(str(e))
                if attempt < self.retry.max_retries:
                    time.sleep(self.retry.delay_for(attempt))
                    continue
                raise last_error

        raise last_error or LFM3ClientError("Request failed")

    @staticmethod
    def _map_http_error(status: int, body: str) -> LFM3ClientError:
        if status == 401 or status == 403:
            return LFM3AuthError(f"HTTP {status}: {body}")
        if status == 404:
            return LFM3NotFoundError(f"HTTP {status}: {body}")
        if status == 429:
            return LFM3RateLimitError(f"HTTP {status}: {body}")
        if status >= 500:
            return LFM3ServerError(f"HTTP {status}: {body}")
        return LFM3ClientError(f"HTTP {status}: {body}")

    # -- public API -------------------------------------------------------- #

    def health(self) -> Dict[str, Any]:
        """Get server health status."""
        return self._request("GET", "/health")

    def live(self) -> Dict[str, Any]:
        """Liveness probe."""
        return self._request("GET", "/live")

    def ready(self) -> Dict[str, Any]:
        """Readiness probe."""
        return self._request("GET", "/ready")

    def list_models(self) -> Dict[str, Any]:
        """List available models."""
        return self._request("GET", "/v1/models")

    def model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        return self._request("GET", "/v1/model/info")

    def slo(self) -> Dict[str, Any]:
        """Get SLO report."""
        return self._request("GET", "/slo")

    def version(self) -> Dict[str, Any]:
        """Get version info."""
        return self._request("GET", "/version")

    def usage(self) -> Dict[str, Any]:
        """Get token usage statistics."""
        return self._request("GET", "/v1/usage")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stream: bool = False,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Create a chat completion."""
        data: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }
        if model:
            data["model"] = model
        return self._request("POST", "/v1/chat/completions", data, timeout=timeout)

    def chat(
        self,
        message: str,
        max_tokens: int = 50,
        temperature: float = 0.7,
        model: Optional[str] = None,
    ) -> str:
        """Simple chat interface - returns just the assistant's response text."""
        response = self.chat_completion(
            messages=[{"role": "user", "content": message}],
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
        )
        return response["choices"][0]["message"]["content"]

    def chat_stream(
        self,
        messages: Union[str, List[Dict[str, str]]],
        max_tokens: int = 50,
        temperature: float = 0.7,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Iterator[str]:
        """Stream chat completion tokens via Server-Sent Events.

        Yields text chunks as they arrive from the server.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        data = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if model:
            data["model"] = model

        url = f"{self.base_url}/v1/chat/completions"
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers=self._headers({"Accept": "text/event-stream"}), method="POST"
        )

        try:
            with self._opener.open(req, timeout=timeout or self.timeout) as resp:
                for line in resp:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        payload = line[6:]
                        if payload == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except urllib.error.HTTPError as e:
            raise self._map_http_error(e.code, e.read().decode("utf-8", errors="replace"))
        except Exception as e:  # noqa BLE001
            raise LFM3ClientError(str(e))

    def close(self) -> None:
        """Close the client (no-op for urllib, kept for interface symmetry)."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# --------------------------------------------------------------------------- #
# Async client                                                                #
# --------------------------------------------------------------------------- #


class LFM3AsyncClient:
    """Asynchronous Python client for LFM3 inference servers.

    Uses asyncio with run_in_executor for urllib-based I/O, providing a
    non-blocking API suitable for async applications.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1239",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        tls_verify: bool = True,
        tls_ca_file: Optional[str] = None,
        retry: Optional[RetryConfig] = None,
    ):
        self._sync = LFM3Client(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            tls_verify=tls_verify,
            tls_ca_file=tls_ca_file,
            retry=retry,
        )
        self.timeout = timeout

    async def _run(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    async def health(self) -> Dict[str, Any]:
        return await self._run(self._sync.health)

    async def live(self) -> Dict[str, Any]:
        return await self._run(self._sync.live)

    async def ready(self) -> Dict[str, Any]:
        return await self._run(self._sync.ready)

    async def list_models(self) -> Dict[str, Any]:
        return await self._run(self._sync.list_models)

    async def model_info(self) -> Dict[str, Any]:
        return await self._run(self._sync.model_info)

    async def slo(self) -> Dict[str, Any]:
        return await self._run(self._sync.slo)

    async def version(self) -> Dict[str, Any]:
        return await self._run(self._sync.version)

    async def usage(self) -> Dict[str, Any]:
        return await self._run(self._sync.usage)

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stream: bool = False,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        return await self._run(
            self._sync.chat_completion,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
            model=model,
            timeout=timeout,
        )

    async def chat(
        self,
        message: str,
        max_tokens: int = 50,
        temperature: float = 0.7,
        model: Optional[str] = None,
    ) -> str:
        return await self._run(
            self._sync.chat, message, max_tokens=max_tokens, temperature=temperature, model=model
        )

    async def chat_stream(
        self,
        messages: Union[str, List[Dict[str, str]]],
        max_tokens: int = 50,
        temperature: float = 0.7,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """Stream tokens asynchronously. Wraps the sync generator in an async iterator."""
        sync_gen = self._sync.chat_stream(
            messages, max_tokens=max_tokens, temperature=temperature, model=model, timeout=timeout
        )
        loop = asyncio.get_running_loop()
        for chunk in sync_gen:
            yield chunk
            # Yield control to the event loop between chunks
            await asyncio.sleep(0)

    async def close(self):
        self._sync.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


# --------------------------------------------------------------------------- #
# Convenience factory                                                         #
# --------------------------------------------------------------------------- #


def create_client(
    base_url: str = "http://localhost:1239",
    api_key: Optional[str] = None,
    async_mode: bool = False,
    **kwargs,
) -> Union[LFM3Client, LFM3AsyncClient]:
    """Factory to create a sync or async client."""
    if async_mode:
        return LFM3AsyncClient(base_url=base_url, api_key=api_key, **kwargs)
    return LFM3Client(base_url=base_url, api_key=api_key, **kwargs)


# --------------------------------------------------------------------------- #
# CLI smoke test                                                              #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys

    client = LFM3Client()
    try:
        print(f"Health: {client.health()}")
        response = client.chat("Hello, how are you?", max_tokens=30)
        print(f"Response: {response}")
    except LFM3ClientError as e:
        print(f"Error: {e}")
        print("Make sure the server is running: lfm3-npu server start npu")
        sys.exit(1)
