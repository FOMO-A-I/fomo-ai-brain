"""Public-web GET fetcher with redirect revalidation and SSRF protections."""

from __future__ import annotations

from collections.abc import Callable
import time
from urllib.parse import urljoin

from ._http import request_public


class SafeWebFetcher:
    def __init__(
        self,
        *,
        timeout_seconds: float = 8,
        max_bytes: int = 1_000_000,
        max_redirects: int = 3,
        transport: Callable[..., tuple[int, dict[str, str], bytes, str]] | None = None,
    ) -> None:
        if not 0 <= max_redirects <= 5:
            raise ValueError("max_redirects must be between 0 and 5")
        if not 0 < timeout_seconds <= 30 or not 1 <= max_bytes <= 5_000_000:
            raise ValueError("web timeout or response size is out of range")
        self.timeout_seconds = timeout_seconds
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects
        self._transport = transport

    def fetch(
        self,
        url: str,
        *,
        allowed_hosts: set[str] | None = None,
        validate_url: Callable[[str], None] | None = None,
    ) -> dict[str, str | int]:
        current = url
        deadline = time.monotonic() + self.timeout_seconds
        for redirect_count in range(self.max_redirects + 1):
            if validate_url is not None:
                validate_url(current)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError("web fetch exceeded its time limit")
            status, headers, body, host = (self._transport or request_public)(
                current,
                timeout=remaining,
                max_bytes=self.max_bytes,
                headers={"Accept": "text/html, text/plain, application/json"},
                allowed_hosts=allowed_hosts,
            )
            if time.monotonic() > deadline:
                raise ValueError("web fetch exceeded its time limit")
            if status in {301, 302, 303, 307, 308}:
                location = headers.get("location")
                if not location:
                    raise ValueError("redirect response is missing Location")
                if redirect_count == self.max_redirects:
                    raise ValueError("maximum redirect count exceeded")
                current = urljoin(current, location)
                continue
            if status < 200 or status >= 300:
                raise ValueError(f"web server returned HTTP {status}")
            content_type = headers.get("content-type", "").split(";", 1)[0].lower()
            if content_type and not (
                content_type.startswith("text/")
                or content_type in {"application/json", "application/xml"}
            ):
                raise ValueError(f"unsupported web response type: {content_type}")
            charset = "utf-8"
            if "charset=" in headers.get("content-type", "").lower():
                charset = headers["content-type"].lower().split("charset=", 1)[1].split(";", 1)[0].strip()
            try:
                text = body.decode(charset, errors="replace")
            except LookupError:
                text = body.decode("utf-8", errors="replace")
            return {"url": current, "host": host, "status": status, "text": text}
        raise ValueError("redirect processing failed")