"""Public-web GET fetcher with redirect revalidation and SSRF protections."""

from __future__ import annotations

from urllib.parse import urljoin

from ._http import request_public


class SafeWebFetcher:
    def __init__(
        self,
        *,
        timeout_seconds: float = 8,
        max_bytes: int = 1_000_000,
        max_redirects: int = 3,
    ) -> None:
        if not 0 <= max_redirects <= 5:
            raise ValueError("max_redirects must be between 0 and 5")
        self.timeout_seconds = timeout_seconds
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects

    def fetch(self, url: str) -> dict[str, str | int]:
        current = url
        for redirect_count in range(self.max_redirects + 1):
            status, headers, body, host = request_public(
                current,
                timeout=self.timeout_seconds,
                max_bytes=self.max_bytes,
                headers={"Accept": "text/html, text/plain, application/json"},
            )
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