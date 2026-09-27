"""Explicitly approved public-URL retrieval for evidence-based research.

This is a fetch provider, not a search engine: it visits only caller-supplied URLs
and returns their URLs with the retrieved excerpts for citation.
"""

from __future__ import annotations

from collections.abc import Sequence
from html.parser import HTMLParser
from urllib.parse import urlsplit

from .web_tool import SafeWebFetcher


class _VisibleText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._in_title = False
        self._hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "title":
            self._in_title = True
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self._hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False
        if tag.lower() in {"script", "style", "noscript", "svg"} and self._hidden_depth:
            self._hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        if not self._hidden_depth:
            self.parts.append(data)


class SafeWebResearchProvider:
    """Fetch a small set of approved public URLs using the SSRF-safe web tool."""

    def __init__(
        self,
        fetcher: SafeWebFetcher | None = None,
        *,
        max_urls: int = 10,
        max_content_chars: int = 20_000,
    ) -> None:
        if not 1 <= max_urls <= 20:
            raise ValueError("max_urls must be between 1 and 20")
        if not 1 <= max_content_chars <= 100_000:
            raise ValueError("max_content_chars must be between 1 and 100,000")
        self.fetcher = fetcher or SafeWebFetcher(
            timeout_seconds=8, max_bytes=500_000, max_redirects=3
        )
        if (
            self.fetcher.timeout_seconds > 8
            or self.fetcher.max_bytes > 500_000
            or self.fetcher.max_redirects > 3
        ):
            raise ValueError("web research limits are capped at 8 seconds, 500 KB, and 3 redirects")
        self.max_urls = max_urls
        self.max_content_chars = max_content_chars

    @staticmethod
    def _host(value: str) -> str:
        parsed = urlsplit(value if "://" in value else f"https://{value}")
        if (
            not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or parsed.port not in (None, 443)
        ):
            raise ValueError("approved domains must be hostnames without credentials")
        return parsed.hostname.rstrip(".").encode("idna").decode("ascii").lower()

    @staticmethod
    def _url_host(url: str) -> str:
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.fragment
        ):
            raise ValueError("research URLs must be HTTP(S), credential-free, and fragment-free")
        return parsed.hostname.rstrip(".").encode("idna").decode("ascii").lower()

    def fetch(
        self,
        urls: Sequence[str],
        *,
        approved_urls: Sequence[str] = (),
        approved_domains: Sequence[str] = (),
    ) -> list[dict[str, str]]:
        if isinstance(urls, (str, bytes)) or not isinstance(urls, Sequence):
            raise ValueError("research URLs must be a sequence of URLs")
        if (
            isinstance(approved_urls, (str, bytes))
            or not isinstance(approved_urls, Sequence)
            or isinstance(approved_domains, (str, bytes))
            or not isinstance(approved_domains, Sequence)
        ):
            raise ValueError("URL and domain approvals must be explicit sequences")
        if len(urls) > self.max_urls:
            raise ValueError(f"web research supports at most {self.max_urls} URLs")
        if not all(isinstance(url, str) for url in approved_urls):
            raise ValueError("each approved URL must be a string")
        if not all(isinstance(domain, str) for domain in approved_domains):
            raise ValueError("each approved domain must be a string")
        approved_exact = set(approved_urls)
        domains = {self._host(domain) for domain in approved_domains}
        exact_hosts = {self._url_host(url) for url in approved_exact}
        permitted_hosts = domains | exact_hosts
        for url in urls:
            if not isinstance(url, str):
                raise ValueError("each research URL must be a string")
            host = self._url_host(url)
            if not self._is_approved(url, host, approved_exact, domains):
                raise ValueError(f"research URL is not explicitly approved: {url}")
            # The SSRF-safe transport's allowlist is exact-host based. Add each
            # approved URL's host so domain approvals can also authorize subdomains.
            permitted_hosts.add(host)

        def validate_target(target: str) -> None:
            host = self._url_host(target)
            if not self._is_approved(target, host, approved_exact, domains):
                raise ValueError(f"redirect URL is not explicitly approved: {target}")

        results: list[dict[str, str]] = []
        for url in urls:
            fetched = self.fetcher.fetch(
                url, allowed_hosts=permitted_hosts, validate_url=validate_target
            )
            text = str(fetched["text"])
            # Fetcher intentionally returns a stable minimal result; HTML parsing is
            # harmless for text responses and keeps source excerpts readable.
            parser = _VisibleText()
            parser.feed(text)
            visible = " ".join(" ".join(parser.parts).split())
            title = (
                " ".join(" ".join(parser.title_parts).split())[:500]
                or str(fetched["url"])
            )
            if not visible:
                visible = " ".join(text.split())
            content = visible[: self.max_content_chars]
            if not content:
                raise ValueError(f"approved URL returned no readable text: {url}")
            results.append(
                {"title": title, "content": content, "url": str(fetched["url"])}
            )
        return results

    @staticmethod
    def _is_approved(
        url: str, host: str, approved_urls: set[str], approved_domains: set[str]
    ) -> bool:
        return url in approved_urls or any(
            host == domain or host.endswith("." + domain) for domain in approved_domains
        )