"""Small public-egress HTTP client with DNS pinning and bounded responses."""

from __future__ import annotations

import http.client
import ipaddress
import socket
import ssl
from urllib.parse import urlsplit


def _public_addresses(host: str, port: int) -> list[str]:
    try:
        literal = ipaddress.ip_address(host)
        addresses = [str(literal)]
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise ValueError(f"could not resolve public host {host!r}") from exc
        addresses = list(dict.fromkeys(info[4][0] for info in infos))
    if not addresses:
        raise ValueError("host did not resolve to an address")
    parsed = [ipaddress.ip_address(address) for address in addresses]
    # Reject mixed public/private DNS answers too, to block rebinding/fallback tricks.
    if not all(address.is_global for address in parsed):
        raise ValueError("requests to private, local, or reserved network addresses are blocked")
    return [str(address) for address in parsed]


def request_public(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    timeout: float = 8,
    max_bytes: int = 1_000_000,
    allowed_hosts: set[str] | None = None,
) -> tuple[int, dict[str, str], bytes, str]:
    """Issue one pinned-IP request; callers must validate any redirect themselves."""
    if method not in {"GET", "POST"}:
        raise ValueError("only GET and POST are supported")
    if not 0 < timeout <= 30 or not 1 <= max_bytes <= 5_000_000:
        raise ValueError("HTTP timeout or response size is out of range")
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        raise ValueError("URL must be HTTP(S), contain no credentials, and have no fragment")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise ValueError("invalid URL port") from exc
    if port != (443 if parsed.scheme == "https" else 80):
        raise ValueError("non-standard network ports are blocked")
    host = parsed.hostname.rstrip(".").encode("idna").decode("ascii").lower()
    if not host or host == "localhost" or host.endswith(".localhost"):
        raise ValueError("local hosts are blocked")
    if allowed_hosts is not None and host not in allowed_hosts:
        raise ValueError(f"host {host!r} is not allowlisted")
    addresses = _public_addresses(host, port)
    request_headers = dict(headers or {})
    forbidden = {"host", "connection", "transfer-encoding", "proxy-authorization"}
    if any(name.lower() in forbidden for name in request_headers):
        raise ValueError("caller cannot override transport security headers")
    request_headers.setdefault("Accept-Encoding", "identity")
    request_headers.setdefault("User-Agent", "FOMO-AI-Brain/1.0")
    request_headers["Connection"] = "close"
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    last_error: OSError | None = None
    for address in addresses:
        connection: http.client.HTTPConnection
        if parsed.scheme == "https":
            connection = _PinnedHTTPSConnection(
                host,
                address,
                port,
                timeout=timeout,
                context=ssl.create_default_context(),
            )
        else:
            connection = _PinnedHTTPConnection(host, address, port, timeout=timeout)
        try:
            connection.request(method, path, body=body, headers=request_headers)
            response = connection.getresponse()
            data = response.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise ValueError("response exceeded the configured size limit")
            return (
                response.status,
                {key.lower(): value for key, value in response.getheaders()},
                data,
                host,
            )
        except OSError as exc:
            last_error = exc
        finally:
            connection.close()
    raise ValueError(f"public host connection failed: {last_error}")


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, host: str, address: str, port: int, **kwargs: object) -> None:
        super().__init__(host, port=port, **kwargs)
        self._address = address

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self._address, self.port), self.timeout, self.source_address
        )


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(
        self,
        host: str,
        address: str,
        port: int,
        *,
        context: ssl.SSLContext,
        timeout: float,
    ) -> None:
        super().__init__(host, port=port, context=context, timeout=timeout)
        self._address = address

    def connect(self) -> None:
        sock = socket.create_connection(
            (self._address, self.port), self.timeout, self.source_address
        )
        try:
            self.sock = self._context.wrap_socket(sock, server_hostname=self.host)
        except BaseException:
            sock.close()
            raise