"""Bounded client for explicitly allowlisted external API origins and paths."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib.parse import quote, unquote, urlsplit

from ._http import request_public


@dataclass(frozen=True)
class ApiPermission:
    host: str
    path_prefixes: tuple[str, ...]
    methods: tuple[str, ...] = ("GET",)
    token_env: str | None = None


class AllowlistedApiClient:
    def __init__(
        self,
        permissions: dict[str, ApiPermission],
        *,
        timeout_seconds: float = 8,
        max_response_bytes: int = 512_000,
    ) -> None:
        self.permissions = {name: permission for name, permission in permissions.items()}
        self.timeout_seconds = timeout_seconds
        self.max_response_bytes = max_response_bytes
        if not self.permissions:
            raise ValueError("at least one explicitly configured API permission is required")
        for name, permission in self.permissions.items():
            if not name or not isinstance(permission, ApiPermission):
                raise ValueError("API permissions must be named ApiPermission entries")
            host = permission.host.lower().rstrip(".").encode("idna").decode("ascii")
            if (
                not host
                or "/" in host
                or ":" in host
                or "@" in host
                or not permission.path_prefixes
                or not permission.methods
                or any(method not in {"GET", "POST"} for method in permission.methods)
                or any(not path.startswith("/") or ".." in path for path in permission.path_prefixes)
            ):
                raise ValueError(f"invalid API permission configuration for {name!r}")
            self.permissions[name] = ApiPermission(
                host=host,
                path_prefixes=permission.path_prefixes,
                methods=permission.methods,
                token_env=permission.token_env,
            )

    def request(
        self,
        name: str,
        *,
        method: str,
        path: str,
        json_body: dict | None = None,
    ) -> dict:
        permission = self.permissions.get(name)
        if permission is None:
            raise ValueError(f"API {name!r} is not allowlisted")
        if method not in permission.methods:
            raise ValueError(f"method {method!r} is not permitted for API {name!r}")
        parsed = urlsplit(path)
        decoded_path = unquote(parsed.path)
        if (
            parsed.scheme
            or parsed.netloc
            or parsed.query
            or parsed.fragment
            or not decoded_path.startswith("/")
            or "\\" in decoded_path
            or "%" in decoded_path
            or any(segment in {".", ".."} for segment in decoded_path.split("/"))
            or not any(decoded_path == prefix or decoded_path.startswith(prefix.rstrip("/") + "/")
                       for prefix in permission.path_prefixes)
        ):
            raise ValueError("API path is outside the configured path allowlist")
        normalized_path = quote(decoded_path, safe="/:@!$&'()*+,;=-._~")
        url = f"https://{permission.host}{normalized_path}"
        body = None
        headers = {"Accept": "application/json"}
        if method == "POST":
            if not isinstance(json_body, dict):
                raise ValueError("POST requests require a JSON object body")
            body = json.dumps(json_body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            if len(body) > self.max_response_bytes:
                raise ValueError("API request body exceeds the configured size limit")
            headers["Content-Type"] = "application/json"
        elif json_body is not None:
            raise ValueError("GET requests do not accept a request body")
        if permission.token_env:
            token = os.environ.get(permission.token_env)
            if not token:
                raise RuntimeError(
                    f"Required API credential environment variable {permission.token_env!r} is unset"
                )
            headers["Authorization"] = f"Bearer {token}"
        status, response_headers, raw, host = request_public(
            url,
            method=method,
            headers=headers,
            body=body,
            timeout=self.timeout_seconds,
            max_bytes=self.max_response_bytes,
            allowed_hosts={permission.host},
        )
        if not 200 <= status < 300:
            raise RuntimeError(f"allowlisted API returned HTTP {status}")
        if response_headers.get("content-type", "").split(";", 1)[0].lower() != "application/json":
            raise RuntimeError("allowlisted API did not return application/json")
        try:
            response = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("allowlisted API returned invalid JSON") from exc
        return {"status": status, "host": host, "data": response}