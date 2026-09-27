"""Validation for API messages and agent requests."""

from typing import Any
from urllib.parse import urlsplit

from fomo.brain.context import Message


def _validated_http_url(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 2_000:
        raise ValueError(f"{label} must be a non-empty URL under 2,000 characters")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid HTTP(S) URL") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not hostname
        or parsed.username
        or parsed.password
        or parsed.fragment
        or port not in (None, 80, 443)
    ):
        raise ValueError(f"{label} must be credential-free HTTP(S) without a fragment")
    try:
        hostname.rstrip(".").encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError(f"{label} has an invalid hostname") from exc
    return value


def _validated_domain(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 253:
        raise ValueError("approved domains must be non-empty hostnames")
    candidate = value if "://" in value else f"https://{value}"
    try:
        parsed = urlsplit(candidate)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError("approved domains must be hostnames without credentials") from exc
    if (
        not hostname
        or parsed.username
        or parsed.password
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
        or port not in (None, 443)
    ):
        raise ValueError("approved domains must be hostnames without credentials")
    try:
        return hostname.rstrip(".").encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise ValueError("approved domain has an invalid hostname") from exc


def validate_messages(value: Any) -> list[Message]:
    if not isinstance(value, list) or not 1 <= len(value) <= 32:
        raise ValueError("messages must be an array of 1–32 items")
    messages = []
    total = 0
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != {"role", "content"}:
            raise ValueError("each message must have only role and content")
        if raw["role"] not in ("system", "user", "assistant"):
            raise ValueError("only system, user and assistant messages are accepted")
        message = Message(raw["role"], raw["content"])
        total += len(message.content)
        if total > 32_000:
            raise ValueError("combined message text exceeds 32,000 characters")
        messages.append(message)
    if messages[-1].role != "user":
        raise ValueError("last message must be from the user")
    return messages


def validate_task(value: Any) -> tuple[str, list[dict[str, str]], dict[str, Any]]:
    allowed_fields = {"prompt", "sources", "browse", "execution"}
    if not isinstance(value, dict) or set(value) - allowed_fields or "prompt" not in value:
        raise ValueError("expected prompt and optional sources, browse, and execution")
    prompt = value["prompt"]
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 10_000:
        raise ValueError("prompt must contain 1–10,000 characters")
    sources = value.get("sources", [])
    if not isinstance(sources, list) or len(sources) > 20:
        raise ValueError("sources must be an array of up to 20 items")
    for item in sources:
        if not isinstance(item, dict) or set(item) != {"title", "content"}:
            raise ValueError("each source needs title and content")
        if not isinstance(item["title"], str) or not isinstance(item["content"], str):
            raise ValueError("source title and content must be text")
        if not item["title"].strip() or len(item["title"]) > 200 or len(item["content"]) > 4000:
            raise ValueError("source exceeds length limit or has no title")
    task_context: dict[str, Any] = {}
    if "browse" in value:
        browse = value["browse"]
        if not isinstance(browse, dict) or set(browse) != {
            "urls",
            "approved_urls",
            "approved_domains",
        }:
            raise ValueError(
                "browse must contain only urls, approved_urls, and approved_domains"
            )
        urls = browse["urls"]
        approved_urls = browse["approved_urls"]
        approved_domains = browse["approved_domains"]
        if not isinstance(urls, list) or not 1 <= len(urls) <= 10:
            raise ValueError("browse urls must contain 1–10 URLs")
        if not isinstance(approved_urls, list) or len(approved_urls) > 10:
            raise ValueError("approved_urls must be an array of up to 10 URLs")
        if not isinstance(approved_domains, list) or len(approved_domains) > 20:
            raise ValueError("approved_domains must be an array of up to 20 domains")
        checked_urls = [_validated_http_url(url, "browse URL") for url in urls]
        checked_approved_urls = [
            _validated_http_url(url, "approved URL") for url in approved_urls
        ]
        checked_domains = [_validated_domain(domain) for domain in approved_domains]
        if len(set(checked_urls)) != len(checked_urls):
            raise ValueError("browse URLs must be unique")
        for url in checked_urls:
            host = urlsplit(url).hostname.rstrip(".").encode("idna").decode("ascii").lower()
            if url not in checked_approved_urls and not any(
                host == domain or host.endswith("." + domain) for domain in checked_domains
            ):
                raise ValueError(f"browse URL is not explicitly approved: {url}")
        task_context.update(
            {
                "browse": True,
                "research_urls": tuple(checked_urls),
                "approved_urls": tuple(checked_approved_urls),
                "approved_domains": tuple(checked_domains),
            }
        )
    if "execution" in value:
        execution = value["execution"]
        if not isinstance(execution, dict) or set(execution) != {
            "code",
            "timeout_seconds",
        }:
            raise ValueError("execution must contain only code and timeout_seconds")
        code = execution["code"]
        timeout_seconds = execution["timeout_seconds"]
        if not isinstance(code, str) or not code.strip() or len(code) > 20_000:
            raise ValueError("execution code must contain 1–20,000 characters")
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 30:
            raise ValueError("execution timeout must be between 1 and 30 seconds")
        # Authorization is assigned by ChatService only after checking server
        # configuration; request data itself never contains an authorization claim.
        task_context["requested_execution"] = {
            "code": code,
            "timeout_seconds": timeout_seconds,
        }
    return prompt.strip(), sources, task_context