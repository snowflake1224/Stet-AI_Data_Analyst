from __future__ import annotations
import base64
import hashlib
import os
import threading
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from openai import OpenAI, APIError, RateLimitError

log = logging.getLogger("agent.model")

# A short retry often lands in the same provider rate-limit window.
_DEFAULT_RETRY_DELAYS = (20.0, 40.0, 65.0)
_pace_lock = threading.Lock()
_next_allowed: dict[str, float] = {}

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_MODEL = "gemini-3.5-flash-lite"
OPENAI_URL = "https://api.openai.com/v1"
OPENAI_MODEL_PRIORITY = ("gpt-5-mini", "gpt-4.1-mini", "gpt-4o-mini")


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    api_key: str
    api_url: str
    model: str
    reasoning_effort: str = ""
    min_interval: float = 0.0
    retry_delays: tuple[float, ...] = _DEFAULT_RETRY_DELAYS

    @classmethod
    def from_env(cls) -> "ModelConfig":
        provider = os.environ.get("MODEL_PROVIDER", "gemini").strip().lower()
        return cls(
            provider=provider,
            api_key=os.environ.get("MODEL_API_KEY", "EMPTY"),
            api_url=os.environ.get(
                "MODEL_API_URL", GEMINI_URL if provider == "gemini" else OPENAI_URL
            ).rstrip("/"),
            model=os.environ.get(
                "MODEL_NAME", GEMINI_MODEL if provider == "gemini" else "gpt-4.1-mini"
            ),
            reasoning_effort=os.environ.get("MODEL_REASONING_EFFORT", "").strip(),
            min_interval=float(
                os.environ.get("MODEL_MIN_INTERVAL_SECONDS", "0") or 0
            ),
        )


def config_for_provider(provider: str, api_key: str) -> ModelConfig:
    """Create a web-job config without mutating process environment variables."""
    provider = provider.strip().lower()
    if provider == "gemini":
        return ModelConfig(
            provider="gemini",
            api_key=api_key,
            api_url=GEMINI_URL,
            model=GEMINI_MODEL,
            reasoning_effort="minimal",
            min_interval=7.0,
        )
    if provider == "openai":
        return ModelConfig(
            provider="openai",
            api_key=api_key,
            api_url=OPENAI_URL,
            model="auto",
            min_interval=0.0,
            retry_delays=(2.0, 5.0, 10.0),
        )
    raise ValueError("Provider must be 'gemini' or 'openai'.")


def redact_secret(text: str, secret: str) -> str:
    """Remove a job credential before an error is logged or returned."""
    result = str(text)
    if secret:
        result = result.replace(secret, "[key]")
    return result


class ModelClient:
    """
    OpenAI-compatible client with retry + exponential back-off.

    Local dev (Gemini free):
        MODEL_API_URL = https://generativelanguage.googleapis.com/v1beta/openai/
        MODEL_API_KEY = <your Gemini key>
        MODEL_NAME    = gemini-3.5-flash-lite

    Official eval (model injected by organizers):
        MODEL_API_URL / MODEL_API_KEY / MODEL_NAME auto-injected.
        Leave MODEL_REASONING_EFFORT and MODEL_MIN_INTERVAL_SECONDS unset there.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig.from_env()
        self.provider = self.config.provider
        self.model = self.config.model
        self.api_url = self.config.api_url.rstrip("/")
        self.api_key = self.config.api_key
        self.reasoning_effort = self.config.reasoning_effort
        self.min_interval = self.config.min_interval
        self._client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        if self.provider == "openai" and self.model == "auto":
            self.model = self._select_openai_model()
        self._pace_key = hashlib.sha256(
            f"{self.provider}:{self.api_key}".encode()
        ).hexdigest()

    def _select_openai_model(self) -> str:
        """Pick the cheapest preferred model visible to this OpenAI project."""
        try:
            available = {item.id for item in self._client.models.list().data}
            for model in OPENAI_MODEL_PRIORITY:
                if model in available:
                    return model
        except Exception as exc:
            log.info("Could not list OpenAI models; using gpt-4.1-mini: %s",
                     redact_secret(str(exc), self.api_key))
        return "gpt-4.1-mini"

    def _pace(self) -> None:
        """Keep local calls under the free-tier requests-per-minute cap."""
        if self.min_interval <= 0:
            return
        with _pace_lock:
            now = time.monotonic()
            wait = _next_allowed.get(self._pace_key, 0.0) - now
            if wait > 0:
                log.info(f"Pacing: waiting {wait:.1f}s before the next model call")
                time.sleep(wait)
                now = time.monotonic()
            _next_allowed[self._pace_key] = now + self.min_interval

    def _request(self, **kwargs):
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        self._pace()
        return self._client.chat.completions.create(**kwargs)

    def _call(self, **kwargs) -> str:
        last_exc: Exception | None = None
        attempts = (0.0,) + self.config.retry_delays
        retry_override = 0.0
        for attempt, delay in enumerate(attempts):
            delay = max(delay, retry_override)
            retry_override = 0.0
            if delay:
                log.warning(
                    "Retry %s/%s after %ss ...",
                    attempt, len(self.config.retry_delays), delay,
                )
                time.sleep(delay)
            try:
                resp = self._request(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_exc = e
                safe_error = redact_secret(str(e), self.api_key)
                if _daily_quota(e) or _zero_quota(e) or _billing_error(e):
                    raise RuntimeError(
                        f"{self.provider.title()} has no available quota or billing credit "
                        f"for {self.model}. Details: {safe_error}"
                    ) from e
                if _is_rate_limit(e):
                    log.warning("Temporary provider limit, backing off: %s", safe_error)
                    retry_override = _retry_after(e)
                    continue
                status = getattr(e, "status_code", None)
                if isinstance(e, APIError) and status and int(status) < 500:
                    raise RuntimeError(
                        f"Model API error {status}: {safe_error}"
                    ) from e
                log.warning("Model call failed (will retry): %s", safe_error)
        raise RuntimeError(
            f"Model API failed after {len(attempts)} attempts: "
            f"{redact_secret(str(last_exc), self.api_key)}"
        ) from last_exc

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        return self._call(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def chat_with_image(self, messages: list[dict], image_path: Path) -> str:
        ext  = image_path.suffix.lower().lstrip(".")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png", "gif": "image/gif",
                "webp": "image/webp"}.get(ext, "image/png")
        b64  = base64.b64encode(image_path.read_bytes()).decode()

        augmented = []
        for i, msg in enumerate(messages):
            if i == len(messages) - 1 and msg["role"] == "user":
                augmented.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": msg["content"]},
                        {"type": "image_url",
                         "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ],
                })
            else:
                augmented.append(msg)

        return self._call(
            model=self.model,
            messages=augmented,
            temperature=0.0,
            max_tokens=4096,
        )


def _is_rate_limit(exc: Exception) -> bool:
    if isinstance(exc, RateLimitError):
        return True
    return getattr(exc, "status_code", None) == 429


def _daily_quota(exc: Exception) -> bool:
    text = str(exc).lower()
    return "per day" in text or "daily" in text or "requests per day" in text


def _zero_quota(exc: Exception) -> bool:
    text = str(exc).lower()
    return "limit: 0" in text or "quota_limit_value: 0" in text


def _billing_error(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = ("insufficient_quota", "billing", "credit balance", "payment required")
    return any(marker in text for marker in markers)


def _retry_after(exc: Exception) -> float:
    """Read a provider Retry-After header without assuming an SDK version."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}
    value = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return max(0.0, min(float(value), 120.0))
    except (TypeError, ValueError):
        return 0.0
