"""
Input validation and sanitization for LFM3-1.7B inference requests.

Provides:
  * Prompt length limits (max tokens / characters)
  * Forbidden pattern detection (prompt injection, jailbreak attempts)
  * Content sanitization (null bytes, control characters)
  * Per-client prompt allow/deny lists

All checks are configurable via environment variables. Validation failures
raise ``ValidationError`` which the server middleware converts to HTTP 400.

Enterprise rationale: LLMs are susceptible to prompt injection, data
exfiltration, and resource-abuse via extremely long prompts. Validation at
the edge is the first line of defense.
"""

import os
import re
from typing import List, Optional, Pattern

# ---------------------------------------------------------------------------
# Configuration (env-overridable)
# ---------------------------------------------------------------------------

MAX_PROMPT_CHARS = int(os.environ.get("LFM3_MAX_PROMPT_CHARS", "4000"))
MAX_PROMPT_TOKENS = int(os.environ.get("LFM3_MAX_PROMPT_TOKENS", "1024"))
MAX_MESSAGES = int(os.environ.get("LFM3_MAX_MESSAGES", "20"))
MAX_MESSAGE_CHARS = int(os.environ.get("LFM3_MAX_MESSAGE_CHARS", "2000"))

# Forbidden patterns (case-insensitive). These detect common prompt-injection
# and jailbreak attempt signatures. Add to this list as new threats emerge.
_FORBIDDEN_PATTERNS_RAW = os.environ.get(
    "LFM3_FORBIDDEN_PATTERNS",
    r"(?i)(ignore\s+all\s+prior\s+instructions|disregard\s+previous\s+instructions|"
    r"system\s*:\s*you\s+are\s+now|jailbreak|dan\s+mode|do\s+anything\s+now|"
    r"reveal\s+your\s+system\s+prompt|print\s+the\s+system\s+prompt|"
    r"output\s+the\s+following\s+json\s*:\s*\{.*\"role\".*\"system\")",
)
_FORBIDDEN_PATTERNS: List[Pattern] = []
for pat in _FORBIDDEN_PATTERNS_RAW.split(","):
    pat = pat.strip()
    if pat:
        _FORBIDDEN_PATTERNS.append(re.compile(pat))

# Allow/deny lists for client identifiers (IPs or API key labels)
_DENY_LIST_RAW = os.environ.get("LFM3_DENY_LIST", "")
DENY_LIST: List[str] = [x.strip() for x in _DENY_LIST_RAW.split(",") if x.strip()]
_ALLOW_LIST_RAW = os.environ.get("LFM3_ALLOW_LIST", "")
ALLOW_LIST: List[str] = [x.strip() for x in _ALLOW_LIST_RAW.split(",") if x.strip()]


class ValidationError(Exception):
    """Raised when a request fails validation."""

    def __init__(self, message: str, code: str = "validation_error"):
        super().__init__(message)
        self.message = message
        self.code = code


def _has_control_chars(text: str) -> bool:
    """Return True if text contains control characters other than common whitespace."""
    for ch in text:
        o = ord(ch)
        if o < 32 and ch not in ("\t", "\n", "\r"):
            return True
        if o == 0x7F:  # DEL
            return True
    return False


def _contains_forbidden(text: str) -> Optional[str]:
    """Return the matched forbidden pattern, or None if clean."""
    for pat in _FORBIDDEN_PATTERNS:
        if pat.search(text):
            return pat.pattern
    return None


def validate_client(client_id: Optional[str]) -> None:
    """Validate client against allow/deny lists.

    client_id may be an IP address or API key label.
    Raises ValidationError if the client is denied or not allowed.
    """
    if client_id and client_id in DENY_LIST:
        raise ValidationError(f"Client '{client_id}' is denied", "client_denied")
    if ALLOW_LIST and client_id and client_id not in ALLOW_LIST:
        raise ValidationError(f"Client '{client_id}' is not in allow list", "client_not_allowed")


def validate_prompt(prompt: str, max_tokens: Optional[int] = None) -> str:
    """
    Validate and sanitize a prompt.

    Returns the sanitized prompt. Raises ValidationError on failure.
    """
    if not isinstance(prompt, str):
        raise ValidationError("Prompt must be a string", "invalid_type")

    # Null byte / control character check
    if "\x00" in prompt or _has_control_chars(prompt):
        raise ValidationError("Prompt contains invalid control characters", "invalid_characters")

    # Length checks
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValidationError(
            f"Prompt too long: {len(prompt)} chars (max {MAX_PROMPT_CHARS})",
            "prompt_too_long",
        )

    # Forbidden pattern check
    forbidden = _contains_forbidden(prompt)
    if forbidden is not None:
        raise ValidationError("Prompt matches forbidden pattern", "forbidden_pattern")

    return prompt


def _msg_role(msg) -> str:
    if isinstance(msg, dict):
        return msg.get("role", "")
    return getattr(msg, "role", "")


def _msg_content(msg) -> str:
    if isinstance(msg, dict):
        return msg.get("content", "")
    return getattr(msg, "content", "")


def validate_messages(messages: list) -> list:
    """
    Validate a list of chat messages.

    Returns the sanitized list. Raises ValidationError on failure.
    """
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValidationError("Messages must be a non-empty list", "invalid_type")
    if len(messages) > MAX_MESSAGES:
        raise ValidationError(
            f"Too many messages: {len(messages)} (max {MAX_MESSAGES})",
            "too_many_messages",
        )

    sanitized = []
    for i, msg in enumerate(messages):
        role = _msg_role(msg)
        content = _msg_content(msg)
        if role not in ("system", "user", "assistant"):
            raise ValidationError(f"Message {i} has invalid role: {role}", "invalid_role")
        if not isinstance(content, str):
            raise ValidationError(f"Message {i} content must be a string", "invalid_type")
        if len(content) > MAX_MESSAGE_CHARS:
            raise ValidationError(
                f"Message {i} content too long: {len(content)} chars "
                f"(max {MAX_MESSAGE_CHARS})",
                "message_too_long",
            )
        # Sanitize content
        clean = validate_prompt(content)
        sanitized.append({"role": role, "content": clean})

    return sanitized


def validate_max_tokens(max_tokens: Optional[int]) -> int:
    """Validate and clamp max_tokens."""
    if max_tokens is None:
        return 50
    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValidationError("max_tokens must be a positive integer", "invalid_max_tokens")
    if max_tokens > MAX_PROMPT_TOKENS:
        raise ValidationError(
            f"max_tokens {max_tokens} exceeds limit {MAX_PROMPT_TOKENS}",
            "max_tokens_too_high",
        )
    return max_tokens


def is_validation_error(exc: Exception) -> bool:
    """Return True if the exception is a ValidationError."""
    return isinstance(exc, ValidationError)


def validation_error_response(exc: ValidationError) -> dict:
    """Build a standardized error response dict for a ValidationError."""
    return {
        "error": {
            "message": exc.message,
            "type": exc.code,
            "code": 400,
        }
    }
