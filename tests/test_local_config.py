from agent.agents.model import _daily_quota, _zero_quota
from serve import key_is_usable


def test_placeholder_and_short_keys_are_rejected():
    assert key_is_usable("") is False
    assert key_is_usable("your_gemini_api_key_here") is False
    assert key_is_usable("short-key") is False
    assert key_is_usable("AIza" + "x" * 20) is True
    assert key_is_usable("line1\nline2-extra-characters") is False


def test_daily_and_zero_quota_are_not_retried_as_ordinary_rate_limits():
    assert _daily_quota(RuntimeError("Quota exceeded: requests per day")) is True
    assert _zero_quota(RuntimeError("quota_limit_value: 0")) is True
    assert _daily_quota(RuntimeError("rate limit exceeded for rpm")) is False
