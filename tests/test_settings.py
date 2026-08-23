"""
Settings parsing tests: ALLOWED_EXCHANGES accepts comma-separated and JSON formats.
"""
from config.settings import Settings


def test_allowed_exchanges_comma_separated():
    st = Settings(ALLOWED_EXCHANGES="bybit, okx , bitget")
    assert st.allowed_exchanges == ["bybit", "okx", "bitget"]


def test_allowed_exchanges_json():
    st = Settings(ALLOWED_EXCHANGES='["bybit","okx"]')
    assert st.allowed_exchanges == ["bybit", "okx"]


def test_allowed_exchanges_empty():
    st = Settings(ALLOWED_EXCHANGES="")
    assert st.allowed_exchanges == []
