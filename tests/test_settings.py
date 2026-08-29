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


def test_dash_warm_neighbors_default_and_alias():
    st = Settings()
    assert st.dash_warm_neighbors == 5
    st2 = Settings(DASH_WARM_NEIGHBORS="0")
    assert st2.dash_warm_neighbors == 0
