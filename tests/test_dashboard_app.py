"""
Integration tests for the dashboard UI using Streamlit's AppTest framework.
Runs the app in demo mode (no database) and simulates Prev/Next navigation.

Regression guard for the bug where chart-side buttons modified
st.session_state.sym_ticker after the selectbox was already instantiated
(StreamlitAPIException) — the click must switch the pair cleanly.
"""
import os

import pytest

os.environ.setdefault("DASHBOARD_DEMO", "1")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_FILE = os.path.join(os.path.dirname(__file__), "..", "dashboard", "app.py")


@pytest.fixture(scope="module")
def app_test():
    at = AppTest.from_file(APP_FILE, default_timeout=60)
    at.run()
    assert not at.exception, f"Dashboard failed to render: {[e.value for e in at.exception]}"
    return at


def _pair(at):
    box = at.selectbox(key="sym_ticker")
    return box.value, list(box.options)


def test_dashboard_boots_in_demo_mode(app_test):
    value, options = _pair(app_test)
    assert value in options
    assert len(options) > 0


def test_chart_side_next_button_switches_pair(app_test):
    before, options = _pair(app_test)
    expected = options[(options.index(before) + 1) % len(options)]

    app_test.button(key="nav_next_15m").click()
    app_test.run()

    assert not app_test.exception, f"Next button raised: {[e.value for e in app_test.exception]}"
    after, _ = _pair(app_test)
    assert after == expected


def test_chart_side_prev_button_switches_pair(app_test):
    before, options = _pair(app_test)
    expected = options[(options.index(before) - 1) % len(options)]

    app_test.button(key="nav_prev_1D").click()
    app_test.run()

    assert not app_test.exception, f"Prev button raised: {[e.value for e in app_test.exception]}"
    after, _ = _pair(app_test)
    assert after == expected


def test_top_row_buttons_switch_pair(app_test):
    before, options = _pair(app_test)

    app_test.button(key="sel_next").click()
    app_test.run()

    assert not app_test.exception
    after, _ = _pair(app_test)
    assert after == options[(options.index(before) + 1) % len(options)]
