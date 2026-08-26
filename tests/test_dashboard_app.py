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

    app_test.button(key="nav_next_pair").click()
    app_test.run()

    assert not app_test.exception, f"Next button raised: {[e.value for e in app_test.exception]}"
    after, _ = _pair(app_test)
    assert after == expected


def test_chart_side_prev_button_switches_pair(app_test):
    before, options = _pair(app_test)
    expected = options[(options.index(before) - 1) % len(options)]

    app_test.button(key="nav_prev_pair").click()
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


def test_volume_toggle_renders_without_errors(app_test):
    """Volume bars are hidden by default; enabling the toggle must not break rendering."""
    app_test.checkbox(key="show_volume").check()
    app_test.run()
    assert not app_test.exception, f"Volume toggle raised: {[e.value for e in app_test.exception]}"

    app_test.checkbox(key="show_volume").uncheck()
    app_test.run()
    assert not app_test.exception


def test_stacked_layout_toggle_and_nav(app_test):
    """'Large stacked' toggle switches to 15m-top/1D-bottom with per-chart nav."""
    app_test.toggle(key="stacked_layout").set_value(True)
    app_test.run()
    assert not app_test.exception, f"Stacked layout raised: {[e.value for e in app_test.exception]}"

    before, options = _pair(app_test)
    expected = options[(options.index(before) + 1) % len(options)]
    app_test.button(key="nav_next_15m").click()
    app_test.run()
    assert not app_test.exception
    after, _ = _pair(app_test)
    assert after == expected


def test_only_with_15m_toggle_defaults_off_and_is_safe(app_test):
    """Chart options checkbox 'Only pairs with 15m data': OFF by default;
    toggling it must not break rendering (demo pairs exist on both TFs,
    so the pair list itself is unchanged here)."""
    box = app_test.checkbox(key="only_with_15m")
    assert box.value is False

    before, options_before = _pair(app_test)
    box.check()
    app_test.run()
    assert not app_test.exception, f"only_with_15m ON raised: {[e.value for e in app_test.exception]}"
    after, options_after = _pair(app_test)
    assert len(options_after) == len(options_before) > 0  # demo: both TFs exist
    assert after == before  # same pair stays selected

    app_test.checkbox(key="only_with_15m").uncheck()
    app_test.run()
    assert not app_test.exception
