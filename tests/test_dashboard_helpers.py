"""dashboard.helpers — чистые утилиты (без Streamlit-рантайма)."""
from dashboard.helpers import parse_jsonb_config


def test_parse_jsonb_config_dict_passthrough():
    d = {"mode": "recent", "PUMP_MIN_PCT": 10.0}
    assert parse_jsonb_config(d) is d


def test_parse_jsonb_config_str_json():
    # Старые строки pump_scan_runs: json.dumps()+$n::jsonb хранили JSON-строку.
    s = '{"mode": "recent", "PUMP_MIN_PCT": 10.0, "PRE_PUMP_DAYS": 1.0}'
    out = parse_jsonb_config(s)
    assert out == {"mode": "recent", "PUMP_MIN_PCT": 10.0, "PRE_PUMP_DAYS": 1.0}
    assert isinstance(out, dict)


def test_parse_jsonb_config_none_garbage():
    assert parse_jsonb_config(None) == {}
    assert parse_jsonb_config("") == {}
    assert parse_jsonb_config("not a json") == {}
    assert parse_jsonb_config("[1, 2, 3]") == {}  # JSON-массив — не конфиг
    assert parse_jsonb_config(42) == {}
    assert parse_jsonb_config("12") == {}  # JSON-число — не конфиг
