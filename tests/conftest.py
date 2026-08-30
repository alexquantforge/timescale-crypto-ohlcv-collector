"""
Test isolation: never let a developer's local .env change what the suite asserts.

Settings uses pydantic-settings with env_file=".env", so a bare `Settings()` —
and the module-level `settings` singleton that every engine imports — silently
picks up whatever the machine running the tests happens to have configured.
On a perfectly good checkout this alone makes three tests fail:

    test_dash_warm_neighbors_default_and_alias         (DASH_WARM_NEIGHBORS=2)
    test_filter_preserves_input_order_with_no_filters  (ALLOWED_EXCHANGES=...)
    test_engine_configured_exchanges_deny_only_...     (ALLOWED_EXCHANGES=...)

Pointing env_file at a guaranteed-nonexistent path makes pydantic-settings read
no file at all, so every test sees the documented model defaults. This runs at
conftest import time (collection order guarantees it happens before any test
module imports `config.settings`, which is when the singleton is built).
Actual .env *parsing* stays covered by tests/test_settings.py, which always
constructs `Settings(**overrides)` explicitly; real OS environment variables
still win (they outrank env_file in pydantic-settings), so CI can override.

NOTE the importlib dance below: config/__init__.py re-exports the *instance*
under the name `config.settings`, shadowing the submodule attribute
(tests/test_import_order.py exists because of that hazard), so a plain
`import config.settings as m` would bind the Settings object, not the module.
"""

import importlib
import pathlib

_ISOLATING_ENV_FILE = str(
    pathlib.Path(__file__).resolve().parent / "_no_such_env_file_for_tests_.env"
)

try:
    _settings_module = importlib.import_module("config.settings")
except ImportError:  # tests run from an installed package, or no project on sys.path
    _settings_module = None

if _settings_module is not None:
    _settings_module.Settings.model_config = {
        **_settings_module.Settings.model_config,
        "env_file": _ISOLATING_ENV_FILE,
    }
