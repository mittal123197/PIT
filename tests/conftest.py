"""Test isolation: pin the documented code defaults for every env-tunable
`pit.config.ArenaConfig` field before any test imports `pit`, so the suite's
hardcoded expectations (e.g. "a $100,000 winner gets +25% -> $125,000") stay
correct regardless of what a developer's local .env happens to set for a
live run.

`ArenaConfig`'s fields default to `_env_float(NAME, <code default>)`,
evaluated once at class-definition time (module import), not per-instance —
so whichever value is in `os.environ` the FIRST time `pit.config` is
imported anywhere in the test session sticks for every bare `ArenaConfig()`
in every test file. `pit.db`'s `load_dotenv()` call (triggered by importing
`pit.db`, transitively pulled in by most `pit` modules) has `override=False`
by default, so it never clobbers a key already present in `os.environ` —
only fills in ones that are missing. Setting these explicitly here, at
conftest.py's module level, wins that race: pytest always imports
conftest.py before collecting any test module, so this runs first.
"""
import os

_TEST_DEFAULTS = {
    "PIT_BASE_CAPITAL": "100000",
    "PIT_DAILY_STOP_LOSS_PCT": "3.78",
    "PIT_RISK_REWARD_RATIO": "1.5",
    "PIT_POSITION_STOP_LOSS_PCT": "8.0",
    "PIT_START_ROUND_DAYS": "7",
}
for _key, _val in _TEST_DEFAULTS.items():
    os.environ[_key] = _val
