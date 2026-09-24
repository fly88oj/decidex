"""Decidex: a local, open reimplementation of the Jev / TypeSafe "System One" API.

Unstructured state in, typed calibrated decisions out. See RESEARCH.md for the
official API evidence this package replicates.
"""

__version__ = "1.0.0"

MODEL_ALIAS = "decidex-latest"
MODEL_ID = f"decidex-{__version__}"
DEFAULT_BASE_URL = "http://127.0.0.1:8600"

# SDK re-exports at the bottom (constants must be defined first to avoid
# circular-import issues when decidex.sdk imports from this module).
from decidex.sdk import (  # noqa: E402,F401
    Choice,
    ChoiceAnswer,
    DecidexClient,
    DecidexError,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    SystemOneResponse,
)
