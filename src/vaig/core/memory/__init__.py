"""Pattern memory subsystem for recurrence detection.

Exports the public API for the memory module:
- ``ObservationFingerprint``: fingerprint computation for findings
- ``PatternEntry``, ``RecurrenceSignal``: memory data models
- ``PatternMemoryStore``: JSONL-based persistent store
- ``RecurrenceAnalyzer``: recurrence detection engine
"""

from vaig.core.memory.fingerprint import ObservationFingerprint
from vaig.core.memory.models import PatternEntry, RecurrenceSignal
from vaig.core.memory.pattern_store import PatternMemoryStore
from vaig.core.memory.recurrence import RecurrenceAnalyzer

__all__ = [
    "ObservationFingerprint",
    "PatternEntry",
    "PatternMemoryStore",
    "RecurrenceAnalyzer",
    "RecurrenceSignal",
]
