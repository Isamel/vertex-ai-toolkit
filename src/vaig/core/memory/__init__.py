"""Pattern memory subsystem for recurrence detection and fix outcome tracking.

Exports the public API for the memory module:
- ``ObservationFingerprint``: fingerprint computation for findings
- ``PatternEntry``, ``RecurrenceSignal``: memory data models
- ``FixOutcome``: fix outcome data model (MEM-03)
- ``PatternMemoryStore``: JSONL-based persistent store
- ``FixOutcomeStore``: JSONL-based fix outcome store (MEM-03)
- ``RecurrenceAnalyzer``: recurrence detection engine
- ``MemoryRAGIndex``, ``build_narrative``: semantic memory RAG index (MEM-04)
"""

from vaig.core.memory.fingerprint import ObservationFingerprint
from vaig.core.memory.memory_rag import MemoryRAGIndex, build_narrative
from vaig.core.memory.models import FixOutcome, PatternEntry, RecurrenceSignal
from vaig.core.memory.outcome_store import FixOutcomeStore
from vaig.core.memory.pattern_store import PatternMemoryStore
from vaig.core.memory.recurrence import RecurrenceAnalyzer

__all__ = [
    "FixOutcome",
    "FixOutcomeStore",
    "MemoryRAGIndex",
    "ObservationFingerprint",
    "PatternEntry",
    "PatternMemoryStore",
    "RecurrenceAnalyzer",
    "RecurrenceSignal",
    "build_narrative",
]

