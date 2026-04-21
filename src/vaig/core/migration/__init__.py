"""vaig.core.migration — shared components for code migration."""
from vaig.core.migration.config import MigrationConfig, detect_source_kind
from vaig.core.migration.domain import DomainModel, DomainNode
from vaig.core.migration.jail import ReadOnlyFilesystemJail, ReadOnlySourceError

__all__ = [
    "MigrationConfig",
    "DomainModel",
    "DomainNode",
    "ReadOnlyFilesystemJail",
    "ReadOnlySourceError",
    "detect_source_kind",
]
