"""Migration source adapters package."""
from vaig.core.migration.adapters.base import SourceAdapter, SourceAdapterRegistry, UnknownSourceError
from vaig.core.migration.adapters.pentaho import PentahoAdapter

__all__ = ["SourceAdapter", "SourceAdapterRegistry", "UnknownSourceError", "PentahoAdapter"]
