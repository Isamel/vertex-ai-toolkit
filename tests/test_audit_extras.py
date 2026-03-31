"""Tests for ``[audit]`` extras lazy import safety.

Verifies that:
- Importing ``vaig.core.quota`` and ``vaig.core.subscribers.audit_subscriber``
  **does not crash** even when google-cloud packages are absent at module level.
- Activating the features without the extras gives a clear ``ImportError`` with
  ``pip install`` instructions.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest


class TestModuleImportSafety:
    """Modules can be imported without [audit] extras installed."""

    def test_quota_module_imports_without_gcs(self) -> None:
        """vaig.core.quota can be imported when google.cloud.storage is missing."""
        # Temporarily hide google.cloud.storage from the import system.
        hidden = {"google.cloud.storage": None}
        # We must also hide the already-imported module so the re-import
        # doesn't just find the cached version.
        mod_name = "vaig.core.quota"
        saved = sys.modules.pop(mod_name, None)
        try:
            with patch.dict("sys.modules", hidden):
                mod = importlib.import_module(mod_name)
                assert hasattr(mod, "QuotaChecker")
        finally:
            if saved is not None:
                sys.modules[mod_name] = saved

    def test_audit_subscriber_module_imports_without_bq(self) -> None:
        """vaig.core.subscribers.audit_subscriber can be imported without BQ/Logging."""
        hidden = {
            "google.cloud.bigquery": None,
            "google.cloud.logging": None,
        }
        mod_name = "vaig.core.subscribers.audit_subscriber"
        saved = sys.modules.pop(mod_name, None)
        try:
            with patch.dict("sys.modules", hidden):
                mod = importlib.import_module(mod_name)
                assert hasattr(mod, "AuditSubscriber")
        finally:
            if saved is not None:
                sys.modules[mod_name] = saved


class TestFeatureWithoutExtras:
    """Enabling audit/rate-limit without extras gives clear errors."""

    def test_quota_checker_init_raises_without_gcs(self) -> None:
        """QuotaChecker.__init__ raises ImportError when GCS is unavailable."""
        from unittest.mock import MagicMock

        from vaig.core.quota import QuotaChecker

        settings = MagicMock()
        settings.rate_limit.policy_gcs_bucket = "bucket"
        settings.rate_limit.policy_gcs_path = "policy.yaml"
        settings.rate_limit.cache_ttl_seconds = 300

        with patch.dict("sys.modules", {"google.cloud.storage": None}):
            with pytest.raises(ImportError, match="pip install 'vertex-ai-toolkit\\[audit\\]'"):
                QuotaChecker(settings=settings, credentials=MagicMock())

    def test_audit_subscriber_init_raises_without_bq(self) -> None:
        """AuditSubscriber.__init__ raises ImportError when BQ/Logging missing."""
        from unittest.mock import MagicMock

        from vaig.core.subscribers.audit_subscriber import AuditSubscriber

        settings = MagicMock()
        settings.audit.bigquery_table = "project.dataset.table"
        settings.audit.cloud_logging_log_name = "audit-log"
        settings.audit.buffer_size = 20

        with patch.dict("sys.modules", {"google.cloud.bigquery": None}):
            with pytest.raises(ImportError, match="pip install 'vertex-ai-toolkit\\[audit\\]'"):
                AuditSubscriber(settings=settings, credentials=MagicMock())
