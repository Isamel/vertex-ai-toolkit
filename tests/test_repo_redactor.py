"""Tests for the secret redaction pipeline (SPEC-V2-REPO-10)."""

from __future__ import annotations

import json
from pathlib import Path

from vaig.core.repo_redactor import (
    RedactionEntry,
    SecretRedactor,
    is_high_entropy_secret,
    write_redaction_log,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REDACTOR = SecretRedactor()


# ---------------------------------------------------------------------------
# 1. AWS access key is redacted
# ---------------------------------------------------------------------------


def test_aws_access_key_is_redacted() -> None:
    content = "export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\n"
    result = REDACTOR.redact(content, file_path="env.sh")

    assert "AKIAIOSFODNN7EXAMPLE" not in result.redacted_content
    assert any(e.kind == "aws_access_key" for e in result.redactions)
    assert not result.skipped


# ---------------------------------------------------------------------------
# 2. JWT token is redacted
# ---------------------------------------------------------------------------


def test_jwt_token_is_redacted() -> None:
    jwt = (
        "eyJhbGciOiJSUzI1NiJ9"
        ".eyJzdWIiOiJ1c2VyMTIzIn0"
        ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    content = f"Authorization: Bearer {jwt}\n"
    result = REDACTOR.redact(content, file_path="request.http")

    assert jwt not in result.redacted_content
    assert any(e.kind in {"jwt", "bearer_token"} for e in result.redactions)


# ---------------------------------------------------------------------------
# 3. YAML key `password:` is redacted with length-preserving placeholder
# ---------------------------------------------------------------------------


def test_yaml_password_key_is_redacted() -> None:
    # Use a value ≥20 chars so the 20-char minimum check passes.
    # Values shorter than 20 chars are intentionally left unchanged to avoid
    # truncating the <redacted:...> placeholder.
    content = "database:\n  password: S3cr3t-P@ssw0rd-12345\n"
    result = REDACTOR.redact(content, file_path="values.yaml")

    assert "S3cr3t-P@ssw0rd-12345" not in result.redacted_content
    # Placeholder must start with <redacted:password>
    assert "<redacted:password>" in result.redacted_content
    # Length of the value portion must be preserved
    original_value = "S3cr3t-P@ssw0rd-12345"
    expected_len = len(original_value)
    # Find the replaced value in the redacted line
    line = [ln for ln in result.redacted_content.splitlines() if "password" in ln][0]
    value_part = line.split(":", 1)[1].strip()
    assert len(value_part) == expected_len, (
        f"Expected length {expected_len}, got {len(value_part)}: {value_part!r}"
    )
    assert any(e.kind == "yaml_key_value" for e in result.redactions)


# ---------------------------------------------------------------------------
# 4. High-entropy 32-char string is redacted
# ---------------------------------------------------------------------------


def test_high_entropy_string_is_redacted() -> None:
    # 32 mixed-case alphanumeric chars with high entropy
    secret = "aB3dEfGhIjKlMnOpQrStUvWxYz012345"
    assert is_high_entropy_secret(secret), "Pre-condition: string should be high entropy"
    content = f"some_var = {secret}\n"
    result = REDACTOR.redact(content, file_path="config.py")

    assert secret not in result.redacted_content
    assert any(e.kind == "high_entropy" for e in result.redactions)


# ---------------------------------------------------------------------------
# 5. SealedSecret file passes through unchanged
# ---------------------------------------------------------------------------


def test_sealed_secret_passes_through_unchanged() -> None:
    content = (
        "apiVersion: bitnami.com/v1alpha1\n"
        "kind: SealedSecret\n"
        "metadata:\n"
        "  name: mysecret\n"
        "spec:\n"
        "  encryptedData:\n"
        "    password: AgBy3...\n"
    )
    result = REDACTOR.redact(content, file_path="sealed.yaml")

    assert result.redacted_content == content
    assert result.redactions == []
    assert not result.skipped


# ---------------------------------------------------------------------------
# 6. SOPS file passes through unchanged
# ---------------------------------------------------------------------------


def test_sops_file_passes_through_unchanged() -> None:
    content = (
        "sops:\n"
        "    kms: []\n"
        "    gcp_kms:\n"
        "    -   resource_id: projects/my-project/...\n"
        "    encrypted_regex: '^(data|stringData)$'\n"
        "password: ENC[AES256_GCM,...]\n"
    )
    result = REDACTOR.redact(content, file_path="secret.enc.yaml")

    assert result.redacted_content == content
    assert result.redactions == []
    assert not result.skipped


# ---------------------------------------------------------------------------
# 7. High-density file (>10 % redactions) is skipped
# ---------------------------------------------------------------------------


def test_high_density_file_is_skipped() -> None:
    # Build a file where every line has an AWS key → density >> 10 %
    lines = [f"key_{i}=AKIA{'A' * 16}\n" for i in range(20)]
    content = "".join(lines)
    result = REDACTOR.redact(content, file_path="secrets_dump.env")

    assert result.skipped is True
    assert result.skip_reason == "secret_density_too_high"
    assert result.redacted_content == ""


# ---------------------------------------------------------------------------
# 8. Redaction preserves line numbers (byte offsets stable)
# ---------------------------------------------------------------------------


def test_redaction_preserves_line_numbers() -> None:
    content = (
        "line1: innocent\n"
        "line2: AKIAIOSFODNN7EXAMPLE\n"
        "line3: also_innocent\n"
    )
    result = REDACTOR.redact(content, file_path="mixed.yaml")

    assert any(e.line == 2 for e in result.redactions), (
        f"Expected redaction on line 2, got lines: {[e.line for e in result.redactions]}"
    )
    # Ensure total line count is preserved
    original_lines = content.splitlines()
    redacted_lines = result.redacted_content.splitlines()
    assert len(redacted_lines) == len(original_lines)


# ---------------------------------------------------------------------------
# 9. Audit log is written to .vaig/repo-redactions/
# ---------------------------------------------------------------------------


def test_audit_log_is_written(tmp_path: Path) -> None:
    entries = [
        RedactionEntry(file="values.yaml", line=3, kind="aws_access_key", reason="AWS access key ID"),
        RedactionEntry(file="config.py", line=7, kind="jwt", reason="JWT token"),
    ]
    log_path = write_redaction_log("run-abc123", entries, log_dir=tmp_path)

    assert log_path.exists()
    assert log_path.suffix == ".jsonl"
    assert log_path.stem == "run-abc123"

    lines = log_path.read_text().splitlines()
    assert len(lines) == 2


# ---------------------------------------------------------------------------
# 10. Audit log contains no original secret values
# ---------------------------------------------------------------------------


def test_audit_log_contains_no_original_secrets(tmp_path: Path) -> None:
    secret = "AKIAIOSFODNN7EXAMPLE"
    content = f"export KEY={secret}\n"

    result = REDACTOR.redact(content, file_path="env.sh")
    log_path = write_redaction_log("run-noSecret", result.redactions, log_dir=tmp_path)

    log_content = log_path.read_text()
    assert secret not in log_content, (
        "Original secret must NEVER appear in the audit log"
    )
    # Verify entries are valid JSONL
    for line in log_content.splitlines():
        entry = json.loads(line)
        assert "file" in entry
        assert "line" in entry
        assert "kind" in entry
        assert "reason" in entry


# ---------------------------------------------------------------------------
# Bonus: disabled redactor returns content unchanged
# ---------------------------------------------------------------------------


def test_disabled_redactor_passes_through() -> None:
    redactor = SecretRedactor(enabled=False)
    secret = "AKIAIOSFODNN7EXAMPLE"
    content = f"key={secret}\n"
    result = redactor.redact(content, file_path="env.sh")

    assert result.redacted_content == content
    assert result.redactions == []
