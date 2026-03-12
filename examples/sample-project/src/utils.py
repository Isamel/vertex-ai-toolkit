"""Utility functions for the task manager."""

from __future__ import annotations

from datetime import datetime, timedelta


def parse_date(date_str: str) -> datetime:
    """Parse a date string in common formats.

    Supports:
        - YYYY-MM-DD
        - YYYY-MM-DD HH:MM
        - "today", "tomorrow", "+Nd" (N days from now)

    Args:
        date_str: The date string to parse.

    Returns:
        Parsed datetime.

    Raises:
        ValueError: If the format is not recognized.
    """
    date_str = date_str.strip().lower()

    if date_str == "today":
        return datetime.now().replace(hour=23, minute=59, second=59)

    if date_str == "tomorrow":
        return (datetime.now() + timedelta(days=1)).replace(
            hour=23, minute=59, second=59
        )

    # "+3d" means 3 days from now
    if date_str.startswith("+") and date_str.endswith("d"):
        try:
            days = int(date_str[1:-1])
            return (datetime.now() + timedelta(days=days)).replace(
                hour=23, minute=59, second=59
            )
        except ValueError:
            pass

    # Try standard formats
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    raise ValueError(
        f"Unrecognized date format: '{date_str}'. "
        "Use YYYY-MM-DD, 'today', 'tomorrow', or '+Nd'."
    )


def truncate(text: str, max_length: int = 50) -> str:
    """Truncate text with ellipsis if too long.

    Args:
        text: The text to truncate.
        max_length: Maximum allowed length.

    Returns:
        Truncated text with '...' suffix if shortened.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def slugify(text: str) -> str:
    """Convert text to a URL-friendly slug.

    Args:
        text: The text to slugify.

    Returns:
        Lowercase, hyphen-separated string.
    """
    import re

    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")
