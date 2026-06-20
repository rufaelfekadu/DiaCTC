"""Shared helpers for metric computation."""

import logging
import os


def read_pair(original, target):
    """Read a reference/hypothesis pair into two lists of lines.

    Accepts, for each argument:
      - a path to an existing file (lines are read from disk),
      - a list of lines, or
      - a single multi-line string.

    Both arguments must be of compatible kinds. When line counts differ, the
    longer side is truncated to the shorter length (with a warning).
    """
    if isinstance(original, str) and isinstance(target, str) \
            and os.path.isfile(original) and os.path.isfile(target):
        with open(original, "r", encoding="utf-8") as file:
            original_content = file.readlines()
        with open(target, "r", encoding="utf-8") as file:
            target_content = file.readlines()
    elif isinstance(original, (list, tuple)) and isinstance(target, (list, tuple)):
        original_content = list(original)
        target_content = list(target)
    elif isinstance(original, str) and isinstance(target, str):
        original_content = original.splitlines()
        target_content = target.splitlines()
    else:
        raise ValueError(
            "Invalid input types for original and target. "
            "Expected file paths, lists of lines, or multi-line strings."
        )

    if len(original_content) != len(target_content):
        logging.warning(
            f"Line count mismatch: original has {len(original_content)} lines, "
            f"target has {len(target_content)} lines. Processing up to minimum length."
        )
        min_len = min(len(original_content), len(target_content))
        original_content = original_content[:min_len]
        target_content = target_content[:min_len]

    return original_content, target_content
