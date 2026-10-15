"""Canonical AFM-KITE product and Skill identifiers."""

from __future__ import annotations

PRODUCT_NAME = "AFM-KITE"
PRODUCT_SLUG = "afm-kite"
SKILL_DIR_NAME = PRODUCT_SLUG
SKILL_VIRTUAL_PATH = f"/{SKILL_DIR_NAME}/"

# Kept only so historical experiment snapshots can still be read after the
# source Skill directory is renamed.
LEGACY_SKILL_DIR_NAME = "afm-analysis-suite"

__all__ = [
    "LEGACY_SKILL_DIR_NAME",
    "PRODUCT_NAME",
    "PRODUCT_SLUG",
    "SKILL_DIR_NAME",
    "SKILL_VIRTUAL_PATH",
]
