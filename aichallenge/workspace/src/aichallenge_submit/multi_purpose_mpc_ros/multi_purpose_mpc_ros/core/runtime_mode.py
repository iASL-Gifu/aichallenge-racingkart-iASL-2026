"""Submission runtime switches for optional diagnostics and visualization."""

import os


ENABLE_RUNTIME_DIAGNOSTICS = (
    os.environ.get("AICHALLENGE_ENABLE_RUNTIME_DIAGNOSTICS", "0") == "1"
)
