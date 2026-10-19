"""Forge torch.compile backend.

Usage:
    import forge.compile  # registers the backend
    model = torch.compile(model, backend="forge")
"""
from forge.compile.backend import forge_backend  # noqa: F401
