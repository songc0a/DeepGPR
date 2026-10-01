"""Device helpers that work with ``torch.device`` objects and plain strings."""

from __future__ import annotations

from typing import Any


def device_type(device: Any) -> str:
    """Return the lower-case backend name of a device.

    Args:
        device: ``torch.device`` or a string such as ``"cuda:1"``.

    Returns:
        ``"cpu"``, ``"cuda"``, ...
    """
    kind = getattr(device, "type", None)
    if kind is not None:
        return str(kind).lower()
    return str(device).split(":", 1)[0].lower()


__all__ = ["device_type"]
