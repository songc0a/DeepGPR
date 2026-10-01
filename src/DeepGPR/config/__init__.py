"""Configuration: physical constants, native ABI codes and API defaults.

Import the specific module for new code, e.g.
``from DeepGPR.config.constants import SPEED_OF_LIGHT``.
"""

from . import constants, defaults
from .constants import *  # noqa: F403  (re-export; names listed in constants.__all__)
from .defaults import *  # noqa: F403  (re-export; names listed in defaults.__all__)

__all__ = ["constants", "defaults", *constants.__all__, *defaults.__all__]
