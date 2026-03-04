# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Plugin / Extension system — secure, signed extensions."""
from flyto_ai.extensions.base import ExtensionBase, ExtensionManifest
from flyto_ai.extensions.hooks import HookRegistry
from flyto_ai.extensions.loader import ExtensionLoader

__all__ = ["ExtensionBase", "ExtensionManifest", "HookRegistry", "ExtensionLoader"]
