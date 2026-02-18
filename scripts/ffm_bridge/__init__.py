"""
Spartan FFM Bridge Generator
----------------------------
Modular package for generating Java 22+ Foreign Function & Memory (FFM) API bindings
from C++26 source code using LibClang AST introspection.
"""

from .generator import FFMBridgeGenerator

__all__ = ['FFMBridgeGenerator']

