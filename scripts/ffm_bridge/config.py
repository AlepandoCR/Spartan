"""
FFM Bridge Configuration
------------------------
Central configuration for paths, naming conventions, and generation options.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BridgeConfig:
    """Configuration container for the FFM bridge generator."""

    # Source paths
    cpp_source: Path = field(default_factory=lambda: Path("src/org/spartan/api/SpartanApi.cpp"))

    # Output paths
    java_output: Path = field(default_factory=lambda: Path("../internal/src/main/java/org/spartan/internal/bridge"))

    # Java package configuration
    java_package: str = "org.spartan.internal.bridge"

    # Native library name (without prefix/extension)
    lib_name: str = "spartan_core"

    # Generation options (async only - safety is not optional)
    generate_async: bool = True

    # Class names
    native_class_name: str = "SpartanNative"
    async_class_name: str = "SpartanNativeAsync"

    @property
    def native_class_file(self) -> Path:
        return self.java_output / f"{self.native_class_name}.java"

    @property
    def async_class_file(self) -> Path:
        return self.java_output / f"{self.async_class_name}.java"


# Default configuration instance
DEFAULT_CONFIG = BridgeConfig()


