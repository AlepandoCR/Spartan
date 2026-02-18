#!/usr/bin/env python3
"""
Spartan FFM Bridge Generator - Entry Point
-------------------------------------------
Generates Java 22+ FFM bindings from C++26 source code.

Usage:
    python generate_ffm_spartan_bridge.py [--no-async] [--cpp-source PATH] [--output PATH]
"""

import argparse
import sys
from pathlib import Path

from ffm_bridge import FFMBridgeGenerator
from ffm_bridge.config import BridgeConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Java FFM bindings from C++ source"
    )
    parser.add_argument(
        "--cpp-source",
        type=Path,
        default=Path("src/org/spartan/api/SpartanApi.cpp"),
        help="Path to C++ source file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../internal/src/main/java/org/spartan/core/bridge"),
        help="Output directory for generated Java files"
    )
    parser.add_argument(
        "--no-async",
        action="store_true",
        help="Disable async method generation"
    )
    parser.add_argument(
        "--lib-name",
        type=str,
        default="spartan_core",
        help="Native library name (without extension)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = BridgeConfig(
        cpp_source=args.cpp_source,
        java_output=args.output,
        lib_name=args.lib_name,
        generate_async=not args.no_async
    )

    generator = FFMBridgeGenerator(config)

    success = generator.generate()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
