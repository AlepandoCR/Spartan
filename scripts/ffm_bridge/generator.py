"""
FFM Bridge Generator
--------------------
Main orchestrator that coordinates parsing, type resolution, and code generation.
"""

import os

from .config import BridgeConfig, DEFAULT_CONFIG
from .emitters import MethodHandleEmitter, SyncMethodEmitter
from .parser import ASTParser, NativeFunction
from .templates import JavaClassTemplate
from .types import TYPE_REGISTRY


class FFMBridgeGenerator:
    """
    Main generator class that orchestrates the FFM binding generation process.

    Usage:
        generator = FFMBridgeGenerator(config)
        generator.generate()
    """

    def __init__(self, config: BridgeConfig = DEFAULT_CONFIG):
        self.config = config
        self.parser = ASTParser(TYPE_REGISTRY)
        self.handle_emitter = MethodHandleEmitter()
        self.sync_emitter = SyncMethodEmitter()
        self.template = JavaClassTemplate(config)

    def generate(self) -> bool:
        """
        Execute the full generation pipeline.

        Returns:
            True if generation was successful, False otherwise.
        """
        try:
            # Parse C++ source
            functions = self.parser.parse(self.config.cpp_source)

            if not functions:
                print("[Warning] No exportable functions found.")
                return False

            # Generate code components
            handle_decls = self._generate_handle_declarations(functions)
            handle_inits = self._generate_handle_initializations(functions)
            sync_methods = self._generate_sync_methods(functions)

            # Render final class
            java_code = self.template.render(
                handle_declarations=handle_decls,
                handle_initializations=handle_inits,
                sync_methods=sync_methods
            )

            # Write output
            self._write_output(java_code)

            print(f"--- Binding Generation Successful: {self.config.native_class_file} ---")
            return True

        except Exception as e:
            print(f"[Fatal] Generation failed: {e}")
            raise

    def _generate_handle_declarations(self, functions: list[NativeFunction]) -> str:
        """Generate all MethodHandle field declarations."""
        return "\n".join(
            self.handle_emitter.emit_declaration(func)
            for func in functions
        )

    def _generate_handle_initializations(self, functions: list[NativeFunction]) -> str:
        """Generate all MethodHandle initializations for static block."""
        return "\n".join(
            self.handle_emitter.emit_initialization(func)
            for func in functions
        )

    def _generate_sync_methods(self, functions: list[NativeFunction]) -> str:
        """Generate all sync wrapper methods."""
        return "\n".join(
            self.sync_emitter.emit(func)
            for func in functions
        )

    def _write_output(self, content: str):
        """Write generated code to output file."""
        output_path = self.config.native_class_file
        os.makedirs(output_path.parent, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)



