"""
AST Parser Module
-----------------
Handles C++ source parsing and function extraction using LibClang.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from clang.cindex import Index, CursorKind, Cursor

from .types import TypeRegistry, TypeDescriptor, TYPE_REGISTRY


@dataclass
class FunctionParameter:
    """Represents a C++ function parameter."""
    name: str
    type_desc: TypeDescriptor
    index: int

    @property
    def java_declaration(self) -> str:
        """Generate Java parameter declaration with annotations."""
        prefix = "@NotNull " if self.type_desc.needs_nullable else ""
        return f"{prefix}{self.type_desc.java_type} {self.name}"


@dataclass
class NativeFunction:
    """
    Represents an exported C++ function with full type information.
    """
    name: str
    return_type: TypeDescriptor
    parameters: list[FunctionParameter] = field(default_factory=list)
    documentation: Optional[str] = None

    @property
    def java_method_name(self) -> str:
        """Convert snake_case to camelCase."""
        parts = self.name.split('_')
        return parts[0] + ''.join(word.capitalize() for word in parts[1:])

    @property
    def handle_name(self) -> str:
        """Generate the MethodHandle constant name."""
        return f"{self.name.upper()}_HANDLE"

    @property
    def has_string_params(self) -> bool:
        """Check if any parameter requires string marshalling."""
        from .types import MarshalStrategy
        return any(p.type_desc.marshal_strategy == MarshalStrategy.STRING for p in self.parameters)

    @property
    def has_list_params(self) -> bool:
        """Check if any parameter requires list/array marshalling."""
        from .types import MarshalStrategy
        return any(p.type_desc.marshal_strategy in (MarshalStrategy.LIST, MarshalStrategy.ARRAY) for p in self.parameters)

    @property
    def params_declaration(self) -> str:
        """Generate Java method parameters string."""
        return ", ".join(p.java_declaration for p in self.parameters)

    @property
    def ffm_layouts(self) -> list[str]:
        """Get FFM layouts for all parameters."""
        return [p.type_desc.ffm_layout for p in self.parameters]


class ASTParser:
    """
    Parses C++ source files and extracts exported function declarations.
    """

    def __init__(self, type_registry: TypeRegistry = TYPE_REGISTRY):
        self.type_registry = type_registry
        self._index = Index.create()

    def parse(self, source_path: Path) -> list[NativeFunction]:
        """
        Parse a C++ source file and extract all exported functions.

        Args:
            source_path: Path to the C++ source file

        Returns:
            List of NativeFunction descriptors
        """
        if not source_path.exists():
            raise FileNotFoundError(f"C++ source file missing: {source_path}")

        print(f"--- Initializing AST Parsing for: {source_path} ---")

        # Parse with C++26 standard
        tu = self._index.parse(str(source_path), args=['-std=c++26'])

        functions = self._extract_functions(tu.cursor, source_path)

        print(f"AST Analysis complete. Detected {len(functions)} exportable functions.")

        return functions

    def _extract_functions(self, cursor: Cursor, source_path: Path) -> list[NativeFunction]:
        """Traverse AST and extract function declarations from the source file."""
        functions = []
        source_normalized = os.path.normcase(os.path.abspath(str(source_path)))

        def visit(node: Cursor):
            # Filter to only nodes in our source file
            if node.location.file:
                node_file = os.path.normcase(os.path.abspath(node.location.file.name))
                if node_file != source_normalized:
                    return

            if node.kind == CursorKind.FUNCTION_DECL:
                if node.location.file:
                    node_file = os.path.normcase(os.path.abspath(node.location.file.name))
                    if node_file == source_normalized:
                        func = self._parse_function(node)
                        if func:
                            functions.append(func)

            for child in node.get_children():
                visit(child)

        visit(cursor)
        return functions

    def _parse_function(self, cursor: Cursor) -> Optional[NativeFunction]:
        """Parse a function cursor into a NativeFunction descriptor."""
        func_name = cursor.spelling

        # Extract return type
        return_type = self.type_registry.resolve(cursor.result_type)

        # Extract parameters
        parameters = []
        for idx, arg in enumerate(cursor.get_arguments()):
            param_name = arg.spelling if arg.spelling else f"arg{idx}"
            param_type = self.type_registry.resolve(arg.type)
            parameters.append(FunctionParameter(
                name=param_name,
                type_desc=param_type,
                index=idx
            ))

        # Extract documentation if available
        doc = cursor.brief_comment if cursor.brief_comment else None

        print(f" -> Parsing function: {func_name}")

        return NativeFunction(
            name=func_name,
            return_type=return_type,
            parameters=parameters,
            documentation=doc
        )

