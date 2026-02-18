"""
FFM Type System
---------------
Defines the type mapping layer between C++ and Java FFM, including:
- Primitive types
- Wrapper types for CompletableFuture compatibility
- Array/List handling (double pointers)
- String marshalling
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
from clang.cindex import TypeKind


class MarshalStrategy(Enum):
    """Defines how a type should be marshalled between Java and C++."""
    NONE = auto()
    STRING = auto()
    LIST = auto()
    ARRAY = auto()


@dataclass
class TypeDescriptor:
    """
    Complete type information for FFM binding generation.

    Attributes:
        java_type: The Java type string (e.g., "int", "String")
        java_wrapper: The boxed Java type for generics (e.g., "Integer", "String")
        ffm_layout: The FFM ValueLayout constant
        is_void: Whether this represents void return type
        marshal_strategy: How to convert between Java and native
        element_type: For collections, the inner type descriptor
    """
    java_type: str
    java_wrapper: str
    ffm_layout: str
    is_void: bool = False
    marshal_strategy: MarshalStrategy = MarshalStrategy.NONE
    element_type: Optional['TypeDescriptor'] = None
    needs_nullable: bool = False

    @property
    def is_collection(self) -> bool:
        return self.marshal_strategy in (MarshalStrategy.LIST, MarshalStrategy.ARRAY)

    @property
    def requires_marshalling(self) -> bool:
        return self.marshal_strategy != MarshalStrategy.NONE


class TypeMapper(ABC):
    """Abstract base for type mapping strategies."""

    @abstractmethod
    def can_map(self, clang_type) -> bool:
        """Check if this mapper handles the given Clang type."""
        pass

    @abstractmethod
    def map(self, clang_type) -> TypeDescriptor:
        """Convert Clang type to TypeDescriptor."""
        pass


class VoidTypeMapper(TypeMapper):
    """Maps C++ void to Java void."""

    def can_map(self, clang_type) -> bool:
        return clang_type.kind == TypeKind.VOID

    def map(self, clang_type) -> TypeDescriptor:
        return TypeDescriptor(
            java_type="void",
            java_wrapper="Void",
            ffm_layout="null",
            is_void=True
        )


class PrimitiveTypeMapper(TypeMapper):
    """Maps C++ primitives to Java primitives with wrapper support."""

    PRIMITIVE_MAP = {
        TypeKind.BOOL: ("boolean", "Boolean", "ValueLayout.JAVA_BOOLEAN"),
        TypeKind.CHAR_S: ("byte", "Byte", "ValueLayout.JAVA_BYTE"),
        TypeKind.CHAR_U: ("byte", "Byte", "ValueLayout.JAVA_BYTE"),
        TypeKind.UCHAR: ("byte", "Byte", "ValueLayout.JAVA_BYTE"),
        TypeKind.SCHAR: ("byte", "Byte", "ValueLayout.JAVA_BYTE"),
        TypeKind.SHORT: ("short", "Short", "ValueLayout.JAVA_SHORT"),
        TypeKind.USHORT: ("short", "Short", "ValueLayout.JAVA_SHORT"),
        TypeKind.INT: ("int", "Integer", "ValueLayout.JAVA_INT"),
        TypeKind.UINT: ("int", "Integer", "ValueLayout.JAVA_INT"),
        TypeKind.LONG: ("long", "Long", "ValueLayout.JAVA_LONG"),
        TypeKind.ULONG: ("long", "Long", "ValueLayout.JAVA_LONG"),
        TypeKind.LONGLONG: ("long", "Long", "ValueLayout.JAVA_LONG"),
        TypeKind.ULONGLONG: ("long", "Long", "ValueLayout.JAVA_LONG"),
        TypeKind.FLOAT: ("float", "Float", "ValueLayout.JAVA_FLOAT"),
        TypeKind.DOUBLE: ("double", "Double", "ValueLayout.JAVA_DOUBLE"),
    }

    def can_map(self, clang_type) -> bool:
        return clang_type.kind in self.PRIMITIVE_MAP

    def map(self, clang_type) -> TypeDescriptor:
        java_type, wrapper, layout = self.PRIMITIVE_MAP[clang_type.kind]
        return TypeDescriptor(
            java_type=java_type,
            java_wrapper=wrapper,
            ffm_layout=layout
        )


class StringTypeMapper(TypeMapper):
    """Maps C++ char* / const char* to Java String."""

    CHAR_KINDS = {TypeKind.CHAR_S, TypeKind.SCHAR, TypeKind.UCHAR, TypeKind.CHAR_U}

    def can_map(self, clang_type) -> bool:
        if clang_type.kind != TypeKind.POINTER:
            return False
        pointee = clang_type.get_pointee()
        return pointee.kind in self.CHAR_KINDS

    def map(self, clang_type) -> TypeDescriptor:
        return TypeDescriptor(
            java_type="String",
            java_wrapper="String",
            ffm_layout="ValueLayout.ADDRESS",
            marshal_strategy=MarshalStrategy.STRING,
            needs_nullable=True
        )


class ArrayTypeMapper(TypeMapper):
    """
    Maps C++ double pointers (T**) to Java List<T>.
    Also handles explicit array parameters with size.
    """

    def __init__(self, type_registry: 'TypeRegistry'):
        self.registry = type_registry

    def can_map(self, clang_type) -> bool:
        if clang_type.kind != TypeKind.POINTER:
            return False
        pointee = clang_type.get_pointee()
        # Double pointer detection (T**)
        if pointee.kind == TypeKind.POINTER:
            return True
        # Incomplete array type
        if clang_type.kind == TypeKind.INCOMPLETEARRAY:
            return True
        return False

    def map(self, clang_type) -> TypeDescriptor:
        pointee = clang_type.get_pointee()
        inner_pointee = pointee.get_pointee()

        # Resolve inner element type
        element_desc = self.registry.resolve(inner_pointee)

        return TypeDescriptor(
            java_type=f"List<{element_desc.java_wrapper}>",
            java_wrapper=f"List<{element_desc.java_wrapper}>",
            ffm_layout="ValueLayout.ADDRESS",
            marshal_strategy=MarshalStrategy.LIST,
            element_type=element_desc,
            needs_nullable=True
        )


class OpaquePointerMapper(TypeMapper):
    """Maps generic C++ pointers to Java MemorySegment."""

    def can_map(self, clang_type) -> bool:
        return clang_type.kind == TypeKind.POINTER

    def map(self, clang_type) -> TypeDescriptor:
        return TypeDescriptor(
            java_type="MemorySegment",
            java_wrapper="MemorySegment",
            ffm_layout="ValueLayout.ADDRESS",
            needs_nullable=True
        )


class TypeRegistry:
    """
    Central registry managing all type mappers.
    Uses chain of responsibility pattern for type resolution.
    """

    def __init__(self):
        self._mappers: list[TypeMapper] = []
        self._register_default_mappers()

    def _register_default_mappers(self):
        """Register mappers in priority order (first match wins)."""
        self._mappers.append(VoidTypeMapper())
        self._mappers.append(PrimitiveTypeMapper())
        self._mappers.append(StringTypeMapper())
        self._mappers.append(ArrayTypeMapper(self))
        self._mappers.append(OpaquePointerMapper())  # Fallback for other pointers

    def register(self, mapper: TypeMapper, priority: int = -1):
        """Register a custom type mapper with optional priority."""
        if priority < 0:
            self._mappers.append(mapper)
        else:
            self._mappers.insert(priority, mapper)

    def resolve(self, clang_type) -> TypeDescriptor:
        """Resolve a Clang type to its TypeDescriptor."""
        for mapper in self._mappers:
            if mapper.can_map(clang_type):
                return mapper.map(clang_type)

        # Ultimate fallback - treat as opaque pointer
        return TypeDescriptor(
            java_type="MemorySegment",
            java_wrapper="MemorySegment",
            ffm_layout="ValueLayout.ADDRESS",
            needs_nullable=True
        )


# Global type registry singleton
TYPE_REGISTRY = TypeRegistry()

