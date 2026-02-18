"""
Code Emitters Module
--------------------
Generates Java source code from parsed function descriptors.
Implements Template Method pattern for different output formats.
"""

from abc import ABC, abstractmethod
from typing import TextIO

from .parser import NativeFunction, FunctionParameter
from .types import MarshalStrategy


class CodeEmitter(ABC):
    """Abstract base for Java code generation."""

    INDENT = "    "

    @abstractmethod
    def emit(self, functions: list[NativeFunction], output: TextIO):
        """Generate code for the given functions."""
        pass

    def _i(self, level: int) -> str:
        return self.INDENT * level


class MethodHandleEmitter(CodeEmitter):
    """Generates MethodHandle declarations and initializations."""

    def emit_declaration(self, func: NativeFunction) -> str:
        """Generate static MethodHandle field declaration."""
        return self._i(1) + f"private static final MethodHandle {func.handle_name};"

    def emit_initialization(self, func: NativeFunction) -> str:
        """Generate MethodHandle initialization in static block."""
        layouts = func.ffm_layouts
        layouts_str = ", ".join(layouts)

        if func.return_type.is_void:
            desc = f"FunctionDescriptor.ofVoid({layouts_str})" if layouts_str else "FunctionDescriptor.ofVoid()"
        else:
            ret_layout = func.return_type.ffm_layout
            desc = f"FunctionDescriptor.of({ret_layout}, {layouts_str})" if layouts_str else f"FunctionDescriptor.of({ret_layout})"

        lines = [
            self._i(2) + "try {",
            self._i(3) + f'var addr = loader.find("{func.name}").orElseThrow(() -> new RuntimeException("Native symbol resolution failed: {func.name}"));',
            self._i(3) + f"{func.handle_name} = linker.downcallHandle(addr, {desc});",
            self._i(2) + "} catch (Exception e) {",
            self._i(3) + f'throw new RuntimeException("Failed to bind native function: {func.name}", e);',
            self._i(2) + "}"
        ]
        return "\n".join(lines)

    def emit(self, functions: list[NativeFunction], output: TextIO):
        pass


class SyncMethodEmitter(CodeEmitter):


    def emit(self, func: NativeFunction) -> str:
        """Generate a synchronous Java method wrapper."""
        ret_type = func.return_type.java_type
        method_name = func.java_method_name
        params = func.params_declaration

        body = self._build_method_body(func)
        doc = self._build_javadoc(func)

        lines = [
            "",
            doc,
            self._i(1) + f"public static {ret_type} {method_name}({params}) {{",
            body,
            self._i(1) + "}"
        ]
        return "\n".join(lines)

    def _build_javadoc(self, func: NativeFunction) -> str:
        """Generate JavaDoc for the method."""
        doc = func.documentation or f"Native binding for: {func.name}"
        return "\n".join([
            self._i(1) + "/**",
            self._i(1) + f" * {doc}",
            self._i(1) + " */"
        ])

    def _build_method_body(self, func: NativeFunction) -> str:
        """Build the method body with appropriate marshalling."""
        has_marshalling = func.has_string_params or func.has_list_params

        invoke_args = self._build_invoke_args(func)
        args_str = ", ".join(invoke_args)

        return_stmt = "" if func.return_type.is_void else "return "
        cast = "" if func.return_type.is_void else f"({func.return_type.java_type}) "

        if has_marshalling:
            return self._build_marshalled_body(func, invoke_args, return_stmt, cast)
        else:
            return self._build_direct_body(func, args_str, return_stmt, cast)

    def _build_invoke_args(self, func: NativeFunction) -> list[str]:
        """Build the list of arguments for MethodHandle invocation."""
        args = []
        for param in func.parameters:
            if param.type_desc.marshal_strategy == MarshalStrategy.STRING:
                args.append(f"{param.name}Segment")
            elif param.type_desc.marshal_strategy == MarshalStrategy.LIST:
                args.append(f"{param.name}Segment")
            else:
                args.append(param.name)
        return args

    def _build_marshalled_body(self, func: NativeFunction, invoke_args: list[str], return_stmt: str, cast: str) -> str:
        """Build method body with Arena-based marshalling."""
        conversions = []

        for param in func.parameters:
            if param.type_desc.marshal_strategy == MarshalStrategy.STRING:
                conversions.append(self._i(3) + f"var {param.name}Segment = arena.allocateFrom({param.name});")
            elif param.type_desc.marshal_strategy == MarshalStrategy.LIST:
                conversions.extend(self._build_list_marshalling(param))

        conversions_str = "\n".join(conversions)
        args_str = ", ".join(invoke_args)

        lines = [
            self._i(2) + "try (var arena = Arena.ofConfined()) {",
            conversions_str,
            self._i(3) + f"{return_stmt}{cast}{func.handle_name}.invokeExact({args_str});",
            self._i(2) + "} catch (Throwable t) {",
            self._i(3) + f'throw new RuntimeException("Native invocation failed: {func.name}", t);',
            self._i(2) + "}"
        ]
        return "\n".join(lines)

    def _build_list_marshalling(self, param: FunctionParameter) -> list[str]:
        """Generate list-to-native marshalling code."""
        elem_type = param.type_desc.element_type
        elem_layout = elem_type.ffm_layout if elem_type else "ValueLayout.ADDRESS"

        return [
            self._i(3) + f"var {param.name}Segment = arena.allocate({elem_layout}, {param.name}.size());",
            self._i(3) + f"for (int i = 0; i < {param.name}.size(); i++) {{",
            self._i(4) + f"{param.name}Segment.setAtIndex({elem_layout}, i, {param.name}.get(i));",
            self._i(3) + "}"
        ]

    def _build_direct_body(self, func: NativeFunction, args_str: str, return_stmt: str, cast: str) -> str:
        """Build direct invocation body without marshalling."""
        lines = [
            self._i(2) + "try {",
            self._i(3) + f"{return_stmt}{cast}{func.handle_name}.invokeExact({args_str});",
            self._i(2) + "} catch (Throwable t) {",
            self._i(3) + f'throw new RuntimeException("Native invocation failed: {func.name}", t);',
            self._i(2) + "}"
        ]
        return "\n".join(lines)


class AsyncMethodEmitter(CodeEmitter):
    """Generates CompletableFuture-based async methods with direct native invocation."""

    def emit(self, func: NativeFunction) -> str:
        """Generate an async Java method returning CompletableFuture."""
        wrapper_type = func.return_type.java_wrapper
        method_name = func.java_method_name
        params = func.params_declaration

        body = self._build_async_body(func)
        doc = self._build_javadoc(func)

        if func.return_type.is_void:
            lines = [
                "",
                doc,
                self._i(1) + f"public static @NotNull CompletableFuture<Void> {method_name}({params}) {{",
                self._i(2) + "return CompletableFuture.runAsync(() -> {",
                body,
                self._i(2) + "});",
                self._i(1) + "}"
            ]
        else:
            lines = [
                "",
                doc,
                self._i(1) + f"public static @NotNull CompletableFuture<{wrapper_type}> {method_name}({params}) {{",
                self._i(2) + "return CompletableFuture.supplyAsync(() -> {",
                body,
                self._i(2) + "});",
                self._i(1) + "}"
            ]
        return "\n".join(lines)

    def _build_javadoc(self, func: NativeFunction) -> str:
        """Generate JavaDoc for async method."""
        doc = func.documentation or f"Native binding for: {func.name}"
        return "\n".join([
            self._i(1) + "/**",
            self._i(1) + f" * {doc}",
            self._i(1) + " * @return CompletableFuture completing with the native result",
            self._i(1) + " */"
        ])

    def _build_async_body(self, func: NativeFunction) -> str:
        """Build the async method body with native invocation."""
        has_marshalling = func.has_string_params or func.has_list_params

        invoke_args = self._build_invoke_args(func)
        args_str = ", ".join(invoke_args)

        return_stmt = "" if func.return_type.is_void else "return "
        cast = "" if func.return_type.is_void else f"({func.return_type.java_type}) "

        if has_marshalling:
            return self._build_marshalled_body(func, invoke_args, return_stmt, cast)
        else:
            return self._build_direct_body(func, args_str, return_stmt, cast)

    def _build_invoke_args(self, func: NativeFunction) -> list[str]:
        """Build the list of arguments for MethodHandle invocation."""
        args = []
        for param in func.parameters:
            if param.type_desc.marshal_strategy == MarshalStrategy.STRING:
                args.append(f"{param.name}Segment")
            elif param.type_desc.marshal_strategy == MarshalStrategy.LIST:
                args.append(f"{param.name}Segment")
            else:
                args.append(param.name)
        return args

    def _build_marshalled_body(self, func: NativeFunction, invoke_args: list[str], return_stmt: str, cast: str) -> str:
        """Build method body with Arena-based marshalling."""
        conversions = []

        for param in func.parameters:
            if param.type_desc.marshal_strategy == MarshalStrategy.STRING:
                conversions.append(self._i(4) + f"var {param.name}Segment = arena.allocateFrom({param.name});")
            elif param.type_desc.marshal_strategy == MarshalStrategy.LIST:
                conversions.extend(self._build_list_marshalling(param))

        conversions_str = "\n".join(conversions)
        args_str = ", ".join(invoke_args)

        lines = [
            self._i(3) + "try (var arena = Arena.ofConfined()) {",
            conversions_str,
            self._i(4) + f"{return_stmt}{cast}{func.handle_name}.invokeExact({args_str});",
            self._i(3) + "} catch (Throwable t) {",
            self._i(4) + f'throw new RuntimeException("Native invocation failed: {func.name}", t);',
            self._i(3) + "}"
        ]
        return "\n".join(lines)

    def _build_list_marshalling(self, param: FunctionParameter) -> list[str]:
        """Generate list-to-native marshalling code."""
        elem_type = param.type_desc.element_type
        elem_layout = elem_type.ffm_layout if elem_type else "ValueLayout.ADDRESS"

        return [
            self._i(4) + f"var {param.name}Segment = arena.allocate({elem_layout}, {param.name}.size());",
            self._i(4) + f"for (int i = 0; i < {param.name}.size(); i++) {{",
            self._i(5) + f"{param.name}Segment.setAtIndex({elem_layout}, i, {param.name}.get(i));",
            self._i(4) + "}"
        ]

    def _build_direct_body(self, func: NativeFunction, args_str: str, return_stmt: str, cast: str) -> str:
        """Build direct invocation body without marshalling."""
        lines = [
            self._i(3) + "try {",
            self._i(4) + f"{return_stmt}{cast}{func.handle_name}.invokeExact({args_str});",
            self._i(3) + "} catch (Throwable t) {",
            self._i(4) + f'throw new RuntimeException("Native invocation failed: {func.name}", t);',
            self._i(3) + "}"
        ]
        return "\n".join(lines)


