import os
from clang.cindex import Index, CursorKind, TypeKind

# --- CONFIGURATION ---
CPP_SOURCE = "src/org/spartan/core/SpartanApi.cpp"
JAVA_OUTPUT = "../internal/src/main/java/org/spartan/core/bridge/SpartanNative.java"
JAVA_PACKAGE = "org.spartan.core.bridge"
DLL_NAME = "spartan_core"

# --- TYPE MAPPING ---
# Maps C++ types to: (Java Type, FFM Layout, IsVoid?, NeedsConversion?, ConversionType)
# ConversionType: None, "string", "array"
TYPE_MAP = {
    TypeKind.VOID: ("void", "null", True, False, None),
    TypeKind.BOOL: ("boolean", "ValueLayout.JAVA_BOOLEAN", False, False, None),
    TypeKind.INT: ("int", "ValueLayout.JAVA_INT", False, False, None),
    TypeKind.LONG: ("long", "ValueLayout.JAVA_LONG", False, False, None),
    TypeKind.FLOAT: ("float", "ValueLayout.JAVA_FLOAT", False, False, None),
    TypeKind.DOUBLE: ("double", "ValueLayout.JAVA_DOUBLE", False, False, None),
    TypeKind.POINTER: ("MemorySegment", "ValueLayout.ADDRESS", False, False, None),
}


def get_java_info(clang_type):
    """Returns a tuple: (JavaTypeString, FFMLayoutString, IsVoid, NeedsConversion, ConversionType)"""
    kind = clang_type.kind

    # Check for char pointer (const char* or char*) -> treat as String
    if kind == TypeKind.POINTER:
        pointee = clang_type.get_pointee()
        # Check if it's a char type (C string)
        if pointee.kind == TypeKind.CHAR_S or pointee.kind == TypeKind.SCHAR or pointee.kind == TypeKind.UCHAR:
            return ("String", "ValueLayout.ADDRESS", False, True, "string")
        return ("MemorySegment", "ValueLayout.ADDRESS", False, False, None)

    if kind in TYPE_MAP:
        return TYPE_MAP[kind]

    # Default fallback for pointers/arrays
    if kind == TypeKind.INCOMPLETEARRAY:
        return ("MemorySegment", "ValueLayout.ADDRESS", False, False, None)

    return ("MemorySegment", "ValueLayout.ADDRESS", False, False, None)


def to_camel_case(snake_str):
    """Convert snake_case to camelCase (e.g., spartan_log -> spartanLog)"""
    parts = snake_str.split('_')
    return parts[0] + ''.join(word.capitalize() for word in parts[1:])


def find_exported_functions(cursor, source_path):
    """Find all function declarations inside extern 'C' blocks from our source file"""
    functions = []
    source_path_normalized = os.path.normcase(os.path.abspath(source_path))

    def visit(node):
        # Check if node is from our file
        if node.location.file:
            node_file = os.path.normcase(os.path.abspath(node.location.file.name))
            if node_file != source_path_normalized:
                return  # Skip nodes from other files

        if node.kind == CursorKind.FUNCTION_DECL:
            # Check if it has dllexport or is in extern "C" context
            # For simplicity, we check if function name doesn't start with underscore (system funcs)
            if node.location.file:
                node_file = os.path.normcase(os.path.abspath(node.location.file.name))
                if node_file == source_path_normalized:
                    functions.append(node)

        for child in node.get_children():
            visit(child)

    visit(cursor)
    return functions


def generate_bindings():
    if not os.path.exists(CPP_SOURCE):
        print(f"[Error] C++ source not found: {CPP_SOURCE}")
        return

    print(f"--- Parsing {CPP_SOURCE} with LibClang ---")

    index = Index.create()
    # Parse with C++26 standard to ensure correct AST interpretation
    tu = index.parse(CPP_SOURCE, args=['-std=c++26'])

    # Lists to hold generated code parts
    handle_declarations = []
    handle_initializations = []
    method_wrappers = []

    # Find all exported functions from our source file
    exported_functions = find_exported_functions(tu.cursor, CPP_SOURCE)

    print(f"Found {len(exported_functions)} exported functions")

    for cursor in exported_functions:
        func_name = cursor.spelling
        java_method_name = to_camel_case(func_name)

        print(f" -> Found export: {func_name} -> {java_method_name}")

        # Analyze Return Type
        ret_info = get_java_info(cursor.result_type)
        ret_java, ret_layout, ret_is_void = ret_info[0], ret_info[1], ret_info[2]

        # Analyze Arguments
        arg_names = []
        arg_java_types = []
        arg_layouts = []
        arg_conversions = []  # Track which args need conversion

        for arg in cursor.get_arguments():
            arg_name = arg.spelling if arg.spelling else "arg" + str(len(arg_names))
            arg_info = get_java_info(arg.type)
            a_java, a_layout, _, needs_conv, conv_type = arg_info

            # Logic for Contracts / Annotations
            prefix = ""
            if a_java == "MemorySegment":
                prefix = "@NotNull "
            elif a_java == "String":
                prefix = "@NotNull "

            arg_names.append(arg_name)
            arg_java_types.append(f"{prefix}{a_java}")
            arg_layouts.append(a_layout)
            arg_conversions.append((needs_conv, conv_type, arg_name))

        # --- GENERATE CODE BLOCKS ---

        # 1. MethodHandle Declaration
        handle_name = f"{func_name.upper()}_HANDLE"
        handle_declarations.append(f"    private static final MethodHandle {handle_name};")

        # 2. MethodHandle Initialization (Inside static block logic)
        # descriptor: FunctionDescriptor.of(RETURN_LAYOUT, ARG1_LAYOUT, ...)
        # or FunctionDescriptor.ofVoid(ARG1_LAYOUT, ...)

        args_layout_str = ", ".join(arg_layouts)
        if ret_is_void:
            desc_str = f"FunctionDescriptor.ofVoid({args_layout_str})" if args_layout_str else "FunctionDescriptor.ofVoid()"
        else:
            desc_str = f"FunctionDescriptor.of({ret_layout}, {args_layout_str})" if args_layout_str else f"FunctionDescriptor.of({ret_layout})"

        handle_initializations.append(f"""        try {{
            var addr = loader.find("{func_name}").orElseThrow(() -> new RuntimeException("Symbol not found: {func_name}"));
            {handle_name} = linker.downcallHandle(addr, {desc_str});
        }} catch (Exception e) {{
            throw new RuntimeException("Failed to bind {func_name}", e);
        }}""")

        # 3. Public Wrapper Method
        # Signatures
        params_str = ", ".join([f"{t} {n}" for t, n in zip(arg_java_types, arg_names)])

        # Build the method body with conversions
        has_string_args = any(conv[0] and conv[1] == "string" for conv in arg_conversions)

        # Build invoke args (convert strings to MemorySegment)
        invoke_args = []
        for i, (needs_conv, conv_type, name) in enumerate(arg_conversions):
            if needs_conv and conv_type == "string":
                invoke_args.append(f"{name}Segment")
            else:
                invoke_args.append(name)
        args_call_str = ", ".join(invoke_args)

        # Return handling
        return_kw = "" if ret_is_void else "return "
        cast_str = "" if ret_is_void else f"({ret_java}) "

        if has_string_args:
            # Generate method with Arena for string conversion
            string_conversions = []
            for needs_conv, conv_type, name in arg_conversions:
                if needs_conv and conv_type == "string":
                    string_conversions.append(f"            var {name}Segment = arena.allocateFrom({name});")

            conversions_str = "\n".join(string_conversions)

            method_wrappers.append(f"""
    /**
     * Binding for C++ function: {func_name}
     */
    public static {ret_java} {java_method_name}({params_str}) {{
        try (var arena = Arena.ofConfined()) {{
{conversions_str}
            {return_kw}{cast_str}{handle_name}.invokeExact({args_call_str});
        }} catch (Throwable t) {{
            throw new RuntimeException("Native error in {func_name}", t);
        }}
    }}""")
        else:
            method_wrappers.append(f"""
    /**
     * Binding for C++ function: {func_name}
     */
    public static {ret_java} {java_method_name}({params_str}) {{
        try {{
            {return_kw}{cast_str}{handle_name}.invokeExact({args_call_str});
        }} catch (Throwable t) {{
            throw new RuntimeException("Native error in {func_name}", t);
        }}
    }}""")

    # --- ASSEMBLE FILE ---
    content = f"""package {JAVA_PACKAGE};

import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import org.jetbrains.annotations.NotNull;

/**
 * AUTOMATICALLY GENERATED - DO NOT EDIT.
 * Generated by scripts/generate_ffm_spartan_bridge.py
 * Represents the FFM bindings for Spartan Core (C++ 26).
 */
public class SpartanNative {{

    private static final String LIB_NAME = "{DLL_NAME}";
    private static final Linker linker = Linker.nativeLinker();
    private static final SymbolLookup loader;

    static {{
        loadNativeLibrary();
        loader = SymbolLookup.loaderLookup();

        // Initialize Handles
{chr(10).join(handle_initializations)}
    }}

    /**
     * Extracts and loads the native library from JAR resources.
     */
    private static void loadNativeLibrary() {{
        String osName = System.getProperty("os.name").toLowerCase();
        String libExtension;
        String libPrefix;
        
        if (osName.contains("win")) {{
            libExtension = ".dll";
            libPrefix = "";
        }} else if (osName.contains("mac")) {{
            libExtension = ".dylib";
            libPrefix = "lib";
        }} else {{
            libExtension = ".so";
            libPrefix = "lib";
        }}
        
        String libFileName = libPrefix + LIB_NAME + libExtension;
        String resourcePath = "/native/" + libFileName;
        
        try (InputStream is = SpartanNative.class.getResourceAsStream(resourcePath)) {{
            if (is == null) {{
                // Fallback: try loading from system path
                System.loadLibrary(LIB_NAME);
                return;
            }}
            
            Path tempDir = Files.createTempDirectory("spartan-native");
            Path tempLib = tempDir.resolve(libFileName);
            Files.copy(is, tempLib, StandardCopyOption.REPLACE_EXISTING);
            tempLib.toFile().deleteOnExit();
            tempDir.toFile().deleteOnExit();
            
            System.load(tempLib.toAbsolutePath().toString());
        }} catch (IOException e) {{
            throw new RuntimeException("Failed to load native library: " + LIB_NAME, e);
        }}
    }}

    // --- Method Handles ---
{chr(10).join(handle_declarations)}

    // --- Public API Wrappers ---
{chr(10).join(method_wrappers)}
}}
"""

    # Write to file
    os.makedirs(os.path.dirname(JAVA_OUTPUT), exist_ok=True)
    with open(JAVA_OUTPUT, "w", encoding='utf-8') as f:
        f.write(content)

    print(f"--- Generated Java Binding: {JAVA_OUTPUT} ---")


if __name__ == "__main__":
    generate_bindings()