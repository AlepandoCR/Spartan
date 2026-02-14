import os
from clang.cindex import Index, CursorKind, TypeKind

"""
Spartan FFM Bridge Generator
----------------------------
Automates the generation of Java 22+ Foreign Function & Memory (FFM) API bindings 
by introspecting C++26 source code using LibClang.

This script parses the C++ Abstract Syntax Tree (AST), maps native types to their 
Java FFM counterparts, and generates a type-safe 'SpartanNative' Java class 
capable of loading and invoking the compiled native library.
"""

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
CPP_SOURCE = "src/org/spartan/core/SpartanApi.cpp"
JAVA_OUTPUT = "../internal/src/main/java/org/spartan/core/bridge/SpartanNative.java"
JAVA_PACKAGE = "org.spartan.core.bridge"
DLL_NAME = "spartan_core"

# ==========================================
# FFM TYPE MAPPING DEFINITIONS
# ==========================================
# Defines the interoperability layer between C++ and Java FFM.
# Structure: (Java Type, FFM MemoryLayout, IsVoid?, NeedsMarshalling?, MarshallingStrategy)
TYPE_MAP = {
    TypeKind.VOID:    ("void", "null", True, False, None),
    TypeKind.BOOL:    ("boolean", "ValueLayout.JAVA_BOOLEAN", False, False, None),
    TypeKind.INT:     ("int", "ValueLayout.JAVA_INT", False, False, None),
    TypeKind.LONG:    ("long", "ValueLayout.JAVA_LONG", False, False, None),
    TypeKind.FLOAT:   ("float", "ValueLayout.JAVA_FLOAT", False, False, None),
    TypeKind.DOUBLE:  ("double", "ValueLayout.JAVA_DOUBLE", False, False, None),
    TypeKind.POINTER: ("MemorySegment", "ValueLayout.ADDRESS", False, False, None),
}


def get_java_info(clang_type):
    """
    Resolves the Java FFM type descriptor for a given Clang type.

    Performs inspection of pointer types to detect C-style strings (char*)
    vs. generic opaque pointers, assigning appropriate marshaling strategies.

    Args:
        clang_type (Cursor.type): The AST type node from LibClang.

    Returns:
        tuple: (JavaTypeStr, LayoutStr, IsVoid, NeedsMarshalling, MarshallingStrategy)
    """
    kind = clang_type.kind

    # Pointer Introspection: Differentiate between Strings and Opaque Pointers
    if kind == TypeKind.POINTER:
        pointee = clang_type.get_pointee()
        # Detect char*, const char*, unsigned char* -> Map to Java String
        if pointee.kind == TypeKind.CHAR_S or pointee.kind == TypeKind.SCHAR or pointee.kind == TypeKind.UCHAR:
            return ("String", "ValueLayout.ADDRESS", False, True, "string")
        # Default behavior for other pointers -> Map to MemorySegment
        return ("MemorySegment", "ValueLayout.ADDRESS", False, False, None)

    # Standard primitive mapping
    if kind in TYPE_MAP:
        return TYPE_MAP[kind]

    # Fallback for arrays or incomplete types -> Treat as generic address
    if kind == TypeKind.INCOMPLETEARRAY:
        return ("MemorySegment", "ValueLayout.ADDRESS", False, False, None)

    return ("MemorySegment", "ValueLayout.ADDRESS", False, False, None)


def to_camel_case(snake_str):
    """
    Utility to transform C++ naming conventions (snake_case) to Java conventions (camelCase).

    Example: 'spartan_run_inference' -> 'spartanRunInference'
    """
    parts = snake_str.split('_')
    return parts[0] + ''.join(word.capitalize() for word in parts[1:])


def find_exported_functions(cursor, source_path):
    """
    Traverses the Translation Unit (TU) to locate eligible function declarations.

    Filters the AST to strictly include functions defined within the target
    source file, ignoring system headers and included dependencies.

    Args:
        cursor (Cursor): The root cursor of the translation unit.
        source_path (str): The specific file path to analyze.

    Returns:
        list[Cursor]: A list of Clang cursors representing exported functions.
    """
    functions = []
    source_path_normalized = os.path.normcase(os.path.abspath(source_path))

    def visit(node):
        # Scope enforcement: Ensure node belongs to the target source file
        if node.location.file:
            node_file = os.path.normcase(os.path.abspath(node.location.file.name))
            if node_file != source_path_normalized:
                return  # Prune branch: Node belongs to an include/header

        if node.kind == CursorKind.FUNCTION_DECL:
            # Identification logic: Locate functions in the local compilation unit.
            # TODO: Once C++26 contracts are officially supported,
            #  enhance this filter to check for contracts like not null
            if node.location.file:
                node_file = os.path.normcase(os.path.abspath(node.location.file.name))
                if node_file == source_path_normalized:
                    functions.append(node)

        # Recursive descent
        for child in node.get_children():
            visit(child)

    visit(cursor)
    return functions


def generate_bindings():
    """
    Main Execution Routine.
    1. Parses C++ source using LibClang (C++26 Standard).
    2. Reflects over function signatures, return types, and arguments.
    3. Generates the 'SpartanNative' Java class with MethodHandles and Arenas.
    4. Writes the output to the specified Java source path.
    """
    if not os.path.exists(CPP_SOURCE):
        print(f"[Fatal] C++ source file missing: {CPP_SOURCE}")
        return

    print(f"--- Initializing AST Parsing for: {CPP_SOURCE} ---")

    index = Index.create()
    # Enforce C++26 standard to ensure AST compliance with modern contracts/attributes
    tu = index.parse(CPP_SOURCE, args=['-std=c++26'])

    # Buffers for code generation phases
    handle_declarations = []
    handle_initializations = []
    method_wrappers = []

    # AST Traversal: Extract exportable symbols
    exported_functions = find_exported_functions(tu.cursor, CPP_SOURCE)

    print(f"AST Analysis complete. Detected {len(exported_functions)} exportable functions.")

    for cursor in exported_functions:
        func_name = cursor.spelling
        java_method_name = to_camel_case(func_name)

        print(f" -> Generating binding: {func_name} :: {java_method_name}")

        # Introspect Return Type
        ret_info = get_java_info(cursor.result_type)
        ret_java, ret_layout, ret_is_void = ret_info[0], ret_info[1], ret_info[2]

        # Introspect Arguments
        arg_names = []
        arg_java_types = []
        arg_layouts = []
        arg_conversions = []  # Track arguments requiring marshaling (e.g., String -> Segment)

        for arg in cursor.get_arguments():
            arg_name = arg.spelling if arg.spelling else "arg" + str(len(arg_names))
            arg_info = get_java_info(arg.type)
            a_java, a_layout, _, needs_conv, conv_type = arg_info

            # Apply Nullability Annotations based on Type Semantics
            prefix = ""
            if a_java == "MemorySegment":
                prefix = "@NotNull "
            elif a_java == "String":
                prefix = "@NotNull "

            arg_names.append(arg_name)
            arg_java_types.append(f"{prefix}{a_java}")
            arg_layouts.append(a_layout)
            arg_conversions.append((needs_conv, conv_type, arg_name))

        # --- CODE GENERATION BLOCKS ---

        # 1. MethodHandle Definition (Static Field)
        handle_name = f"{func_name.upper()}_HANDLE"
        handle_declarations.append(f"    private static final MethodHandle {handle_name};")

        # 2. MethodHandle Initialization (Static Block)
        # Constructs the FFM FunctionDescriptor used by the Linker.

        args_layout_str = ", ".join(arg_layouts)
        if ret_is_void:
            desc_str = f"FunctionDescriptor.ofVoid({args_layout_str})" if args_layout_str else "FunctionDescriptor.ofVoid()"
        else:
            desc_str = f"FunctionDescriptor.of({ret_layout}, {args_layout_str})" if args_layout_str else f"FunctionDescriptor.of({ret_layout})"

        handle_initializations.append(f"""        try {{
            var addr = loader.find("{func_name}").orElseThrow(() -> new RuntimeException("Native Symbol resolution failed: {func_name}"));
            {handle_name} = linker.downcallHandle(addr, {desc_str});
        }} catch (Exception e) {{
            throw new RuntimeException("Failed to bind native function: {func_name}", e);
        }}""")

        # 3. Public Wrapper Implementation
        # Generates the high-level Java method that delegates to the MethodHandle.
        params_str = ", ".join([f"{t} {n}" for t, n in zip(arg_java_types, arg_names)])

        # Check for marshaling requirements
        has_string_args = any(conv[0] and conv[1] == "string" for conv in arg_conversions)

        # Prepare invocation arguments (substituting raw Strings for segments where needed)
        invoke_args = []
        for i, (needs_conv, conv_type, name) in enumerate(arg_conversions):
            if needs_conv and conv_type == "string":
                invoke_args.append(f"{name}Segment")
            else:
                invoke_args.append(name)
        args_call_str = ", ".join(invoke_args)

        # Handle Return semantics
        return_kw = "" if ret_is_void else "return "
        cast_str = "" if ret_is_void else f"({ret_java}) "

        if has_string_args:
            # Generate wrapper with Confined Arena for safe String allocation
            string_conversions = []
            for needs_conv, conv_type, name in arg_conversions:
                if needs_conv and conv_type == "string":
                    string_conversions.append(f"            var {name}Segment = arena.allocateFrom({name});")

            conversions_str = "\n".join(string_conversions)

            method_wrappers.append(f"""
    /**
     * Native binding for: {func_name}
     * Auto-generated wrapper dealing with String marshalling via Confined Arena.
     */
    public static {ret_java} {java_method_name}({params_str}) {{
        try (var arena = Arena.ofConfined()) {{
{conversions_str}
            {return_kw}{cast_str}{handle_name}.invokeExact({args_call_str});
        }} catch (Throwable t) {{
            throw new RuntimeException("Critical native failure in {func_name}", t);
        }}
    }}""")
        else:
            # Direct invocation for primitives and existing MemorySegments
            method_wrappers.append(f"""
    /**
     * Native binding for: {func_name}
     */
    public static {ret_java} {java_method_name}({params_str}) {{
        try {{
            {return_kw}{cast_str}{handle_name}.invokeExact({args_call_str});
        }} catch (Throwable t) {{
            throw new RuntimeException("Critical native failure in {func_name}", t);
        }}
    }}""")

    # --- FINAL ASSEMBLY & I/O ---
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
 * * Represents the low-level Foreign Function & Memory (FFM) bindings 
 * for the Spartan Core Engine (C++ 26).
 */
public class SpartanNative {{

    private static final String LIB_NAME = "{DLL_NAME}";
    private static final Linker linker = Linker.nativeLinker();
    private static final SymbolLookup loader;

    static {{
        loadNativeLibrary();
        // Utilize the system-specific loader lookup to resolve symbols
        loader = SymbolLookup.loaderLookup();

        // Initialize Native Method Handles
{chr(10).join(handle_initializations)}
    }}

    /**
     * Robust Native Library Loader.
     * Attempts to load the shared object from the classpath (JAR resources).
     * Falls back to System.loadLibrary if resource extraction fails.
     */
    private static void loadNativeLibrary() {{
        String osName = System.getProperty("os.name").toLowerCase();
        String libExtension;
        String libPrefix;
        
        // Detect OS Architecture for artifact resolution
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
                // Fallback strategy: Attempt to load from java.library.path
                System.loadLibrary(LIB_NAME);
                return;
            }}
            
            // Extract the native library to a temporary location for linking
            Path tempDir = Files.createTempDirectory("spartan-native");
            Path tempLib = tempDir.resolve(libFileName);
            Files.copy(is, tempLib, StandardCopyOption.REPLACE_EXISTING);
            tempLib.toFile().deleteOnExit();
            tempDir.toFile().deleteOnExit();
            
            System.load(tempLib.toAbsolutePath().toString());
        }} catch (IOException e) {{
            throw new RuntimeException("Fatal Error: Could not load native library: " + LIB_NAME, e);
        }}
    }}

    // --- FFM Method Handles ---
{chr(10).join(handle_declarations)}

    // --- Public API Bridges ---
{chr(10).join(method_wrappers)}
}}
"""

    # Commit generated code to disk
    os.makedirs(os.path.dirname(JAVA_OUTPUT), exist_ok=True)
    with open(JAVA_OUTPUT, "w", encoding='utf-8') as f:
        f.write(content)

    print(f"--- Binding Generation Successful: {JAVA_OUTPUT} ---")


if __name__ == "__main__":
    generate_bindings()