import os
import lit.formats

# Name shown in test summaries
config.name = 'ocamlc2'

# Where tests live (source) and where lit should run (build)
config.test_source_root = os.path.dirname(__file__)
config.test_build_root = os.path.join(config.test_source_root, '..', 'build')

# File extensions to treat as tests
config.suffixes = ['.ml', '.mli']

# Use shell-like test format (interprets RUN: lines)
config.test_format = lit.formats.ShTest()

# Environment substitutions (so %clang and %FileCheck work)
llvm_tools_dir = os.environ.get('LLVM_TOOLS_DIR', '')
if llvm_tools_dir:
    config.substitutions.append(
        ('%FileCheck', os.path.join(llvm_tools_dir, 'FileCheck'))
    )
    config.substitutions.append(
        ('%clang',    os.path.join(llvm_tools_dir, 'clang'))
    )

# Pass through environment variables if needed
config.environment['PATH'] = os.environ.get('PATH', '')
config.environment['PATH'] += ':' + os.path.join(config.test_build_root, 'bin')

# Description of this configuration
config.description = """
Basic Lit+FileCheck config for ocamlc2.
"""
