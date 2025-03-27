import argparse
import sys
from typing import Optional, List
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)

class Driver:

    class TargetFormat(Enum):
        """Enum representing the available output formats for code generation."""
        ast = auto()
        mlir = auto()
        llvm = auto()
        obj = auto()
        executable = auto()

    class DumpFormat(Enum):
        """Enum representing the available dump formats for code generation."""
        ast = auto()
        mlir = auto()
        llvm = auto()

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # Main dump argument
        self.parser.add_argument('--dump', '-d', type=str, action='append', default=[], help='Dump intermediate representation at specified stage', choices=Driver.DumpFormat.__members__.keys())
        
        # Dump aliases
        self.parser.add_argument('--dump-all', action='store_const', const=['ast', 'mlir', 'llvm'], dest='dump', help='Dump all intermediate representations')
        self.parser.add_argument('--debug', action='store_true', dest='debug', help='Alias for --dump-all')
        
        # Target format with aliases
        self.parser.add_argument('--target-format', '-t', default='obj', type=str, help='Target format', choices=Driver.TargetFormat.__members__.keys())
        self.parser.add_argument('--emit-ast', action='store_const', const='ast', dest='target_format', help='Generate AST output')
        self.parser.add_argument('--emit-mlir', action='store_const', const='mlir', dest='target_format', help='Generate MLIR output')
        self.parser.add_argument('--emit-llvm', action='store_const', const='llvm', dest='target_format', help='Generate LLVM output')
        self.parser.add_argument('--emit-obj', action='store_const', const='obj', dest='target_format', help='Generate object file (default)')
        
        # Output file
        self.parser.add_argument('--output', '-o', default=sys.stdout, type=str, help='Output file')
        
        self.parser.add_argument('--log-level', type=str, default='INFO',
                               choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                               help='Set the logging level')
        self.parser.add_argument('source_file', type=str, help='OCaml source file')
        self.args: Optional[argparse.Namespace] = None
        self.dump: List[Driver.DumpFormat] = []


    def parse_args(self) -> argparse.Namespace:
        self.args = self.parser.parse_args()
        for dump in self.args.dump:
            self.dump.append(Driver.DumpFormat[dump])

        if self.args.debug:
            logger.debug(self.args)
            logger.debug(self.dump)
            self.args.log_level = 'DEBUG'
            self.args.dump = [Driver.DumpFormat.ast, Driver.DumpFormat.mlir, Driver.DumpFormat.llvm]
        
        return self.args

    def target_format(self) -> TargetFormat:
        assert self.args is not None, "Arguments not parsed"
        match self.args.target_format:
            case 'ast':
                return Driver.TargetFormat.ast
            case 'mlir':
                return Driver.TargetFormat.mlir
            case 'llvm':
                return Driver.TargetFormat.llvm
            case 'obj':
                return Driver.TargetFormat.obj
            case _:
                assert False, f"Invalid target format: {self.args.target_format}"

    def targets(self, target: TargetFormat) -> bool:
        return target.value <= self.target_format().value

    def dumps(self, dump: DumpFormat) -> bool:
        logger.debug(f"{dump}, {self.dump}, {dump in self.dump}")
        return dump in self.dump
