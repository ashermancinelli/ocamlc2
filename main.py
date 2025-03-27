#!/usr/bin/env python3
import logging

from driver import Driver
from fe import Parser, Tree
from lower import Lower


def setup_logging(level: str = "INFO"):
    """Configure logging with the specified level."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main():
    driver = Driver()
    args = driver.parse_args()

    setup_logging(args.log_level if hasattr(args, "log_level") else "INFO")

    parser = Parser()
    tree: Tree = parser.parse(open(args.source_file, "r").read())

    if driver.dumps(Driver.DumpFormat.ast):
        parser.pprint(tree)

    lower = Lower(tree)
    module = lower.lower()

    if driver.dumps(Driver.DumpFormat.mlir):
        print(module)

    if driver.targets(Driver.TargetFormat.mlir):
        print(module, file=args.output)
        return

if __name__ == "__main__":
    main()
