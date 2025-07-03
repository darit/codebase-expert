#!/usr/bin/env python3
"""
Backward-compatibility shim for codebase_expert.py

This script allows existing users to continue using `python codebase_expert.py ...`
while transitioning to the new `codebase-expert` command.
"""

import sys
import warnings
import pathlib


def _bootstrap_and_run():
    """
    A backward-compatibility shim to execute the new CLI.
    
    This allows users to continue running `python codebase_expert.py ...`
    during the transition period.
    """
    warnings.warn(
        "Running via 'python codebase_expert.py' is deprecated and will be removed "
        "in a future version. Please install the package (`pip install .` or "
        "`uvx install .`) and use the 'codebase-expert' command.",
        DeprecationWarning,
        stacklevel=2
    )

    # Add the 'src' directory to the path to find the new package
    # This is the magic that makes the import work without installation.
    src_dir = pathlib.Path(__file__).parent / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))

    try:
        from codebase_expert.cli import app
    except ImportError as e:
        print("=" * 80, file=sys.stderr)
        print("ERROR: Could not import the codebase_expert package.", file=sys.stderr)
        print("This script is now a compatibility shim.", file=sys.stderr)
        print("", file=sys.stderr)
        print("To fix this issue:", file=sys.stderr)
        print("1. Install in editable mode for development:", file=sys.stderr)
        print("   pip install -e .", file=sys.stderr)
        print("", file=sys.stderr)
        print("2. Or install normally:", file=sys.stderr)
        print("   pip install .", file=sys.stderr)
        print("", file=sys.stderr)
        print("3. Or use uvx:", file=sys.stderr)
        print("   uvx install .", file=sys.stderr)
        print("", file=sys.stderr)
        print(f"Import error details: {e}", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        sys.exit(1)

    # Run the Typer app with the command line arguments
    # Typer will handle parsing and routing to the correct command
    try:
        app()
    except Exception as e:
        print(f"Error running command: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _bootstrap_and_run()