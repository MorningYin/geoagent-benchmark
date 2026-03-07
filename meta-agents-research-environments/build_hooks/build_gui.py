#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


"""Cross-platform GUI build script for the hatch build process."""

import os
import platform
import subprocess
import sys
from pathlib import Path


def run_npm_windows(cmd_args, cwd):
    """Run npm command on Windows."""
    # On Windows, try npm.cmd first (the proper Windows command)
    npm_commands = ["npm.cmd", "npm"]

    for npm_cmd in npm_commands:
        try:
            # Test if this npm command works
            subprocess.run(
                [npm_cmd, "--version"], capture_output=True, check=True, shell=True
            )

            # If it works, use it for the actual command
            return subprocess.run(
                [npm_cmd] + cmd_args,
                cwd=cwd,
                check=True,
                capture_output=False,
                shell=True,  # Required on Windows to find npm in PATH
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    # If we get here, none of the npm commands worked
    raise FileNotFoundError("npm command not found")


def run_npm_unix(cmd_args, cwd):
    """Run npm command on Unix-like systems (Linux, macOS)."""
    return subprocess.run(
        ["npm"] + cmd_args,
        cwd=cwd,
        check=True,
        capture_output=False,
        # No shell=True needed on Unix systems
    )


def main():
    """Build the GUI frontend if BUILD_GUI environment variable is set."""
    build_gui = os.environ.get("BUILD_GUI", "").strip().lower()

    # Only consider "1" or "true" as positive values
    if build_gui not in {"1", "true"}:
        return 0

    # Change to the GUI client directory (go up one level from build_hooks to project root)
    project_root = Path(__file__).parent.parent
    client_dir = project_root / "are" / "simulation" / "gui" / "client"

    if not client_dir.exists():
        print(f"Error: GUI client directory not found: {client_dir}")
        return 1

    try:
        # Branch based on operating system
        if platform.system() == "Windows":
            # Windows-specific npm execution
            run_npm_windows(["install"], client_dir)
            run_npm_windows(["run", "build"], client_dir)
        else:
            # Unix-like systems (Linux, macOS, etc.)
            run_npm_unix(["install"], client_dir)
            run_npm_unix(["run", "build"], client_dir)

        return 0

    except subprocess.CalledProcessError as e:
        print(f"Error during npm build: {e}")
        return 1
    except FileNotFoundError as e:
        print(
            f"Error: npm not found. Please ensure Node.js is installed and npm is in your PATH: {e}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
