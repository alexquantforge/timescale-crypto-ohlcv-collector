#!/usr/bin/env python3
"""
Project Bundler Utility for AI Agents / LLMs.
Combines all modular source code into a single 'project_bundle.txt' file for easy sharing.
"""
import os

EXCLUDE_DIRS = {".venv", "__pycache__", ".git", ".pytest_cache", "logs", "uploads"}
EXCLUDE_FILES = {"poetry.lock", "project_bundle.txt"}

def bundle():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(root_dir, "project_bundle.txt")

    with open(output_path, "w", encoding="utf-8") as out:
        out.write("=" * 80 + "\n")
        out.write("TIMESCALE CRYPTO OHLCV COLLECTOR — COMBINED CODEBASE BUNDLE\n")
        out.write("=" * 80 + "\n\n")

        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for file in sorted(files):
                if file in EXCLUDE_FILES or file.startswith("."):
                    continue
                if file.endswith((".py", ".toml", ".yml", ".md", ".json", ".sql", ".sh", "Dockerfile")):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, root_dir)
                    out.write(f"\n{'=' * 80}\n")
                    out.write(f"FILE: {rel_path}\n")
                    out.write(f"{'=' * 80}\n\n")
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            out.write(f.read())
                            out.write("\n")
                    except Exception as e:
                        out.write(f"Error reading file: {e}\n")

    print(f"✓ Project codebase successfully bundled into '{output_path}'")

if __name__ == "__main__":
    bundle()
