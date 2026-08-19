#!/usr/bin/env python3
"""
Project Zip Creator Utility.
Creates clean 'project.zip' containing ONLY source code files (under 1 MB).
"""
import os
import zipfile

EXCLUDE_DIRS = {".venv", "__pycache__", ".git", ".pytest_cache", ".cache", "logs", "uploads"}
EXCLUDE_FILES = {"project.zip", "project_bundle.txt", ".DS_Store"}

def create_zip():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(root_dir, "project.zip")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
            for file in sorted(files):
                if file in EXCLUDE_FILES or file.startswith("."):
                    continue
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, root_dir)
                zipf.write(file_path, rel_path)

    print(f"✓ Project successfully zipped into '{zip_path}'")

if __name__ == "__main__":
    create_zip()
