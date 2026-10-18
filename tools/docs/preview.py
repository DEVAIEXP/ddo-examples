"""Expose only public documentation, preserving repository-relative links.

Virtual MkDocs files retain the repository layout without copying source files,
scanning the repository root or exposing environments, outputs and local records.
"""
from pathlib import Path

from mkdocs.structure.files import File, Files


def on_files(files, config):
    root = Path(config.config_file_path).resolve().parent
    public = Files([])
    for file in files:
        if Path(file.src_dir).resolve() != (root / "docs").resolve():
            public.append(file)
            continue
        source = Path(file.abs_src_path).resolve(strict=True)
        if not source.is_relative_to(root / "docs"):
            raise ValueError(f"Documentation source escapes docs/: {source}")
        public.append(File.generated(config, f"docs/{file.src_uri}", abs_src_path=str(source)))
    for relative in ("README.md", "AGENTS.md"):
        source = (root / relative).resolve(strict=True)
        if not source.is_relative_to(root):
            raise ValueError(f"Public documentation source escapes repository: {relative}")
        public.append(File.generated(config, relative, abs_src_path=str(source)))
    return public
