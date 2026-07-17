from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def execute_notebook(path: Path) -> None:
    """Execute code cells with a shared Python namespace."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    previous_directory = Path.cwd()
    os.chdir(path.parent)
    try:
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            print(f"\n--- {path.name}: code cell {index} ---")
            exec(compile(source, f"{path.name}:cell-{index}", "exec"), namespace)
    finally:
        os.chdir(previous_directory)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute one verification notebook.")
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()
    path = args.notebook.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    execute_notebook(path)


if __name__ == "__main__":
    main()

