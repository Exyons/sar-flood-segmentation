"""Convert notebooks/scripts/*.py → notebooks/*.ipynb using nbformat.

Cell markers:
    # %%              → code cell
    # %% [markdown]   → markdown cell (content lines stripped of leading '# ')

Usage:
    uv run python notebooks/convert_to_nb.py
"""

import glob
import pathlib

import nbformat


def parse_cells(py_path: str) -> list[dict]:
    """Parse a .py file with # %% markers into cells."""
    with open(py_path) as f:
        lines = f.readlines()

    cells = []
    current_type = None
    current_lines = []

    for line in lines:
        stripped = line.strip()

        if stripped == "# %%":
            # Flush previous cell
            if current_lines:
                cells.append({"type": current_type, "source": current_lines})
            current_type = "code"
            current_lines = []

        elif stripped == "# %% [markdown]":
            if current_lines:
                cells.append({"type": current_type, "source": current_lines})
            current_type = "markdown"
            current_lines = []

        else:
            if current_type is None:
                # Lines before first marker — treat as code
                current_type = "code"

            if current_type == "markdown":
                # Strip leading '# ' from markdown lines
                if line.startswith("# "):
                    current_lines.append(line[2:])
                elif stripped == "#":
                    current_lines.append("\n")
                else:
                    current_lines.append(line)
            else:
                current_lines.append(line)

    # Flush last cell
    if current_lines:
        cells.append({"type": current_type, "source": current_lines})

    return cells


def convert_file(py_path: str, out_dir: str) -> str:
    """Convert a single .py script to .ipynb."""
    cells_data = parse_cells(py_path)

    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    for cell_data in cells_data:
        source = "".join(cell_data["source"]).rstrip("\n")
        if not source.strip():
            continue

        if cell_data["type"] == "markdown":
            nb.cells.append(nbformat.v4.new_markdown_cell(source))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(source))

    stem = pathlib.Path(py_path).stem
    out_path = f"{out_dir}/{stem}.ipynb"
    with open(out_path, "w") as f:
        nbformat.write(nb, f)

    return out_path


def main():
    scripts = sorted(glob.glob("notebooks/scripts/*.py"))
    if not scripts:
        print("No scripts found in notebooks/scripts/")
        return

    print(f"Converting {len(scripts)} scripts...")
    for py_path in scripts:
        out_path = convert_file(py_path, "notebooks")
        print(f"  {py_path} → {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
