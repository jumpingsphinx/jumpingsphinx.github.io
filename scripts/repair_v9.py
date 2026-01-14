import nbformat
import os
import textwrap
import re

def load_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)

def save_notebook(nb, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

def dedent_cell(source):
    lines = source.splitlines()
    first_code_line = None
    for line in lines:
        if line.strip() and not line.strip().startswith('#'):
            first_code_line = line
            break
    if first_code_line:
        params = re.match(r"^(\s+)", first_code_line)
        if params:
            return textwrap.dedent(source)
    return source

def repair_v9(base_path):
    print("Executing Repair v9 (Helper Injection)...")

    # --- Mod 4 Ex 1: Inject Helpers ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise1-perceptron.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        helpers = textwrap.dedent("""
            import numpy as np
            import matplotlib.pyplot as plt
            def sigmoid(z): return 1 / (1 + np.exp(-z))
            def relu(z): return np.maximum(0, z)
            def tanh(z): return np.tanh(z)
        """).strip()
        
        # Inject at top (after markdown?)
        # Find first code cell
        for cell in nb.cells:
            if cell.cell_type == 'code':
                if "def tanh" not in cell.source:
                    cell.source = helpers + "\n\n" + cell.source
                    print("Injected helpers in Mod 4 Ex 1.")
                break # Only first cell
        save_notebook(nb, path)

    # --- Mod 4 Ex 2: Inject Helpers ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise2-feedforward-networks.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        helpers = textwrap.dedent("""
            import numpy as np
            def sigmoid(z): return 1 / (1 + np.exp(-z))
            def softmax(z):
                exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
                return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        """).strip()
        
        for cell in nb.cells:
            if cell.cell_type == 'code':
                if "def sigmoid" not in cell.source:
                    cell.source = helpers + "\n\n" + cell.source
                    print("Injected helpers in Mod 4 Ex 2.")
                break
        save_notebook(nb, path)

    # --- Mod 4 Ex 4: Fix 'one_hot' Indentation ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise4-numpy-implementation.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        for cell in nb.cells:
            if cell.cell_type == 'code':
                if "one_hot" in cell.source:
                    # check for indent
                    new_source = dedent_cell(cell.source)
                    if new_source != cell.source:
                        cell.source = new_source
                        print("Dedented one_hot cell in Mod 4 Ex 4.")
        save_notebook(nb, path)

if __name__ == "__main__":
    repair_v9("c:\\dev\\python\\ML101")
