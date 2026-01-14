import nbformat
import os
import textwrap

def load_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)

def save_notebook(nb, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

def repair_v13(base_path):
    print("Executing Repair v13 (Import Restoration)...")

    # --- Mod 4 Ex 2: Restore Imports ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise2-feedforward-networks.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        
        # Imports to restore
        imports = textwrap.dedent("""
            import numpy as np
            import matplotlib.pyplot as plt
            from sklearn.datasets import make_moons, make_circles, make_classification
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.neural_network import MLPClassifier
            from sklearn.metrics import accuracy_score
        """).strip()
        
        for cell in nb.cells:
            # Detect the cell we modified (contains softmax_activation)
            if "def softmax_activation" in cell.source and "import matplotlib" not in cell.source:
                cell.source = imports + "\n\n" + cell.source
                print("Restored imports in Mod 4 Ex 2.")
        save_notebook(nb, path)

if __name__ == "__main__":
    repair_v13("c:\\dev\\python\\ML101")
