import nbformat
import os
import textwrap

def load_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)

def save_notebook(nb, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

def repair_v17(base_path):
    print("Executing Repair v17 (Mod 4 Ex 6 Fix)...")

    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise6-pytorch-advanced.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        
        # 1. Remove Legacy Training Loop (outputs = model_digits(X_train))
        # This is likely copy-pasted in Ex 6 as well
        # We need to find the cells that define model_digits and the loop
        
        new_cells = []
        for cell in nb.cells:
             # Detection logic similar to v16
             if "model_digits = nn.Sequential" in cell.source or "outputs = model_digits(X_train)" in cell.source:
                 # Check if this cell is indeed the legacy one.
                 # Ex 6 also has legitimate model definitions later?
                 # Let's be specific: if it uses 64 inputs and 32 hidden, AND loop
                 if "nn.Linear(64, 32)" in cell.source:
                     print("Removing legacy model/loop in Mod 4 Ex 6.")
                     continue
             new_cells.append(cell)
             
        nb.cells = new_cells
        save_notebook(nb, path)

if __name__ == "__main__":
    repair_v17("c:\\dev\\python\\ML101")
