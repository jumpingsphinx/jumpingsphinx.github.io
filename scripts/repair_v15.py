import nbformat
import os
import textwrap

def load_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)

def save_notebook(nb, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

def repair_v15(base_path):
    print("Executing Repair v15 (Variable Alias)...")

    # --- Mod 4 Ex 5: Alias X_train to X_xor or appropriate ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise5-pytorch-basics.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        
        # Inject definition
        alias_code = textwrap.dedent("""
            # Alias X_train/y_train to X_xor/y_xor for compatibility
            X_train = X_xor
            y_train = y_xor
        """).strip()
        
        for cell in nb.cells:
            # Find XOR setup cell (we injected it in v12, contains X_xor definition)
            if "X_xor =" in cell.source and "X_train" not in cell.source:
                cell.source = cell.source + "\n" + alias_code
                print("Injected X_train alias in Mod 4 Ex 5.")
                break # Only need once
        save_notebook(nb, path)

if __name__ == "__main__":
    repair_v15("c:\\dev\\python\\ML101")
