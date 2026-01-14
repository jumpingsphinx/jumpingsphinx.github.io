import nbformat
import os

def load_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)

def save_notebook(nb, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

def repair_v10(base_path):
    print("Executing Repair v10 (Line Stripper)...")

    # --- Mod 4 Ex 2: Fix NeuralNetwork Init Defaults ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise2-feedforward-networks.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        # Update init to have defaults
        for cell in nb.cells:
             if "def __init__(self, n_input, n_hidden, n_output, hidden_activation=sigmoid):" in cell.source:
                 cell.source = cell.source.replace(
                     "def __init__(self, n_input, n_hidden, n_output, hidden_activation=sigmoid):",
                     "def __init__(self, n_input=2, n_hidden=3, n_output=1, hidden_activation=sigmoid):"
                 )
                 print("Added defaults to NeuralNetwork in Mod 4 Ex 2.")
        save_notebook(nb, path)

    # --- Mod 4 Ex 3: Force strip backprop lines ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise3-backpropagation.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        for cell in nb.cells:
            if cell.cell_type == 'code':
                lines = cell.source.splitlines()
                new_lines = []
                modified_cell = False
                for line in lines:
                    stripped = line.lstrip()
                    # Target specific lines causing issues
                    if stripped.startswith("dz_dw =") or stripped.startswith("dz_db =") or stripped.startswith("dz_dx ="):
                         new_lines.append(stripped)
                         modified_cell = True
                    elif stripped.startswith("print("): # De-indent prints too just in case
                         new_lines.append(stripped)
                         modified_cell = True
                    else:
                         new_lines.append(line)
                
                if modified_cell:
                    cell.source = "\n".join(new_lines)
                    print("Stripped indentation in Mod 4 Ex 3.")
        save_notebook(nb, path)

    # --- Mod 4 Ex 4: Force strip one_hot lines ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise4-numpy-implementation.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        for cell in nb.cells:
            if cell.cell_type == 'code':
                lines = cell.source.splitlines()
                new_lines = []
                modified_cell = False
                for line in lines:
                    stripped = line.lstrip()
                    if stripped.startswith("_onehot =") or stripped.startswith("plt.show()"):
                         new_lines.append(stripped)
                         modified_cell = True
                    elif stripped.startswith("model = NeuralNetwork()"): # Ensure top level
                         new_lines.append(stripped)
                         modified_cell = True
                    else:
                         new_lines.append(line)
                
                if modified_cell:
                    cell.source = "\n".join(new_lines)
                    print("Stripped indentation in Mod 4 Ex 4.")
        save_notebook(nb, path)

if __name__ == "__main__":
    repair_v10("c:\\dev\\python\\ML101")
