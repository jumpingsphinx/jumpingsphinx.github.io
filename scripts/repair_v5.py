import nbformat
import os
import re

def load_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)

def save_notebook(nb, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

def repair_v5(base_path):
    print("Executing Repair v5 (Phantom Stripper)...")

    # --- Mod 3 Ex 4: Remove 'regressors' garbage ---
    path = os.path.join(base_path, 'notebooks/module3-trees/solutions/solution_exercise4-boosting.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        new_cells = []
        for cell in nb.cells:
            if cell.cell_type == 'code' and "regressors" in cell.source and "items()" in cell.source:
                print("Dropping garbage 'regressors' cell in Mod 3 Ex 4.")
                continue
            new_cells.append(cell)
        nb.cells = new_cells
        save_notebook(nb, path)

    # --- Mod 4 Ex 1: Remove phantom indented lines ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise1-perceptron.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        for cell in nb.cells:
            if cell.cell_type == 'code':
                # Remove lines starting with 8 spaces followed by z = ...
                lines = cell.source.splitlines()
                new_lines = []
                for line in lines:
                    if re.match(r"^\s{8}z\s*=\s*np\.dot", line):
                        print("Skipping phantom line: z = np.dot... in Mod 4 Ex 1")
                        continue
                    if re.match(r"^\s{8}# 1\. Compute", line):
                        continue
                    if re.match(r"^\s+dz_db\s*=\s*1", line): # Fix dz_db indent too
                        new_lines.append(line.lstrip())
                        print("Dedented dz_db line.")
                        continue
                    new_lines.append(line)
                cell.source = "\n".join(new_lines)
        save_notebook(nb, path)

    # --- Mod 4 Ex 2: Fix NeuralNetwork Attribute (Loose Regex) ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise2-feedforward-networks.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        for cell in nb.cells:
            if "class NeuralNetwork" in cell.source:
                if "self.hidden_activation = sigmoid" not in cell.source:
                    # Look for self.b2 = ... and append.
                    # Handle flexible whitespace
                    cell.source = re.sub(
                        r"(self\.b2\s*=\s*np\.zeros\(\(1,\s*output_size\)\))", 
                        r"\1\n        self.hidden_activation = sigmoid", 
                        cell.source
                    )
                    print("Patched NeuralNetwork in Mod 4 Ex 2.")
        save_notebook(nb, path)

    # --- Mod 4 Ex 3: Remove phantom indented lines ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise3-backpropagation.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        for cell in nb.cells:
            if cell.cell_type == 'code':
                lines = cell.source.splitlines()
                new_lines = []
                for line in lines:
                    if re.match(r"^\s+z\s*=\s*np\.dot", line) and "def" not in cell.source:
                        print("Skipping phantom line: z = np.dot... in Mod 4 Ex 3")
                        continue
                    # Also fix indentation of Ex 1 copy-paste (dz_db) if present
                    if re.match(r"^\s+dz_db\s*=\s*1", line):
                        new_lines.append(line.lstrip())
                        continue 
                    new_lines.append(line)
                
                # Special check: If cell ONLY had phantom lines and is now empty/comment only?
                # Ideally we leave it or empty string.
                cell.source = "\n".join(new_lines)
        save_notebook(nb, path)

    # --- Mod 4 Ex 4: Remove phantom indented lines ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise4-numpy-implementation.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        for cell in nb.cells:
            if cell.cell_type == 'code':
                lines = cell.source.splitlines()
                new_lines = []
                for line in lines:
                    if re.match(r"^\s+model\s*=\s*NeuralNetwork\(\)", line):
                        print("Dedenting model = NN line.")
                        new_lines.append(line.lstrip())
                        continue
                    new_lines.append(line)
                cell.source = "\n".join(new_lines)
        save_notebook(nb, path)

if __name__ == "__main__":
    repair_v5("c:\\dev\\python\\ML101")
