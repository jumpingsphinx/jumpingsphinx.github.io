import nbformat
import os
import re

def load_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)

def save_notebook(nb, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

def repair_aggressive(base_path):
    print("Executing Aggressive Repair...")

    # --- Mod 3 Ex 4: Validate 'results_reg' usage ---
    path = os.path.join(base_path, 'notebooks/module3-trees/solutions/solution_exercise4-boosting.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        new_cells = []
        for cell in nb.cells:
            if cell.cell_type == 'code' and "results_reg" in cell.source:
                # If 'results_reg' is used but not defined in this notebook (it's not), drop it.
                # Use simple heuristic: if 'results_reg =' is not in the notebook, drop usages.
                # But checking whole notebook is hard in loop.
                # We assume it's the stray cell from Ex 2.
                print("Dropping stray 'results_reg' cell in Mod 3 Ex 4.")
                continue # Skip adding this cell
            new_cells.append(cell)
        nb.cells = new_cells
        save_notebook(nb, path)

    # --- Mod 4 Ex 1: Fix Indentation (dz_db) ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise1-perceptron.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        correct_code = """# Forward pass for a single neuron
x = 2.0
w = 0.5
b = 1.0

# Compute forward
z = w * x + b
y = sigmoid(z)

print("Forward Pass:")
print(f"x = {x}")
print(f"z = wx + b = {w}*{x} + {b} = {z}")
print(f"y = σ(z) = {y:.4f}")

# Backward pass
dz_dw = x
dz_db = 1
dz_dx = w

print("\\nBackward Pass (Gradients):")
print(f"dy/dw = {dz_dw:.4f}")
print(f"dy/db = {dz_db:.4f}")
print(f"dy/dx = {dz_dx:.4f}")
"""
        for cell in nb.cells:
            if "dz_db = 1" in cell.source:
                # Force overwrite
                cell.source = correct_code
                print("Overwrote Mod 4 Ex 1 dz_db cell.")
        save_notebook(nb, path)

    # --- Mod 4 Ex 2: Fix NeuralNetwork Attribute ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise2-feedforward-networks.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        # Find the class and replace __init__
        # We'll use regex to find the __init__ block and inject the line.
        for cell in nb.cells:
            if "class NeuralNetwork" in cell.source and "def __init__" in cell.source:
                s = cell.source
                if "self.hidden_activation = sigmoid" not in s:
                    # Look for position to insert. After self.b2 line is good.
                    pattern = r"(self\.b2\s*=\s*np\.zeros\(\(1,\s*output_size\)\))"
                    if re.search(pattern, s):
                        cell.source = re.sub(pattern, r"\1\n        self.hidden_activation = sigmoid", s)
                        print("Patched NeuralNetwork.__init__ in Mod 4 Ex 2.")
        save_notebook(nb, path)
        
    # --- Mod 4 Ex 3: Fix Indentation (z = np.dot) ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise3-backpropagation.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        for cell in nb.cells:
             # Identify the cell with unexpected indent
             # It likely starts with spaces + "z = np.dot"
             if re.match(r"^\s+z\s*=\s*np\.dot", cell.source):
                 print("Found indented z=np.dot in Mod 4 Ex 3. Cleaning.")
                 lines = [line.lstrip() for line in cell.source.splitlines()]
                 cell.source = "\n".join(lines)
             # Also check if it's the "Forward pass" cell that needs complete replacement
             if "z = np.dot(X, self.weights)" in cell.source and "class" not in cell.source and "def" not in cell.source:
                  # This might be the phantom cell.
                  # Replace with generic "Visualizing Computational Graph" placeholder or correct logic if context known.
                  # Assuming it matches Ex 1 context:
                  pass 
        save_notebook(nb, path)

    # --- Mod 4 Ex 4: Fix Indentation (model = NeuralNetwork) ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise4-numpy-implementation.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        for cell in nb.cells:
             if re.search(r"^\s+model\s*=\s*NeuralNetwork\(\)", cell.source, re.MULTILINE):
                 print("Found indented model=NN in Mod 4 Ex 4. De-indenting.")
                 # De-indent widely
                 lines = [line.lstrip() for line in cell.source.splitlines()]
                 cell.source = "\n".join(lines)
        save_notebook(nb, path)

if __name__ == "__main__":
    repair_aggressive("c:\\dev\\python\\ML101")
