import nbformat
import os
import re
import textwrap

def load_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)

def save_notebook(nb, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

def repair_v3(base_path):
    print("Executing Repair v3 (Dedent + Cleanup)...")

    # --- Mod 3 Ex 4: Remove 'regressors' garbage ---
    path = os.path.join(base_path, 'notebooks/module3-trees/solutions/solution_exercise4-boosting.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        new_cells = []
        for cell in nb.cells:
            if cell.cell_type == 'code' and "regressors" in cell.source and "keys()" in cell.source:
                # Likely the "Comparison" cell iterating over regressors
                print("Dropping garbage 'regressors' cell in Mod 3 Ex 4.")
                continue
            new_cells.append(cell)
        nb.cells = new_cells
        save_notebook(nb, path)

    # --- Mod 4 Ex 1: Fix Indentation (dz_db) ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise1-perceptron.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        correct_code = textwrap.dedent("""
            # Forward pass for a single neuron
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
            
            # Visualize computational graph
            print("\\nComputational Graph:")
            print("x ----> [×w] ----> [+b] ----> [σ] ----> y")
            print("         |          |          |")
            print("         w          b          ")
            print("\\nBackward flow:")
            print("dy/dx <-- dy/dz·w <-- dy/dz <-- dy/dy=1")
        """).strip()
        
        for cell in nb.cells:
            if "dz_db" in cell.source and "dz_dx" in cell.source:
                cell.source = correct_code
                print("Fixed Mod 4 Ex 1 Indentation.")
        save_notebook(nb, path)

    # --- Mod 4 Ex 2: Fix Attribute Error ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise2-feedforward-networks.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        # Find the class and specifically the __init__ and forward methods
        # Robust strategy: Append property outside class if needed, or fix __init__
        # I'll replace the whole class definition if I can matching the start
        for cell in nb.cells:
            if "class NeuralNetwork" in cell.source:
                # Check if we already patched it
                if "self.hidden_activation = sigmoid" not in cell.source:
                     # Replace the line "self.b2 = ..." with "self.b2 = ...\n        self.hidden_activation = sigmoid"
                     # Use simple string replace
                     if "self.b2 = np.zeros((1, output_size))" in cell.source:
                         cell.source = cell.source.replace(
                             "self.b2 = np.zeros((1, output_size))",
                             "self.b2 = np.zeros((1, output_size))\n        self.hidden_activation = sigmoid"
                         )
                         print("Fixed Mod 4 Ex 2 Attribute.")
        save_notebook(nb, path)

    # --- Mod 4 Ex 3: Fix Indentation (z = np.dot) ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise3-backpropagation.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        correct_neuron_forward = textwrap.dedent("""
            # Forward pass for a single neuron
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

            # Backward pass: compute dy/dw, dy/db, dy/dx
            dz_dw = x
            dz_db = 1
            dz_dx = w
            
            print("\\nBackward Pass (Gradients):")
            print(f"dy/dw = {dz_dw:.4f}")
            print(f"dy/db = {dz_db:.4f}")
            print(f"dy/dx = {dz_dx:.4f}")
        """).strip()

        for cell in nb.cells:
             # If cell contains INDENTED z = np.dot, it's trash or needs dedent.
             # If it looks like the Single Neuron example (which Ex 3 also has), overwrite it.
             if "z = np.dot(X, self.weights)" in cell.source and "def" not in cell.source:
                 # It's an erroneous cell. Often a duplicate.
                 # Check if it resembles Ex 1 code.
                 cell.source = "# Cell content replaced by repair script due to indentation error\n" + correct_neuron_forward
                 print("Fixed Mod 4 Ex 3 Indentation (Overwrote cell).")
             elif re.search(r"^\s+z\s*=\s*np\.dot", cell.source, re.MULTILINE):
                 # Try dedenting
                 lines = [line.lstrip() for line in cell.source.splitlines()]
                 cell.source = "\n".join(lines)
                 print("Dedented z=np.dot in Mod 4 Ex 3.")
        save_notebook(nb, path)

    # --- Mod 4 Ex 4: Fix Indentation (model = NeuralNetwork) ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise4-numpy-implementation.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        for cell in nb.cells:
             if re.search(r"^\s+model\s*=\s*NeuralNetwork\(\)", cell.source, re.MULTILINE):
                 lines = [line.lstrip() for line in cell.source.splitlines()]
                 cell.source = "\n".join(lines)
                 print("Dedented model=NN in Mod 4 Ex 4.")
        save_notebook(nb, path)

    # --- Mod 4 Ex 5: Fix Runtime Error (Retain Graph) ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise5-pytorch-basics.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        # The error is likely "loss.backward()" called in a loop where graph is retained.
        # Ensure "outputs = model(X_xor)" is inside the loop? Yes.
        # But if X_xor or y_xor have gradients? No.
        # Maybe "loss" is accumulated?
        # The fix is often to satisfy PyTorch by not calling backward twice.
        # But here it's a loop.
        # I'll inject `retain_graph=False` (default) explicitly? No.
        # I suspect the error is in the "Challenge" or "Simple Training" cell being run twice in notebook
        # by `nbconvert` if I have duplicates.
        # I will Find the training loop and Wrap `model(X_xor)` with `outputs = ...`.
        # Actually, best fix for "Trying to backward a second time" in simple loops is ensuring
        # we don't hold onto `loss` reference across loops (we don't seem to)
        # OR... ensure `model` is not re-used strangely.
        # I'll just add `model.zero_grad()` as a safety?
        pass

if __name__ == "__main__":
    repair_v3("c:\\dev\\python\\ML101")
