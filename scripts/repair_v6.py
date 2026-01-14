import nbformat
import os
import textwrap

def load_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)

def save_notebook(nb, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

def repair_v6(base_path):
    print("Executing Repair v6 (The Overwriter)...")

    # --- Mod 4 Ex 2: Overwrite NeuralNetwork Class ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise2-feedforward-networks.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        correct_nn_class = textwrap.dedent("""
            class NeuralNetwork:
                def __init__(self, input_size, hidden_size, output_size):
                    self.W1 = np.random.randn(input_size, hidden_size) * 0.01
                    self.b1 = np.zeros((1, hidden_size))
                    self.W2 = np.random.randn(hidden_size, output_size) * 0.01
                    self.b2 = np.zeros((1, output_size))
                    self.hidden_activation = sigmoid
                
                def forward(self, X):
                    # Layer 1
                    self.Z1 = np.dot(X, self.W1) + self.b1
                    self.A1 = self.hidden_activation(self.Z1)
                    
                    # Layer 2
                    self.Z2 = np.dot(self.A1, self.W2) + self.b2
                    self.A2 = softmax(self.Z2)
                    
                    return self.A2
        """).strip()
        
        for cell in nb.cells:
            if "class NeuralNetwork" in cell.source and "def __init__" in cell.source:
                cell.source = correct_nn_class
                print("Overwrote NeuralNetwork class in Mod 4 Ex 2.")
        save_notebook(nb, path)

    # --- Mod 4 Ex 3: Overwrite Indented Backprop ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise3-backpropagation.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        correct_backprop_cell = textwrap.dedent("""
            # Backward pass
            dz_dw = x * y * (1 - y)  # Example gradient
            dz_db = y * (1 - y)
            dz_dx = w * y * (1 - y)

            print("\\nBackward Pass (Gradients):")
            print(f"dy/dw = {dz_dw:.4f}")
            print(f"dy/db = {dz_db:.4f}")
            print(f"dy/dx = {dz_dx:.4f}")
        """).strip()
        
        # Or better: just strictly dedent whatever is there if it contains dz_db
        for cell in nb.cells:
             if "dz_db = 1" in cell.source or "dz_dx = w" in cell.source:
                 lines = [line.lstrip() for line in cell.source.splitlines()]
                 cell.source = "\n".join(lines)
                 print("Dedented backprop cell in Mod 4 Ex 3.")
        save_notebook(nb, path)

    # --- Mod 4 Ex 4: Overwrite Model Instantiation ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise4-numpy-implementation.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        for cell in nb.cells:
             if "model = NeuralNetwork()" in cell.source:
                 lines = [line.lstrip() for line in cell.source.splitlines()]
                 cell.source = "\n".join(lines)
                 print("Dedented model instantiation in Mod 4 Ex 4.")
        save_notebook(nb, path)

    # --- Mod 4 Ex 1: Fix axes NameError ---
    # This usually means 'fig, axes = plt.subplots' cell failed or wasn't run before using 'axes'.
    # If indentation fixed 'z=np.dot' but 'axes' is used in same cell?
    # Inspecting 1264 view: 'axes' is used in cell 120-134.
    # Where is 'axes' defined? It should be in the SAME cell or previous.
    # If lines 132-133 (z=np.dot) caused failure, 'axes' plotting might have run?
    # But now 'NameError: axes' suggests `axes` is not in scope.
    # We will PREPEND `fig, axes = plt.subplots(2, 2, ...)` to the cell using axes[1, 1] if it's missing.
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise1-perceptron.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        for cell in nb.cells:
            if "axes[1, 1].plot" in cell.source and "plt.subplots" not in cell.source:
                # Inject definition
                header = "fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n"
                cell.source = header + cell.source
                print("Injected axes definition in Mod 4 Ex 1.")
        save_notebook(nb, path)

if __name__ == "__main__":
    repair_v6("c:\\dev\\python\\ML101")
