import nbformat
import os
import textwrap

def load_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)

def save_notebook(nb, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

def repair_v7(base_path):
    print("Executing Repair v7 (Signature Match & Cleanup)...")

    # --- Mod 3 Ex 4: Delete 'regressors' garbage ---
    path = os.path.join(base_path, 'notebooks/module3-trees/solutions/solution_exercise4-boosting.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        new_cells = []
        for cell in nb.cells:
            # Drop cells mentioning regressors (the garbage list from Ex 2)
            if cell.cell_type == 'code' and "regressors" in cell.source:
                print("Dropping 'regressors' cell in Mod 3 Ex 4.")
                continue
            new_cells.append(cell)
        nb.cells = new_cells
        save_notebook(nb, path)

    # --- Mod 4 Ex 2: Overwrite NeuralNetwork Class (Signature Fix) ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise2-feedforward-networks.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        # Fix signature to match 'n_input', 'n_hidden', 'n_output'
        # And ensure self.hidden_activation is set.
        correct_nn_class = textwrap.dedent("""
            class NeuralNetwork:
                def __init__(self, n_input, n_hidden, n_output):
                    self.W1 = np.random.randn(n_input, n_hidden) * 0.01
                    self.b1 = np.zeros((1, n_hidden))
                    self.W2 = np.random.randn(n_hidden, n_output) * 0.01
                    self.b2 = np.zeros((1, n_output))
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
                print("Overwrote NeuralNetwork class (sig fix) in Mod 4 Ex 2.")
        save_notebook(nb, path)

    # --- Mod 4 Ex 3: Overwrite Indented Backprop ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise3-backpropagation.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        correct_backprop = textwrap.dedent("""
            # Backward pass
            dz_dw = x * y * (1 - y)  # Example gradient
            dz_db = y * (1 - y)
            dz_dx = w * y * (1 - y)

            print("\\nBackward Pass (Gradients):")
            print(f"dy/dw = {dz_dw:.4f}")
            print(f"dy/db = {dz_db:.4f}")
            print(f"dy/dx = {dz_dx:.4f}")
        """).strip()
        
        for cell in nb.cells:
             # Find cell with dz_dx (likely indented) and replace it
             if "dz_dx = w" in cell.source:
                 cell.source = correct_backprop
                 print("Overwrote backprop cell in Mod 4 Ex 3.")
        save_notebook(nb, path)

    # --- Mod 4 Ex 4: Overwrite Model Instantiation ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise4-numpy-implementation.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        correct_model_cell = textwrap.dedent("""
            # Create model
            model = NeuralNetwork()
            
            # Forward pass
            output = model.forward(X)
            print("Model output shape:", output.shape)
        """).strip()

        for cell in nb.cells:
             if "model = NeuralNetwork()" in cell.source:
                 cell.source = correct_model_cell
                 print("Overwrote model instantiation in Mod 4 Ex 4.")
        save_notebook(nb, path)

if __name__ == "__main__":
    repair_v7("c:\\dev\\python\\ML101")
