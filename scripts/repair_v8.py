import nbformat
import os
import textwrap

def load_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)

def save_notebook(nb, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

def repair_v8(base_path):
    print("Executing Repair v8 (Context Injector)...")

    # --- Mod 4 Ex 1: Inject sigmoid in plotting cell ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise1-perceptron.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        for cell in nb.cells:
            # If cell uses sigmoid but doesn't define it
            if "sigmoid(" in cell.source and "def sigmoid" not in cell.source:
                # Prepend definition
                # We also need axes?
                header = textwrap.dedent("""
                    import numpy as np
                    import matplotlib.pyplot as plt
                    def sigmoid(z): return 1 / (1 + np.exp(-z))
                    def relu(z): return np.maximum(0, z)
                """).strip() + "\n"
                
                # Only prepend if not already there
                if "def sigmoid" not in cell.source:
                    cell.source = header + cell.source
                    print("Injected sigmoid/relu in Mod 4 Ex 1.")
        save_notebook(nb, path)

    # --- Mod 4 Ex 2: Update NeuralNetwork Init ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise2-feedforward-networks.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        correct_nn_class = textwrap.dedent("""
            class NeuralNetwork:
                def __init__(self, n_input, n_hidden, n_output, hidden_activation=sigmoid):
                    self.W1 = np.random.randn(n_input, n_hidden) * 0.01
                    self.b1 = np.zeros((1, n_hidden))
                    self.W2 = np.random.randn(n_hidden, n_output) * 0.01
                    self.b2 = np.zeros((1, n_output))
                    self.hidden_activation = hidden_activation
                
                def forward(self, X):
                    self.Z1 = np.dot(X, self.W1) + self.b1
                    self.A1 = self.hidden_activation(self.Z1)
                    self.Z2 = np.dot(self.A1, self.W2) + self.b2
                    self.A2 = softmax(self.Z2)
                    return self.A2
        """).strip()
        
        for cell in nb.cells:
            if "class NeuralNetwork" in cell.source and "def __init__" in cell.source:
                cell.source = correct_nn_class
                print("Overwrote NeuralNetwork (kwarg fix) in Mod 4 Ex 2.")
        save_notebook(nb, path)

    # --- Mod 4 Ex 3: Inject context for backprop ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise3-backpropagation.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        correct_backprop = textwrap.dedent("""
            # Setup for backprop example
            x = 1.0
            w = 0.5
            b = 0.0
            z = w * x + b
            y = 1 / (1 + np.exp(-z))  # sigmoid
            
            # Backward pass
            dz_dw = x * y * (1 - y)
            dz_db = y * (1 - y)
            dz_dx = w * y * (1 - y)

            print("\\nBackward Pass (Gradients):")
            print(f"dy/dw = {dz_dw:.4f}")
            print(f"dy/db = {dz_db:.4f}")
            print(f"dy/dx = {dz_dx:.4f}")
        """).strip()
        
        for cell in nb.cells:
             # Match by content derived from v7
             if "dz_dx = w" in cell.source:
                 cell.source = correct_backprop
                 print("Injected context into backprop cell in Mod 4 Ex 3.")
        save_notebook(nb, path)

    # --- Mod 4 Ex 4: Inject context for model test ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise4-numpy-implementation.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        correct_model_cell = textwrap.dedent("""
            # Create model
            model = NeuralNetwork()
            
            # Create dummy data
            X = np.random.randn(5, 2) # Batch 5, 2 input features
            
            # Forward pass
            output = model.forward(X)
            print("Model output shape:", output.shape)
        """).strip()

        for cell in nb.cells:
             if "model = NeuralNetwork()" in cell.source:
                 cell.source = correct_model_cell
                 print("Injected X context in Mod 4 Ex 4.")
        save_notebook(nb, path)

if __name__ == "__main__":
    repair_v8("c:\\dev\\python\\ML101")
