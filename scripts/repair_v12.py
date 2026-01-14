import nbformat
import os
import textwrap
import re

def load_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)

def save_notebook(nb, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

def repair_v12(base_path):
    print("Executing Repair v12 (Renaming & Resetting)...")

    # --- Mod 4 Ex 2: Rename softmax to avoid conflict ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise2-feedforward-networks.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        
        # 1. Update helper definition
        helpers = textwrap.dedent("""
            import numpy as np
            def sigmoid(z): return 1 / (1 + np.exp(-z))
            def softmax_activation(z):
                exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
                return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        """).strip()
        
        for cell in nb.cells:
            if "def softmax(z):" in cell.source:
                cell.source = helpers + "\n\nprint('Helpers injected')"
                print("Updated helpers in Mod 4 Ex 2.")
        
        # 2. Update NeuralNetwork Class usage
        nn_class = textwrap.dedent("""
            class NeuralNetwork:
                def __init__(self, n_input=2, n_hidden=2, n_output=1, hidden_activation=sigmoid):
                    self.W1 = np.random.randn(n_hidden, n_input) * 0.01
                    self.b1 = np.zeros((n_hidden, 1))
                    self.W2 = np.random.randn(n_output, n_hidden) * 0.01
                    self.b2 = np.zeros((n_output, 1))
                    self.hidden_activation = hidden_activation
                
                def forward(self, X):
                    # Input X shape: (n_input, m)
                    self.Z1 = np.dot(self.W1, X) + self.b1
                    self.A1 = self.hidden_activation(self.Z1)
                    
                    self.Z2 = np.dot(self.W2, self.A1) + self.b2
                    self.A2 = softmax_activation(self.Z2)
                    return self.A2
        """).strip()
        
        for cell in nb.cells:
            if "class NeuralNetwork" in cell.source:
                cell.source = nn_class
                print("Updated NeuralNetwork class (using softmax_activation) in Mod 4 Ex 2.")
        save_notebook(nb, path)

    # --- Mod 4 Ex 3: Strict rewrite of Backprop cell ---
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
             if "dz_dw =" in cell.source:
                 cell.source = correct_backprop
                 print("Strictly rewrote backprop cell in Mod 4 Ex 3.")
        save_notebook(nb, path)

    # --- Mod 4 Ex 5: Fix Training Loop Runtime Error ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise5-pytorch-basics.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        # Find the XOR training loop
        # We need to recreate the model inside the cell or before the loop to reset graph state?
        # Actually just ensure model creation is in the same cell.
        correct_loop = textwrap.dedent("""
            # Re-create model and data to ensure clean graph state
            X_xor = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
            y_xor = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
            
            model = nn.Sequential(
                nn.Linear(2, 4),
                nn.ReLU(),
                nn.Linear(4, 1),
                nn.Sigmoid()
            )
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.1)
            
            # Training loop
            losses = []
            for epoch in range(1000):
                # Forward pass
                outputs = model(X_xor)
                loss = criterion(outputs, y_xor)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (epoch+1) % 100 == 0:
                    print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')
                losses.append(loss.item()) # Store float, not tensor
        """).strip()
        
        for cell in nb.cells:
            if "for epoch in range(1000):" in cell.source and "X_xor" in cell.source:
                cell.source = correct_loop
                print("Patched XOR Training Loop in Mod 4 Ex 5.")
                
        save_notebook(nb, path)

if __name__ == "__main__":
    repair_v12("c:\\dev\\python\\ML101")
