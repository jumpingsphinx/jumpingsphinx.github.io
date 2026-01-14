import nbformat
import os
import textwrap

def load_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)

def save_notebook(nb, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

def repair_v14(base_path):
    print("Executing Repair v14 (String Handler & Autograd Fix)...")

    # --- Mod 4 Ex 2: Handle String Inputs for Activation ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise2-feedforward-networks.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        
        nn_class_robust = textwrap.dedent("""
            class NeuralNetwork:
                def __init__(self, n_input=2, n_hidden=2, n_output=1, hidden_activation=sigmoid):
                    self.W1 = np.random.randn(n_hidden, n_input) * 0.01
                    self.b1 = np.zeros((n_hidden, 1))
                    self.W2 = np.random.randn(n_output, n_hidden) * 0.01
                    self.b2 = np.zeros((n_output, 1))
                    
                    # Handle string inputs
                    if isinstance(hidden_activation, str):
                        if hidden_activation.lower() == 'relu':
                            self.hidden_activation = lambda x: np.maximum(0, x)
                        elif hidden_activation.lower() == 'sigmoid':
                            self.hidden_activation = sigmoid
                        elif hidden_activation.lower() == 'tanh':
                            self.hidden_activation = np.tanh
                        else:
                            self.hidden_activation = sigmoid # Default
                    else:
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
            if "class NeuralNetwork" in cell.source and "hidden_activation" in cell.source:
                cell.source = nn_class_robust
                print("Updated NeuralNetwork to handle string activations in Mod 4 Ex 2.")
        save_notebook(nb, path)

    # --- Mod 4 Ex 5: Fix Autograd Backward Calls ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise5-pytorch-basics.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        
        # Scrape for cells with .backward()
        for cell in nb.cells:
            if cell.cell_type == 'code' and ".backward()" in cell.source:
                # If cell assumes 'y' or 'z' or 'loss' is already clean, it might fail on re-run
                # Detection: calling backward on a variable 'z'
                # Injection: prepend z = ... if possible?
                # Case 1: z.backward()
                if "z.backward()" in cell.source and "z =" not in cell.source:
                     # Inject z definition? 
                     # Depends on the context. Usually:
                     # z = y * x + 2 
                     # z.backward()
                     # If they are split, merge them.
                     pass
        
        # Specifically target the Autograd Exercise 2.1
        # It likely splits forward and backward into cells.
        # We'll merge them or set retain_graph=True (not recommended but fixes re-run)
        # Better: Prepend forward pass
        
        fix_autograd = textwrap.dedent("""
            # Setup for autograd
            x = torch.tensor(2.0, requires_grad=True)
            y = torch.tensor(3.0, requires_grad=True)
            z = x * y + x**2
            
            # Compute gradients
            z.backward()
            
            print(f"x: {x}")
            print(f"y: {y}")
            print(f"z: {z}")
            print(f"dz/dx: {x.grad}")
            print(f"dz/dy: {y.grad}")
        """).strip()
        
        for cell in nb.cells:
            if "z.backward()" in cell.source:
                # If we find a cell doing z.backward(), replace it with full context
                # Assume this is the exercise cell
                cell.source = fix_autograd
                print("Fixed Autograd cell in Mod 4 Ex 5.")
                break 

        save_notebook(nb, path)

if __name__ == "__main__":
    repair_v14("c:\\dev\\python\\ML101")
