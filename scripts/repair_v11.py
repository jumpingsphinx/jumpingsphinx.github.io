import nbformat
import os
import textwrap

def load_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)

def save_notebook(nb, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

def repair_v11(base_path):
    print("Executing Repair v11 (Matrix Dimensions & Loop Strip)...")

    # --- Mod 4 Ex 2: NeuralNetwork for (n_x, m) input shape ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise2-feedforward-networks.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        correct_nn_class = textwrap.dedent("""
            class NeuralNetwork:
                def __init__(self, n_input=2, n_hidden=2, n_output=1, hidden_activation=sigmoid):
                    # Weights: (n_neurons, n_inputs) for W @ X
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
                    self.A2 = softmax(self.Z2) # Or output activation
                    return self.A2
        """).strip()
        
        for cell in nb.cells:
            # Replace the class definition completely
            if "class NeuralNetwork" in cell.source and "def __init__" in cell.source:
                cell.source = correct_nn_class
                print("Overwrote NeuralNetwork (Matrix Math Layout) in Mod 4 Ex 2.")
        save_notebook(nb, path)

    # --- Mod 4 Ex 3 & 4: Aggressive Indent Strip ---
    # Targets: dz_dw, dz_dx, _onehot, plt.show
    target_files = [
        'notebooks/module4-neural-networks/solutions/solution_exercise3-backpropagation.ipynb',
        'notebooks/module4-neural-networks/solutions/solution_exercise4-numpy-implementation.ipynb'
    ]
    
    for rel_path in target_files:
        path = os.path.join(base_path, rel_path)
        if not os.path.exists(path): continue
        
        nb = load_notebook(path)
        for cell in nb.cells:
            if cell.cell_type == 'code':
                lines = cell.source.splitlines()
                new_lines = []
                modified_cell = False
                for line in lines:
                    stripped = line.strip() # strip BOTH sides for checking logic
                    
                    # Target specific lines to force UNINDENT (lstrip)
                    if stripped.startswith("dz_dw") or \
                       stripped.startswith("dz_db") or \
                       stripped.startswith("dz_dx") or \
                       stripped.startswith("_onehot") or \
                       stripped.startswith("plt.show()") or \
                       stripped.startswith("model = NeuralNetwork()"):
                         
                         new_lines.append(line.lstrip()) # Keep right whitespace, kill left
                         modified_cell = True
                    else:
                         new_lines.append(line)
                
                if modified_cell:
                    cell.source = "\n".join(new_lines)
                    print(f"Stripped lines in {os.path.basename(rel_path)}")
        save_notebook(nb, path)

if __name__ == "__main__":
    repair_v11("c:\\dev\\python\\ML101")
