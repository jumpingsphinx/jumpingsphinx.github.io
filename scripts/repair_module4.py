import nbformat
import os
import numpy as np

def load_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)

def save_notebook(nb, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

def find_cell_with_text(nb, text):
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and text in cell.source:
            return idx
    return -1

def fix_module4(base_path):
    print("Starting Module 4 Fixes...")
    
    # --- Ex 1: Perceptron (Fix Step Function) ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise1-perceptron.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        # Fix step_function to handle arrays
        step_code = """def step_function(z):
    \"\"\"
    Step activation function.
    Returns 1 if z >= 0, else 0
    \"\"\"
    return np.where(z >= 0, 1, 0)
"""
        idx = find_cell_with_text(nb, "def step_function(z):")
        if idx != -1:
            nb.cells[idx].source = step_code
        save_notebook(nb, path)

    # --- Ex 2: Feedforward (Fix hidden_activation definition) ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise2-feedforward-networks.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        # We need to find the NeuralNetwork class and ensure hidden_activation is set
        idx = find_cell_with_text(nb, "class NeuralNetwork:")
        if idx != -1:
             # Just inject the __init__ fix or full class?
             # Let's be surgical if possible, or robust overwrite.
             # Overwrite __init__ is perilous.
             # Let's overwrite the WHOLE CLASS if we have the content.
             # I don't have full content handy.
             # I will use replace logic on the source string.
             source = nb.cells[idx].source
             if "self.hidden_activation = sigmoid" not in source:
                 # Check init signature
                 if "def __init__(self" in source:
                     # Insert property initialization
                     source = source.replace("self.b2 = np.zeros((1, output_size))", 
                                             "self.b2 = np.zeros((1, output_size))\n        self.hidden_activation = sigmoid")
                     nb.cells[idx].source = source
        save_notebook(nb, path)

    # --- Ex 3: Backprop (Fix df_dx NameError) ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise3-backpropagation.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        # Fix compute_simple_derivative
        deriv_code = """def compute_simple_derivative():
    # Forward pass
    x = 1.0
    u = 3 * x + 2
    f = u ** 2
    
    # Backward pass
    df_du = 2 * u
    du_dx = 3
    df_dx = df_du * du_dx
    
    return f, df_dx
"""
        idx = find_cell_with_text(nb, "def compute_simple_derivative():")
        if idx != -1:
            nb.cells[idx].source = deriv_code
        save_notebook(nb, path)

    # --- Ex 4: Numpy Impl (Fix MSELoss backward) ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise4-numpy-implementation.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        # MSELoss backward
        mse_code = """class MSELoss:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true) ** 2)
    
    def backward(self):
        n = self.y_true.shape[0]
        return 2 * (self.y_pred - self.y_true) / n
"""
        idx = find_cell_with_text(nb, "class MSELoss:")
        if idx != -1:
            nb.cells[idx].source = mse_code
        save_notebook(nb, path)

def fix_mod3_stragglers(base_path):
    print("Fixing Mod 3 Stragglers...")
    # Ex 4 Plotting Call
    path = os.path.join(base_path, 'notebooks/module3-trees/solutions/solution_exercise4-boosting.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        # Look for the broken plot call
        target = "plot_decision_boundary(X_test_m, y_moons[y_moons != y_train_m]"
        # Robust search
        for idx, cell in enumerate(nb.cells):
             if "plot_decision_boundary(X_test_m," in cell.source:
                 # Replace with valid call using y_test_m
                 cell.source = cell.source.replace("y_moons[y_moons != y_train_m]", "y_test_m")
        save_notebook(nb, path)
        
    # Ex 2 Information Gain Typo
    path = os.path.join(base_path, 'notebooks/module3-trees/solutions/solution_exercise2-tree-algorithms.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        # Inspect for 'finedrmation' or broken text
        for idx, cell in enumerate(nb.cells):
             if "rmation" in cell.source and "information" not in cell.source: # partial
                 print(f"Found broken text in cell {idx}: {cell.source[:50]}...")
                 # Replace? hard to know what it was. 
                 # Often it's `information_gain` being split?
                 # If we find `print(f"{col:15s}: {ig:.4f}")`, lines before should be `ig = information_gain(...)`
                 if "print(f\"{col:15s}: {ig:.4f}\")" in cell.source:
                     # Force overwrite this cell
                     cell.source = """# Calculate information gain for each attribute
X_tennis = df_tennis.drop('Play', axis=1)
y_tennis = df_tennis['Play']

print("Information Gain for Each Attribute:\\n")
for col in X_tennis.columns:
    ig = information_gain(X_tennis, y_tennis, col)
    print(f"{col:15s}: {ig:.4f}")

print("\\nBest attribute to split on: The one with highest IG!")"""
        save_notebook(nb, path)

if __name__ == "__main__":
    base = "c:\\dev\\python\\ML101"
    fix_module4(base)
    fix_mod3_stragglers(base)
