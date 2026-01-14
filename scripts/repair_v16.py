import nbformat
import os
import textwrap

def load_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)

def save_notebook(nb, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

def repair_v16(base_path):
    print("Executing Repair v16 (MNIST Fix)...")

    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise5-pytorch-basics.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        
        # 1. Fix MNISTClassifier Class
        # Expect 28*28 = 784 inputs
        # Flatten in forward
        correct_mnist_class = textwrap.dedent("""
            class MNISTClassifier(nn.Module):
                \"\"\"
                Neural network for MNIST classification.
                Architecture: Flatten -> Dense -> ReLU -> Dense -> ReLU -> Dense
                \"\"\"
                def __init__(self):
                    super(MNISTClassifier, self).__init__()
                    # Input is 28x28 = 784
                    self.fc1 = nn.Linear(784, 128)
                    self.fc2 = nn.Linear(128, 64)
                    self.fc3 = nn.Linear(64, 10) # 10 digits
                    
                def forward(self, x):
                    # Flatten input [Batch, 1, 28, 28] -> [Batch, 784]
                    x = x.view(-1, 784)
                    
                    x = F.relu(self.fc1(x))
                    x = F.relu(self.fc2(x))
                    x = self.fc3(x)
                    return x

            # Initialize model
            model = MNISTClassifier()
            print("Model initialized:")
            print(model)
        """).strip()
        
        for cell in nb.cells:
            if "class MNISTClassifier" in cell.source:
                cell.source = correct_mnist_class
                print("Fixed MNISTClassifier architecture in Mod 4 Ex 5.")

        # 2. Remove Legacy Training Loop
        # Look for the loop that uses X_train directly
        legacy_loop_marker = "outputs = model_digits(X_train)"
        
        new_cells = []
        for cell in nb.cells:
            if legacy_loop_marker in cell.source:
                print("Removing legacy training loop in Mod 4 Ex 5.")
                # We can just skip adding this cell to new_cells, deleting it.
                continue 
            
            # Also remove the cell defining model_digits if it's separate?
            # In previous view, model_digits was defined right after class.
            # IN my overwrite above, I replaced the class cell with class + init.
            # So if model_digits was in a separate cell, I should check.
            # View 1588 showed:
            # Cell 1: class definition ...
            # Cell 2: model_digits = nn.Sequential... AND loop
            # Wait, view 1588 showed `model_digits = nn.Sequential` AND the loop in the SAME cell (lines 599-620).
            # So simply deleting the cell with `model_digits(X_train)` might delete the model definition if I didn't verify structure.
            # Ah, the `class MNISTClassifier` was in a separate cell (lines 589-597).
            # Then `model_digits` and the loop were in the NEXT cell.
            # So:
            # 1. Update Class cell.
            # 2. Delete the `model_digits` + Loop cell.
            # 3. Ensure `model` is instantiated (I did that in step 1).
            
            if "model_digits = nn.Sequential" in cell.source:
                 print("Removing legacy model_digits cell in Mod 4 Ex 5.")
                 continue

            new_cells.append(cell)
            
        nb.cells = new_cells
        save_notebook(nb, path)

if __name__ == "__main__":
    repair_v16("c:\\dev\\python\\ML101")
