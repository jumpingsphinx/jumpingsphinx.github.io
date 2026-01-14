import nbformat
import os
import textwrap

def load_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)

def save_notebook(nb, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

def repair_v18(base_path):
    print("Executing Repair v18 (Device Fix)...")

    # --- Mod 4 Ex 5: Move Model to Device ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise5-pytorch-basics.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        
        # We need to find where model is instantiated and add .to(device)
        # In repair_v16, we injected:
        # model = MNISTClassifier()
        # print("Model initialized:")
        
        for cell in nb.cells:
            if "model = MNISTClassifier()" in cell.source and ".to(device)" not in cell.source:
                cell.source = cell.source.replace(
                    "model = MNISTClassifier()",
                    "model = MNISTClassifier().to(device)"
                )
                print("Moved model to device in Mod 4 Ex 5.")
                
        save_notebook(nb, path)

    # --- Mod 4 Ex 6: Check/Move Model to Device ---
    path = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions/solution_exercise6-pytorch-advanced.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        
        # Finding model instantiation. It might be different in Ex 6 (e.g. transfer learning)
        # But generally, look for model = ...
        # Safeguard: if there's a cell with model = ... and it doesn't have .to(device), append it?
        # Ex 6 often uses `model = models.resnet18(...)`
        
        for cell in nb.cells:
             if ("model = " in cell.source or "net =" in cell.source) and ".to(device)" not in cell.source:
                 # Logic is risky to regex replace blindly.
                 # Let's look for standard patterns
                 if "model = MNISTClassifier()" in cell.source:
                     cell.source = cell.source.replace("model = MNISTClassifier()", "model = MNISTClassifier().to(device)")
                     print("Moved MNISTClassifier to device in Mod 4 Ex 6.")
        save_notebook(nb, path)

if __name__ == "__main__":
    repair_v18("c:\\dev\\python\\ML101")
