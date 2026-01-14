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

def dedent_cell(source):
    # Split lines
    lines = source.splitlines()
    # Find first non-empty line
    first_code_line = None
    for line in lines:
        if line.strip() and not line.strip().startswith('#'):
            first_code_line = line
            break
    
    if first_code_line:
        # Check indentation of first code line
        params = re.match(r"^(\s+)", first_code_line)
        if params:
            # It's indented. Dedent the whole block.
            print(f"Dedenting logic found. Indent: {len(params.group(1))} chars.")
            return textwrap.dedent(source)
    return source

def repair_dedent_all(base_path):
    print("Executing Repair Deduplication & Dedent all...")
    
    mod4_dir = os.path.join(base_path, 'notebooks/module4-neural-networks/solutions')
    notebooks = [f for f in os.listdir(mod4_dir) if f.endswith('.ipynb')]
    
    for nb_file in notebooks:
        path = os.path.join(mod4_dir, nb_file)
        nb = load_notebook(path)
        modified = False
        print(f"Scanning {nb_file}...")
        
        for cell in nb.cells:
            if cell.cell_type == 'code':
                original = cell.source
                # Apply dedent check
                # We specifically target cells starting with indent
                # But allow 'class ' or 'def ' if they are indented? No, top level shouldn't be indented.
                # Exception: magic commands? %timeit? usually fine.
                
                # Check for specific garbage patterns from previous failures
                if "z = np.dot" in original and "def " not in original:
                     # This is likely the orphan cell. Dedent it.
                     new_source = dedent_cell(original)
                     if new_source != original:
                         cell.source = new_source
                         modified = True
                         print("Dedented 'z = np.dot' cell.")
                
                elif "model = " in original and "def " not in original:
                     new_source = dedent_cell(original)
                     if new_source != original:
                         cell.source = new_source
                         modified = True
                         print("Dedented 'model =' cell.")

                elif "dz_db =" in original:
                     new_source = dedent_cell(original)
                     if new_source != original:
                         cell.source = new_source
                         modified = True
                         print("Dedented 'dz_db =' cell.")
                
                # Mod 4 Ex 2 Fix
                if nb_file == "solution_exercise2-feedforward-networks.ipynb":
                    if "class NeuralNetwork" in cell.source and "def __init__" in cell.source:
                        if "self.hidden_activation = sigmoid" not in cell.source:
                             # Regex inject
                             pattern = r"(self\.b2\s*=\s*np\.zeros\(\(1,\s*output_size\)\))"
                             if re.search(pattern, cell.source):
                                 cell.source = re.sub(pattern, r"\1\n        self.hidden_activation = sigmoid", cell.source)
                                 modified = True
                                 print("Patched NeuralNetwork (Robust Regex).")
        
        if modified:
            save_notebook(nb, path)
            print(f"Saved repairs to {nb_file}")

if __name__ == "__main__":
    repair_dedent_all("c:\\dev\\python\\ML101")
