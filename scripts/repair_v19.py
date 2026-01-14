import nbformat
import os
import re

def load_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)

def save_notebook(nb, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

def normalize_indentation(source):
    lines = source.splitlines()
    new_lines = []
    
    for line in lines:
        # 1. Convert tabs to 4 spaces
        line = line.replace('\t', '    ')
        
        # 2. Check for trailing whitespace (optional but good)
        line = line.rstrip()
        
        new_lines.append(line)
        
    return "\n".join(new_lines)

def repair_v19(base_path):
    print("Executing Repair v19 (Global Indentation Fix)...")
    
    target_dirs = [
        "notebooks/module3-trees/solutions",
        "notebooks/module4-neural-networks/solutions"
    ]
    
    for relative_dir in target_dirs:
        full_dir = os.path.join(base_path, relative_dir)
        if not os.path.exists(full_dir):
            continue
            
        for filename in os.listdir(full_dir):
            if filename.endswith(".ipynb"):
                filepath = os.path.join(full_dir, filename)
                try:
                    nb = load_notebook(filepath)
                    modified = False
                    
                    for cell in nb.cells:
                        if cell.cell_type == 'code':
                            original = cell.source
                            normalized = normalize_indentation(original)
                            
                            if original != normalized:
                                cell.source = normalized
                                modified = True
                    
                    if modified:
                        save_notebook(nb, filepath)
                        print(f"Fixed indentation in {filename}")
                        
                except Exception as e:
                    print(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    repair_v19("c:\\dev\\python\\ML101")
