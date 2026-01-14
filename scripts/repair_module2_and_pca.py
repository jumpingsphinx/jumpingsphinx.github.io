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

def fix_pca_digits(base_path):
    # The PCA failure was: ValueError: matmul: Input operand 1 has a mismatch... size 64 is different from 2
    # In 'Reconstruct: X_reconstructed = X_compressed @ pca_temp.components_ + pca_temp.mean_'
    # Components shape is (k, n_features) -> (2, 64). 
    # X_compressed is (n_samples, k) -> (1797, 2).
    # X_compressed @ components -> (n, k) @ (k, d) = (n, d). Correct.
    # Waitttt. 
    # previous repair script:
    # self.components_ = eigenvectors[:, :self.n_components]
    # self.eigenvalues_ = eigenvalues[:self.n_components]
    # Eigenvectors of covariance matrix (d, d). So components is (d, k).
    # X (n, d) @ components (d, k) -> (n, k).
    # Reconstruct: X_comp (n, k) @ components.T (k, d)?
    # Sklearn components_ is (n_components, n_features) -> (k, d).
    # Our implementation components_ is (d, k).
    
    # Sklearn style: fit_transform returns (n_samples, n_components).
    # inverse_transform: X_transformed @ components_ + mean_
    # If components_ is (n_components, n_features), then X_trans @ components_ -> (n, k) @ (k, d) -> (n, d).
    
    # Our PCA implementation (Cell 232 in solution notebook) defines:
    # self.components_ = eigenvectors[:, :self.n_components] -> This is (d, k)!
    # So X_transformed = X @ self.components_ -> (n, d) @ (d, k) -> (n, k).
    # For reconstruction: X_trans @ self.components_.T -> (n, k) @ (k, d) -> (n, d).
    
    # The failing code in Digits Dataset section (Cell 700ish) uses:
    # X_reconstructed = X_compressed @ pca_temp.components_
    # It assumes components_ is (k, d) like Sklearn? 
    # Or maybe `pca_temp` is an instance of OUR class, or SKLEARN class?
    
    # The notebook mixes "Compare with sklearn" and "Digits dataset".
    # Usually Digits uses sklearn PCA for simplicity or the custom one.
    # The error message says: "size 64 is different from 2".
    # If using sklearn PCA: components_ is (2, 64). X_compressed (n, 2).
    # X_compressed @ components_ -> (n, 2) @ (2, 64) -> (n, 64). Works.
    
    # If using OUR PCA: components_ is (64, 2).
    # X_compressed @ components_ -> (n, 2) @ (64, 2) -> ERROR (2 != 64).
    
    # So the Digits cell is likely using OUR PCA class but treating it like Sklearn (or vice versa).
    # Or implies dot product logic mismatch.
    
    # Let's inspect the cell content in `fix_pca_digits`.
    # Based on the error, we are dotting (n, 2) with something that expects matching dim 2...
    # If (n, 2) @ (2, 64) -> OK.
    # If (n, 2) @ (64, 2) -> Fail.
    
    # So `pca_temp.components_` is (64, 2) -> Our Class.
    # And we called `X_compressed @ pca_temp.components_`.
    # To fix for Our Class: `X_compressed @ pca_temp.components_.T`
    
    nb_path = os.path.join(base_path, 'notebooks/module1-linear-algebra/solutions/solution_exercise3-pca.ipynb')
    if not os.path.exists(nb_path): return
    
    print(f"Fixing PCA Digits in {nb_path}...")
    nb = load_notebook(nb_path)
    
    # Search for the failing line
    idx = find_cell_with_text(nb, "X_reconstructed = X_compressed @ pca_temp.components_")
    if idx != -1:
        # Detect if it's using our Class or Sklearn.
        # It instantiates: pca_temp = PCA(n_components=n_comp)
        # If imports: from . import PCA? No, defined in notebook.
        # So it uses the class defined in the notebook.
        # Our class has components_ as (n_features, n_components).
        
        # FIX: Transpose components for reconstruction
        source = nb.cells[idx].source
        new_source = source.replace(
            "X_reconstructed = X_compressed @ pca_temp.components_ + pca_temp.mean_",
            "X_reconstructed = X_compressed @ pca_temp.components_.T + pca_temp.mean_"
        )
        nb.cells[idx].source = new_source
        print("Fixed PCA reconstruction matrix multiplication.")
        save_notebook(nb, nb_path)

def repair_module2_part2(base_path):
    # Exercise 2: Gradient Descent
    nb_path = os.path.join(base_path, 'notebooks/module2-regression/solutions/solution_exercise2-gradient-descent.ipynb')
    if os.path.exists(nb_path):
        print(f"Repairing {nb_path}...")
        nb = load_notebook(nb_path)
        
        # 1. Fix Cost Function (Cell 65ish)
        # Current view shows it is mostly correct, but let's ensure.
        # View showed: def compute_cost(X, y, weights): ... return cost
        # It looked correct in the "View File" output.
        # Wait, the view output for module 2 notebooks showed they WERE populated?
        # Let's re-verify the "View File" output for Ex 2.
        # Cell 65: def compute_cost... looks perfect.
        # Cell 135: def batch_gradient_descent... looks perfect.
        # Cell 286: def stochastic_gradient_descent... looks perfect.
        # Cell 401: def minibatch... looks perfect.
        # Cell 499: class GradientDescentRegressor... looks perfect.
        
        # Check Exercise 3: Logistic Regression
        # Cell 65: def sigmoid... perfect.
        # Cell 121: def sigmoid_derivative... perfect.
        # Cell 175: def binary_cross_entropy... perfect.
        # Cell 242: class LogisticRegression... perfect.
        
        # Check Exercise 4: Regularization
        # Cell 151: class RidgeRegression... perfect.
        # Cell 395: class LassoRegression... perfect.
        
        # Wait, if Module 2 (GD, LogReg, Reg) are already correct, why did LinReg fail?
        # LinReg failed because of shifted cells.
        # Did these notebooks escape the shift?
        # In Ex 2 (GD):
        # Cell 37: "Exercise 1.1: Implement Cost Function" -> Cell 60: Code.
        # Cell 115: "Exercise 2.1: Implement Batch GD" -> Cell 130: Code.
        # This looks aligned.
        
        # In Ex 1 (LinReg) before fix:
        # Cell 62: "Fit Simple LR". Code had "def fit_simple_linear_regression".
        # That was also aligned?
        # Why did I think LinReg was broken?
        # Ah, in my earlier analysis of LinReg, I saw:
        # "Cell 493 (Challenge 2 Pinv) has: poly = PolynomialFeatures... (SOLUTION for Challenge 1)"
        # So challenges were shifted.
        
        # Let's review Module 2 Challenges.
        # Ex 2 (GD): No challenges section in View?
        # It ends with part 8 (Compare with Sklearn).
        
        # Ex 3 (LogReg):
        # Ends with Part 7 (Wine Dataset).
        # Any challenges?
        # View cut off at line 800.
        
        # It seems only the MAIN tasks in Ex 2, 3, 4 are fine.
        # But I should check if there are shifted parts near the end.
        pass

if __name__ == "__main__":
    base_path = "c:\\dev\\python\\ML101"
    fix_pca_digits(base_path)
