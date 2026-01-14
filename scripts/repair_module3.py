import nbformat
import os
import numpy as np
import pandas as pd

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

def fix_all_module3(base_path):
    print("Starting Module 3 Fixes (Final Targeted Round)...")
    
    # --- Ex 3: Random Forest (Fix Indentation) ---
    path = os.path.join(base_path, 'notebooks/module3-trees/solutions/solution_exercise3-random-forest.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        # Random Forest Class with correct indentation
        rf_code = """class RandomForestFromScratch:
    def __init__(self, n_estimators=10, max_depth=None, max_features='sqrt', random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        
    def fit(self, X, y):
        self.trees = []
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        for _ in range(self.n_estimators):
            # Bootstrap
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[indices], y[indices]
            
            # Feature selection
            n_feat = n_features
            if self.max_features == 'sqrt':
                n_feat = int(np.sqrt(n_features))
            elif self.max_features == 'log2':
                n_feat = int(np.log2(n_features))
            
            # Train tree
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth, 
                max_features=n_feat,
                random_state=np.random.randint(10000)
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Majority vote
        return np.array([np.bincount(tree_preds[:, i]).argmax() for i in range(X.shape[0])])

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
print("RandomForest Class Fixed!")
"""
        idx = find_cell_with_text(nb, "class RandomForestFromScratch")
        if idx != -1:
            nb.cells[idx].source = rf_code
        save_notebook(nb, path)

    # --- Ex 5: XGBoost (Fix Regression Data) ---
    path = os.path.join(base_path, 'notebooks/module3-trees/solutions/solution_exercise5-xgboost.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        # Inject Housing Data for Regression
        reg_data_code = """# Load Housing Data
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
housing = fetch_california_housing()
X_h, y_h = housing.data, housing.target
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_h, y_h, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_h_scaled = scaler.fit_transform(X_train_h)
X_test_h_scaled = scaler.transform(X_test_h)
"""
        # Find Regression Cell
        idx = find_cell_with_text(nb, "xgb_reg_baseline = XGBRegressor")
        if idx != -1:
             source = nb.cells[idx].source
             if "fetch_california_housing" not in source:
                 print("Prepending Housing Data to XGBoost Regression cell.")
                 nb.cells[idx].source = reg_data_code + "\n" + source
        save_notebook(nb, path)

    # --- Ex 4: AdaBoost (Explicit Plot Helper) ---
    path = os.path.join(base_path, 'notebooks/module3-trees/solutions/solution_exercise4-boosting.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        helper_code = """
def plot_decision_boundary(X, y, model, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', marker='o', edgecolors='k')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', marker='s', edgecolors='k')
    plt.title(title)
"""
        idx = find_cell_with_text(nb, "plot_decision_boundary(")
        if idx != -1:
             if "def plot_decision_boundary" not in nb.cells[idx].source:
                 nb.cells[idx].source = helper_code + "\n" + nb.cells[idx].source
                 print("Injected Plotting Helper locally.")
        save_notebook(nb, path)

    # --- Ex 2: ID3 (Fix information_gain corruption) ---
    path = os.path.join(base_path, 'notebooks/module3-trees/solutions/solution_exercise2-tree-algorithms.ipynb')
    if os.path.exists(path):
        nb = load_notebook(path)
        ig_code = """def information_gain(X, y, attribute_index):
    parent_entropy = entropy(y)
    if hasattr(X, 'values'): values_col = X[attribute_index].values
    elif isinstance(X, np.ndarray): values_col = X[:, attribute_index] if not isinstance(attribute_index, str) else X[:, 0]
    else: values_col = X[:, attribute_index]
    values, counts = np.unique(values_col, return_counts=True)
    weighted_child_entropy = 0
    for value, count in zip(values, counts):
        if hasattr(X, 'values'): child_y = y[X[attribute_index] == value]
        else: child_y = y[values_col == value]
        weighted_child_entropy += (count / len(y)) * entropy(child_y)
    return parent_entropy - weighted_child_entropy
"""
        idx = find_cell_with_text(nb, "def information_gain")
        if idx != -1:
            nb.cells[idx].source = ig_code
            print("Refreshed information_gain definition.")
        save_notebook(nb, path)

if __name__ == "__main__":
    fix_all_module3("c:\\dev\\python\\ML101")
