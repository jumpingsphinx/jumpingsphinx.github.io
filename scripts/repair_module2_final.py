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

def fix_gradient_descent(base_path):
    nb_path = os.path.join(base_path, 'notebooks/module2-regression/solutions/solution_exercise2-gradient-descent.ipynb')
    if not os.path.exists(nb_path): return
    print(f"Repairing {nb_path}...")
    nb = load_notebook(nb_path)
    
    # Fix indentation error in batch_gradient_descent
    # The error was extra indent on print statement inside loop
    # Also need to ensure 'cost' is defined before print if it's used
    # And 'cost' calculation seems missing in the loop in the snippet I saw?
    # Snippet:
    # # 2. Compute errors
    # errors = predictions - y
    # # 3. Compute gradients
    # gradients = ...
    # # 4. Update weights
    # weights = ...
    #     print(...) 
    # 'cost' is not calculated in the loop! The print uses it.
    
    idx = find_cell_with_text(nb, "def batch_gradient_descent(X, y")
    if idx != -1:
        nb.cells[idx].source = """def batch_gradient_descent(X, y, learning_rate=0.01, n_iterations=1000, verbose=False):
    \"\"\"
    Perform batch gradient descent.
    \"\"\"
    m, n = X.shape
    
    # Add bias column
    X_with_bias = np.c_[np.ones((m, 1)), X]
    
    # Initialize weights randomly
    weights = np.random.randn(n + 1) * 0.01
    
    cost_history = []
    
    for iteration in range(n_iterations):
        # 1. Compute predictions
        predictions = X_with_bias @ weights
        
        # 2. Compute errors
        errors = predictions - y
        
        # 3. Compute gradients: (1/m) * X^T * errors
        gradients = (X_with_bias.T @ errors) / m
        
        # 4. Update weights
        weights = weights - learning_rate * gradients
        
        # Calculate cost for history
        cost = np.sum(errors ** 2) / (2 * m)
        cost_history.append(cost)
        
        if verbose and iteration % 100 == 0:
            print(f"Iteration {iteration}: Cost = {cost:.4f}")
    
    return weights, cost_history

# Test on simple data
np.random.seed(42)
X_simple = np.random.rand(100, 1) * 10
y_simple = 3 + 2 * X_simple.squeeze() + np.random.randn(100) * 2

weights, cost_history = batch_gradient_descent(
    X_simple, y_simple, 
    learning_rate=0.01, 
    n_iterations=1000,
    verbose=False
)

print(f"\\nFinal weights: bias={weights[0]:.2f}, slope={weights[1]:.2f}")
print(f"True line:     bias=3.00, slope=2.00")
print(f"Final cost: {cost_history[-1]:.4f}")

# Plot cost history
plt.figure(figsize=(10, 5))
plt.plot(cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost vs Iteration (Batch Gradient Descent)')
plt.grid(True, alpha=0.3)
plt.show()

assert abs(weights[0] - 3) < 1, "Bias should be close to 3"
assert abs(weights[1] - 2) < 0.5, "Slope should be close to 2"
assert cost_history[-1] < cost_history[0], "Cost should decrease"
print("\\n✓ Batch gradient descent works!")"""

    # Also check Stochastic GD (Indentation or missing vars)
    idx = find_cell_with_text(nb, "def stochastic_gradient_descent(X, y")
    if idx != -1:
        nb.cells[idx].source = """def stochastic_gradient_descent(X, y, learning_rate=0.01, n_epochs=50, verbose=False):
    \"\"\"
    Perform stochastic gradient descent.
    \"\"\"
    m, n = X.shape
    
    # Add bias column
    X_with_bias = np.c_[np.ones((m, 1)), X]
    
    # Initialize weights
    weights = np.random.randn(n + 1) * 0.01
    
    cost_history = []
    
    for epoch in range(n_epochs):
        # Shuffle data
        indices = np.random.permutation(m)
        X_shuffled = X_with_bias[indices]
        y_shuffled = y[indices]
        
        epoch_cost = 0
        
        for i in range(m):
            xi = X_shuffled[i]
            yi = y_shuffled[i]
            
            # Compute prediction for this sample
            prediction = np.dot(xi, weights)
            
            # Compute error
            error = prediction - yi
            
            # Compute gradient for this sample
            gradient = xi * error
            
            # Update weights
            weights = weights - learning_rate * gradient
            
            epoch_cost += error ** 2
            
        cost = epoch_cost / (2 * m)
        cost_history.append(cost)
        
        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch}: Cost = {cost:.4f}")
    
    return weights, cost_history

# Test SGD
weights_sgd, cost_history_sgd = stochastic_gradient_descent(
    X_simple, y_simple,
    learning_rate=0.01,
    n_epochs=50,
    verbose=False
)

print(f"\\nSGD weights: bias={weights_sgd[0]:.2f}, slope={weights_sgd[1]:.2f}")"""

    # Also check Mini-batch GD (Indentation or missing vars)
    idx = find_cell_with_text(nb, "def minibatch_gradient_descent(X, y")
    if idx != -1:
        nb.cells[idx].source = """def minibatch_gradient_descent(X, y, learning_rate=0.01, n_epochs=50, batch_size=32, verbose=False):
    \"\"\"
    Perform mini-batch gradient descent.
    \"\"\"
    m, n = X.shape
    
    # Add bias column
    X_with_bias = np.c_[np.ones((m, 1)), X]
    
    # Initialize weights
    weights = np.random.randn(n + 1) * 0.01
    
    cost_history = []
    
    for epoch in range(n_epochs):
        # Shuffle data
        indices = np.random.permutation(m)
        X_shuffled = X_with_bias[indices]
        y_shuffled = y[indices]
        
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            batch_m = len(y_batch)
            
            # Compute predictions for batch
            predictions = X_batch @ weights
            
            # Compute errors
            errors = predictions - y_batch
            
            # Compute gradients (average over batch)
            gradients = (X_batch.T @ errors) / batch_m
            
            # Update weights
            weights = weights - learning_rate * gradients
            
        # Calculate cost for epoch
        predictions = X_with_bias @ weights
        cost = np.sum((predictions - y) ** 2) / (2 * m)
        cost_history.append(cost)
        
        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch}: Cost = {cost:.4f}")
    
    return weights, cost_history

# Test with different batch sizes
batch_sizes = [1, 16, 32, len(X_simple)]
print("Mini-batch GD implemented!")"""

    save_notebook(nb, nb_path)
    print("Repaired solution_exercise2-gradient-descent.ipynb")

def fix_logistic_regression(base_path):
    nb_path = os.path.join(base_path, 'notebooks/module2-regression/solutions/solution_exercise3-logistic-regression.ipynb')
    if not os.path.exists(nb_path): return
    print(f"Repairing {nb_path}...")
    nb = load_notebook(nb_path)
    
    # Fix Part 7: Wine Dataset (Empty assignments)
    idx = find_cell_with_text(nb, "# Your turn: Complete the pipeline")
    if idx != -1:
        nb.cells[idx].source = """# Load wine dataset
wine = load_wine()
X_wine = wine.data
y_wine = wine.target  # 3 classes: 0, 1, 2

# Convert to binary: class 0 vs rest (1 or 2)
y_wine_binary = (y_wine == 0).astype(int)

print("Wine Dataset (Binary):")
print(f"Shape: {X_wine.shape}")
print(f"Original classes: {wine.target_names}")
print(f"Binary: Class 0 ({wine.target_names[0]}) vs Rest")
print(f"Class distribution: {np.bincount(y_wine_binary)}")
print()

# Your turn: Complete the pipeline
# Split data
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
    X_wine, y_wine_binary, test_size=0.2, random_state=42, stratify=y_wine_binary
)

# Scale features
scaler_w = StandardScaler()
X_train_w_scaled = scaler_w.fit_transform(X_train_w)
X_test_w_scaled = scaler_w.transform(X_test_w)

# Train model
model_wine = LogisticRegression(learning_rate=0.1, n_iterations=2000, random_state=42)
model_wine.fit(X_train_w_scaled, y_train_w)

# Evaluate
acc_w = model_wine.score(X_test_w_scaled, y_test_w)
y_proba_w = model_wine.predict_proba(X_test_w_scaled)
auc_w = roc_auc_score(y_test_w, y_proba_w)

print(f"Test Accuracy: {acc_w:.4f}")
print(f"Test AUC:      {auc_w:.4f}")

# Print classification report
y_pred_w = model_wine.predict(X_test_w_scaled)
print("\\nClassification Report:")
print(classification_report(y_test_w, y_pred_w, 
                          target_names=['Class 1&2', 'Class 0']))

print("\\n✓ Successfully applied to wine dataset!")"""

    save_notebook(nb, nb_path)
    print("Repaired solution_exercise3-logistic-regression.ipynb")

def fix_regularization(base_path):
    nb_path = os.path.join(base_path, 'notebooks/module2-regression/solutions/solution_exercise4-regularization.ipynb')
    if not os.path.exists(nb_path): return
    print(f"Repairing {nb_path}...")
    nb = load_notebook(nb_path)
    
    # Fix Ridge Regression (reg_matrix NameError)
    idx = find_cell_with_text(nb, "class RidgeRegression:")
    if idx != -1:
        nb.cells[idx].source = """class RidgeRegression:
    def __init__(self, alpha=1.0):
        \"\"\"
        Ridge Regression using closed-form solution.
        \"\"\"
        self.alpha = alpha
        self.weights = None
    
    def fit(self, X, y):
        \"\"\"
        Fit Ridge regression.
        \"\"\"
        m, n = X.shape
        
        # Add bias column
        X_with_bias = np.c_[np.ones((m, 1)), X]
        
        # Create regularization matrix (identity with 0 at [0,0] to not penalize bias)
        reg_matrix = np.eye(n + 1)
        reg_matrix[0, 0] = 0
        
        # Compute weights
        # w = (X^T X + alpha*Reg)^-1 X^T y
        try:
            self.weights = np.linalg.inv(X_with_bias.T @ X_with_bias + self.alpha * reg_matrix) @ X_with_bias.T @ y
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse if singular
            self.weights = np.linalg.pinv(X_with_bias.T @ X_with_bias + self.alpha * reg_matrix) @ X_with_bias.T @ y
            
        return self
    
    def predict(self, X):
        X_with_bias = np.c_[np.ones((len(X), 1)), X]
        return X_with_bias @ self.weights
    
    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0: return 0.0
        return 1 - (ss_res / ss_tot)

# Test Ridge on polynomial features
degree = 9
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train_simple)
X_test_poly = poly.transform(X_test_simple)

# Compare different alpha values
alphas = [0, 0.1, 1.0, 10.0]

plt.figure(figsize=(16, 4))
for i, alpha in enumerate(alphas, 1):
    # Fit Ridge
    ridge = RidgeRegression(alpha=alpha)
    ridge.fit(X_train_poly, y_train_simple)
    
    # Predictions
    X_plot = np.linspace(0, 10, 200).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot)
    y_plot = ridge.predict(X_plot_poly)
    
    # Scores
    train_r2 = ridge.score(X_train_poly, y_train_simple)
    test_r2 = ridge.score(X_test_poly, y_test_simple)
    
    # Plot
    plt.subplot(1, 4, i)
    plt.scatter(X_train_simple, y_train_simple, alpha=0.6, label='Train')
    plt.scatter(X_test_simple, y_test_simple, alpha=0.6, color='red', label='Test')
    plt.plot(X_plot, y_plot, linewidth=2, label='Ridge')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'Ridge alpha={alpha}\\nTrain R2={train_r2:.3f}, Test R2={test_r2:.3f}')
    plt.grid(True, alpha=0.3)
    plt.ylim(-4, 4)

plt.tight_layout()
plt.show()

print("✓ Ridge regression implemented!")"""

    # Also check RidgeRegressionGD for similar issues?
    # It didn't error but good to check.
    # It calculates gradients manually, does not use reg_matrix.
    
    save_notebook(nb, nb_path)
    print("Repaired solution_exercise4-regularization.ipynb")

    # Fix indentation in Cross-Validation cell (Cell 8ish)
    # The error was: score = ... IndentationError
    idx = find_cell_with_text(nb, "best_alpha = alphas_cv[best_idx]")
    if idx != -1:
         nb.cells[idx].source = """# Cross-validation implementation
n_folds = 5
fold_size = len(X_train_simple) // n_folds
alphas_cv = [0.01, 0.1, 1.0, 10.0, 100.0]
mean_scores = []
std_scores = []

for alpha in alphas_cv:
    scores = []
    for i in range(n_folds):
        # Create folds
        start, end = i * fold_size, (i + 1) * fold_size
        X_val_fold = X_train_poly[start:end]
        y_val_fold = y_train_simple[start:end]
        
        X_train_fold = np.concatenate([X_train_poly[:start], X_train_poly[end:]])
        y_train_fold = np.concatenate([y_train_simple[:start], y_train_simple[end:]])
        
        # Train and evaluate
        ridge_cv = RidgeRegression(alpha=alpha)
        ridge_cv.fit(X_train_fold, y_train_fold)
        score = ridge_cv.score(X_val_fold, y_val_fold)
        scores.append(score)
    
    mean_scores.append(np.mean(scores))
    std_scores.append(np.std(scores))

best_idx = np.argmax(mean_scores)
best_alpha = alphas_cv[best_idx]
best_score = mean_scores[best_idx]

# Plot CV scores
plt.figure(figsize=(10, 6))
plt.errorbar(alphas_cv, mean_scores, yerr=std_scores, marker='o', capsize=5)
plt.axvline(x=best_alpha, color='r', linestyle='--', 
           label=f'Best \u03b1={best_alpha:.3f} (R\u00b2={best_score:.3f})')
plt.xscale('log')
plt.xlabel('Regularization Strength (\u03b1)')
plt.ylabel('Mean Cross-Validation R\u00b2')
plt.title('Cross-Validation for Ridge Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Best alpha: {best_alpha:.4f}")
print(f"Best CV R\u00b2: {best_score:.4f} \u00b1 {std_scores[best_idx]:.4f}")

# Train final model with best alpha
ridge_final = RidgeRegression(alpha=best_alpha)
ridge_final.fit(X_train_poly, y_train_simple)
test_r2 = ridge_final.score(X_test_poly, y_test_simple)

print(f"Test R\u00b2 with best alpha: {test_r2:.4f}")
print("\\n✓ Cross-validation complete!")"""
         save_notebook(nb, nb_path)
         print("Repaired CV loop in solution_exercise4-regularization.ipynb")

if __name__ == "__main__":
    base_path = "c:\\dev\\python\\ML101"
    fix_gradient_descent(base_path)
    fix_logistic_regression(base_path)
    fix_regularization(base_path)
