import nbformat
import os

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

def repair_vectors_notebook(base_path):
    nb_path = os.path.join(base_path, 'notebooks/module1-linear-algebra/solutions/solution_exercise1-vectors.ipynb')
    if not os.path.exists(nb_path):
        print(f"File not found: {nb_path}")
        return

    print(f"Repairing {nb_path}...")
    nb = load_notebook(nb_path)

    # 1. Fix Dot Product
    idx = find_cell_with_text(nb, "# Method 1: Manual calculation")
    if idx != -1:
        nb.cells[idx].source = """x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

# Method 1: Manual calculation
dot_manual = np.sum(x * y)

# Method 2: Using NumPy
dot_numpy = np.dot(x, y)

print(f"Manual dot product: {dot_manual}")
print(f"NumPy dot product: {dot_numpy}")

# Verify
assert dot_manual == 32, "Manual calculation should be 32"
assert dot_numpy == 32, "NumPy calculation should be 32"
assert dot_manual == dot_numpy, "Both methods should match"
print("\\n✓ Dot product calculated correctly!")"""

    # 2. Fix Cosine Similarity
    idx = find_cell_with_text(nb, "def cosine_similarity(a, b):")
    if idx != -1:
        nb.cells[idx].source = """def cosine_similarity(a, b):
    \"\"\"
    Calculate cosine similarity between two vectors.
    
    Parameters:
    -----------
    a : np.ndarray
        First vector
    b : np.ndarray
        Second vector
    
    Returns:
    --------
    float
        Cosine similarity between -1 and 1
    \"\"\"
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

# Test cases
vec1 = np.array([1, 2, 3])
vec2 = np.array([2, 4, 6])  # Same direction, scaled
vec3 = np.array([1, 0, 0])  # Orthogonal to [0, 1, 0]
vec4 = np.array([0, 1, 0])

print(f"Similarity between {vec1} and {vec2}: {cosine_similarity(vec1, vec2):.4f}")
print(f"Similarity between {vec3} and {vec4}: {cosine_similarity(vec3, vec4):.4f}")

# Verify
assert np.isclose(cosine_similarity(vec1, vec2), 1.0), "Parallel vectors should have similarity 1"
assert np.isclose(cosine_similarity(vec3, vec4), 0.0), "Orthogonal vectors should have similarity 0"
print("\\n✓ Cosine similarity implemented correctly!")"""

    # 3. Fix Normalize
    idx = find_cell_with_text(nb, "def normalize(v):")
    if idx != -1:
        nb.cells[idx].source = """def normalize(v):
    \"\"\"
    Normalize a vector to unit length.
    
    Parameters:
    -----------
    v : np.ndarray
        Input vector
    
    Returns:
    --------
    np.ndarray
        Normalized vector with magnitude 1
    \"\"\"
    norm = np.linalg.norm(v)
    if norm == 0:
        return np.zeros_like(v)
    return v / norm

# Test
test_vec = np.array([3, 4])
normalized = normalize(test_vec)
print(f"Original: {test_vec}, Magnitude: {np.linalg.norm(test_vec)}")
print(f"Normalized: {normalized}, Magnitude: {np.linalg.norm(normalized)}")

assert np.isclose(np.linalg.norm(normalized), 1.0), "Normalized vector should have magnitude 1" """

    # 4. Fix Project
    idx = find_cell_with_text(nb, "def project(a, b):")
    if idx != -1:
        nb.cells[idx].source = """def project(a, b):
    \"\"\"
    Project vector a onto vector b.
    
    Parameters:
    -----------
    a : np.ndarray
        Vector to project
    b : np.ndarray
        Vector to project onto
    
    Returns:
    --------
    np.ndarray
        Projection of a onto b
    \"\"\"
    dot_val = np.dot(a, b)
    b_dot_b = np.dot(b, b)
    if b_dot_b == 0:
        return np.zeros_like(b)
    return (dot_val / b_dot_b) * b

# Test
a = np.array([3, 4])
b = np.array([1, 0])
proj = project(a, b)
print(f"Projection of {a} onto {b}: {proj}")

assert np.allclose(proj, [3, 0]), "Projection should be [3, 0]" """

    save_notebook(nb, nb_path)
    print("Repaired solution_exercise1-vectors.ipynb")

def repair_matrices_notebook(base_path):
    nb_path = os.path.join(base_path, 'notebooks/module1-linear-algebra/solutions/solution_exercise2-matrices.ipynb')
    if not os.path.exists(nb_path):
        print(f"File not found: {nb_path}")
        return

    print(f"Repairing {nb_path}...")
    nb = load_notebook(nb_path)

    # 1. Fix Transformations (Cell 314ish)
    idx = find_cell_with_text(nb, "# Rotation matrix for 90 degrees counterclockwise")
    if idx != -1:
        nb.cells[idx].source = """# Rotation matrix for 90 degrees counterclockwise
# [cos(θ), -sin(θ)]
# [sin(θ),  cos(θ)]
theta = np.pi / 2  # 90 degrees in radians
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])

# Scaling matrix
# [sx, 0 ]
# [0,  sy]
scaling_matrix = np.array([
    [2, 0],
    [0, 0.5]
])

# Original points (as columns)
points = np.array([[1, 2, 2, 1],
                   [1, 1, 2, 2]])

# Apply transformations
rotated_points = rotation_matrix @ points
scaled_points = scaling_matrix @ points

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original
axes[0].plot(points[0, :], points[1, :], 'bo-', linewidth=2, markersize=8)
axes[0].set_xlim(-1, 5)
axes[0].set_ylim(-1, 5)
axes[0].grid(True, alpha=0.3)
axes[0].set_aspect('equal')
axes[0].set_title('Original')

# Rotated
axes[1].plot(rotated_points[0, :], rotated_points[1, :], 'ro-', linewidth=2, markersize=8)
axes[1].set_xlim(-5, 5)
axes[1].set_ylim(-1, 5)
axes[1].grid(True, alpha=0.3)
axes[1].set_aspect('equal')
axes[1].set_title('Rotated 90°')

# Scaled
axes[2].plot(scaled_points[0, :], scaled_points[1, :], 'go-', linewidth=2, markersize=8)
axes[2].set_xlim(-1, 5)
axes[2].set_ylim(-1, 5)
axes[2].grid(True, alpha=0.3)
axes[2].set_aspect('equal')
axes[2].set_title('Scaled (2x, 0.5x)')

plt.tight_layout()
plt.show()

print("Rotation matrix:\\n", rotation_matrix)
print("\\nScaling matrix:\\n", scaling_matrix)"""

    # 2. Fix Dataset Preprocessing (Cell 406ish)
    idx = find_cell_with_text(nb, "# Sample dataset: 5 samples, 3 features")
    if idx != -1:
        nb.cells[idx].source = """# Sample dataset: 5 samples, 3 features
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9],
                 [10, 11, 12],
                 [13, 14, 15]])

# Your code here
feature_means = data.mean(axis=0)
centered_data = data - feature_means
feature_stds = centered_data.std(axis=0)
normalized_data = centered_data / feature_stds

normalized_means = normalized_data.mean(axis=0)
normalized_stds = normalized_data.std(axis=0)

print("Original data:\\n", data)
print("\\nFeature means:", feature_means)
print("\\nCentered data:\\n", centered_data)
print("\\nFeature standard deviations:", feature_stds)
print("\\nNormalized data:\\n", normalized_data)
print("\\nNormalized means:", normalized_means)
print("Normalized stds:", normalized_stds)

# Verify
assert np.allclose(feature_means, [7, 8, 9]), "Feature means incorrect"
assert np.allclose(centered_data.mean(axis=0), [0, 0, 0], atol=1e-10), "Centered data should have mean 0"
assert np.allclose(normalized_means, [0, 0, 0], atol=1e-10), "Normalized data should have mean ≈ 0"
assert np.allclose(normalized_stds, [1, 1, 1], atol=1e-10), "Normalized data should have std ≈ 1"
print("\\n✓ Data preprocessing correct!")"""

    # 3. Fix Linear System (Cell 486ish)
    idx = find_cell_with_text(nb, "# Coefficient matrix A")
    if idx != -1:
        nb.cells[idx].source = """# Coefficient matrix A
A = np.array([[2, 3],
              [3, 4]])

# Right-hand side vector b
b = np.array([8, 11])

# Method 1: Using solve
x_solve = np.linalg.solve(A, b)

# Method 2: Using inverse
A_inv = np.linalg.inv(A)
x_inverse = A_inv @ b

print("Coefficient matrix A:\\n", A)
print("\\nRight-hand side b:", b)
print("\\nSolution (using solve):", x_solve)
print("Solution (using inverse):", x_inverse)

# Verify solution
verification = A @ x_solve
print("\\nVerification (A @ x):", verification)
print("Should equal b:", b)

# Assertions
assert np.allclose(x_solve, x_inverse), "Both methods should give same result"
assert np.allclose(A @ x_solve, b), "Solution should satisfy Ax = b"
print("\\n✓ Linear system solved correctly!")"""

    # 4. Fix Matrix Properties (Cell 539ish)
    idx = find_cell_with_text(nb, "M = np.array([[1, 2, 3],")
    if idx != -1:
        nb.cells[idx].source = """M = np.array([[1, 2, 3],
              [0, 1, 4],
              [5, 6, 0]])

# Your code here
determinant = np.linalg.det(M)
trace = np.trace(M)
rank = np.linalg.matrix_rank(M)
is_invertible = abs(determinant) > 1e-10

print(f"Determinant: {determinant:.2f}")
print(f"Trace: {trace}")
print(f"Rank: {rank}")
print(f"Is invertible? {is_invertible}")

# Verify
assert np.isclose(trace, 2.0), "Trace should be 2 (sum of diagonal)"
assert rank == 3, "Full rank matrix should have rank 3"
assert is_invertible, "Matrix with non-zero determinant is invertible"
print("\\n✓ Matrix properties calculated correctly!")"""

    # 5. Fix Matrix Power (Challenge 1)
    idx = find_cell_with_text(nb, "def matrix_power(A, n):")
    if idx != -1:
        nb.cells[idx].source = """def matrix_power(A, n):
    \"\"\"
    Compute A^n (matrix power).
    
    Parameters:
    -----------
    A : np.ndarray
        Square matrix
    n : int
        Exponent (positive integer)
    
    Returns:
    --------
    np.ndarray
        A multiplied by itself n times
    \"\"\"
    if n == 0:
        return np.eye(A.shape[0])
    if n == 1:
        return A
    result = A.copy()
    for _ in range(n - 1):
        result = result @ A
    return result

# Test
A = np.array([[1, 2],
              [3, 4]])
A_cubed = matrix_power(A, 3)
A_cubed_numpy = np.linalg.matrix_power(A, 3)

print("A^3 (your implementation):\\n", A_cubed)
print("\\nA^3 (NumPy):\\n", A_cubed_numpy)

assert np.allclose(A_cubed, A_cubed_numpy), "Should match NumPy's result" """

    # 6. Fix Gram Schmidt (Challenge 2)
    idx = find_cell_with_text(nb, "def gram_schmidt(vectors):")
    if idx != -1:
        nb.cells[idx].source = """def gram_schmidt(vectors):
    \"\"\"
    Apply Gram-Schmidt orthogonalization.
    
    Parameters:
    -----------
    vectors : np.ndarray
        Matrix where each column is a vector
    
    Returns:
    --------
    np.ndarray
        Matrix with orthonormal columns
    \"\"\"
    vectors = vectors.astype(float)
    orthonormal = np.zeros_like(vectors)
    count = vectors.shape[1]
    
    for i in range(count):
        # Start with original vector
        v = vectors[:, i]
        
        # Subtract projection onto previous vectors
        for j in range(i):
            u = orthonormal[:, j]
            # Projection of v onto u: (v.u) * u
            v = v - np.dot(v, u) * u
            
        # Normalize
        if np.linalg.norm(v) > 1e-10:
             orthonormal[:, i] = v / np.linalg.norm(v)
             
    return orthonormal

# Test with 3 vectors in R^3
vectors = np.array([[1, 1, 0],
                    [1, 0, 1],
                    [0, 1, 1]], dtype=float).T

orthonormal = gram_schmidt(vectors)

print("Original vectors:\\n", vectors)
print("\\nOrthonormal vectors:\\n", orthonormal)

# Verify orthonormality
gram_matrix = orthonormal.T @ orthonormal
print("\\nGram matrix (should be identity):\\n", gram_matrix)

assert np.allclose(gram_matrix, np.eye(3), atol=1e-10), "Vectors should be orthonormal" """

    # 7. Fix Linear Regression (Challenge 3)
    idx = find_cell_with_text(nb, "def linear_regression(X, y):")
    if idx != -1:
        nb.cells[idx].source = """def linear_regression(X, y):
    \"\"\"
    Fit linear regression using normal equation.
    
    Parameters:
    -----------
    X : np.ndarray
        Design matrix (n_samples, n_features)
    y : np.ndarray
        Target values (n_samples,)
    
    Returns:
    --------
    np.ndarray
        Optimal weights (n_features,)
    \"\"\"
    # Normal equation: w = (X^T X)^(-1) X^T y
    return np.linalg.inv(X.T @ X) @ X.T @ y

# Test
X = np.array([[1, 1], [1, 2], [1, 3]])
y = np.array([2, 3, 4])
# Expected: y = 1 + x, so w = [1, 1]
w = linear_regression(X, y)
print(f"Weights: {w}")
assert np.allclose(w, [1, 1]), "Should recover weights [1, 1]" """

    save_notebook(nb, nb_path)
    print("Repaired solution_exercise2-matrices.ipynb")

if __name__ == "__main__":
    base_path = "c:\\dev\\python\\ML101"
    repair_vectors_notebook(base_path)
    repair_matrices_notebook(base_path)
