# Module 2 Exercises: Regression Algorithms

## Overview

Ready to build your first machine learning models? These exercises will guide you through implementing regression algorithms from scratch, understanding gradient descent optimization, and applying these techniques to real-world datasets.

## Before You Start

### Setup

1. Create the notebooks directory if it doesn't exist:
   ```bash
   mkdir -p notebooks/module2-regression
   ```

2. Open Jupyter Lab:
   ```bash
   jupyter lab
   ```

3. Navigate to `notebooks/module2-regression/`

4. Start with `exercise1-linear-regression.ipynb`

### Exercise Format

Each exercise includes:
- **Learning objectives**: What you'll practice
- **Background**: Quick concept review
- **Tasks**: Step-by-step implementation
- **Hints**: Help when you're stuck
- **Visualizations**: See your models in action
- **Validation**: Compare with sklearn implementations
- **Reflection questions**: Deepen your understanding

### Tips for Success

!!! tip "Best Practices"
    - **Implement from scratch first**: Understanding beats memorization
    - **Visualize everything**: Plot data, predictions, loss curves
    - **Check dimensions**: Use `.shape` to debug
    - **Start simple**: Test on toy data before real datasets
    - **Compare with sklearn**: Validate your implementation

## Exercise 1: Linear Regression from Scratch

<a href="https://colab.research.google.com/github/jumpingsphinx/jumpingsphinx.github.io/blob/main/notebooks/module2-regression/exercise1-linear-regression.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

**Time:** 2-3 hours

### What You'll Learn

- Implement linear regression using the normal equation
- Fit models to real datasets
- Make predictions and evaluate performance
- Visualize regression lines and residuals
- Handle multiple features (multivariate regression)
- Compare your implementation with sklearn

### Topics Covered

- Simple linear regression (one feature)
- Multiple linear regression (many features)
- Normal equation: w = (X^T X)^(-1) X^T y
- Mean Squared Error (MSE) and R² metrics
- Residual analysis
- Train/test split
- Feature scaling

### Key Functions

```python
np.linalg.lstsq(), np.linalg.inv(),
sklearn.linear_model.LinearRegression,
sklearn.metrics.mean_squared_error, r2_score
```

### Datasets Used

- California Housing (regression with multiple features)
- Diabetes dataset (10 features)
- Custom synthetic data for visualization

### What You'll Build

1. **Linear regression class**: Complete implementation from scratch
2. **Model evaluator**: Calculate MSE, RMSE, R², MAE
3. **Residual plotter**: Visualize prediction errors
4. **Comparison tool**: Validate against sklearn

---

## Exercise 2: Gradient Descent Optimization

<a href="https://colab.research.google.com/github/jumpingsphinx/jumpingsphinx.github.io/blob/main/notebooks/module2-regression/exercise2-gradient-descent.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

**Time:** 2-3 hours

### What You'll Learn

- Implement batch gradient descent from scratch
- Implement stochastic gradient descent (SGD)
- Implement mini-batch gradient descent
- Tune learning rate and visualize its effect
- Understand convergence criteria
- Compare different optimization strategies

### Topics Covered

- Gradient computation for linear regression
- Batch gradient descent
- Stochastic gradient descent (SGD)
- Mini-batch gradient descent
- Learning rate selection
- Convergence analysis
- Loss curve visualization
- Feature normalization importance

### Key Concepts

```python
# Gradient: ∇J(w) = (1/m) X^T (Xw - y)
# Update rule: w = w - α ∇J(w)
# Learning rate: α
```

### What You'll Build

1. **Gradient descent variants**: Batch, SGD, mini-batch implementations
2. **Learning rate finder**: Experiment with different α values
3. **Convergence visualizer**: Plot loss over iterations
4. **Optimizer comparison**: Compare speed and accuracy of methods

---

## Exercise 3: Logistic Regression for Classification

<a href="https://colab.research.google.com/github/jumpingsphinx/jumpingsphinx.github.io/blob/main/notebooks/module2-regression/exercise3-logistic-regression.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

**Time:** 2-3 hours

### What You'll Learn

- Implement logistic regression from scratch
- Understand the sigmoid function
- Compute logistic loss (binary cross-entropy)
- Use gradient descent for optimization
- Interpret probabilities and make predictions
- Evaluate classification performance

### Topics Covered

- Sigmoid activation function
- Binary cross-entropy loss
- Gradient descent for logistic regression
- Decision boundaries
- Probability predictions vs class predictions
- Classification metrics: accuracy, precision, recall, F1
- Confusion matrix
- ROC curve and AUC

### Key Functions

```python
sigmoid(z) = 1 / (1 + e^(-z))
Loss = -(1/m) Σ [y log(h(x)) + (1-y) log(1-h(x))]
sklearn.linear_model.LogisticRegression
sklearn.metrics.classification_report
```

### Datasets Used

- Breast Cancer Wisconsin (binary classification)
- Iris dataset (binary: setosa vs others)
- Wine dataset (binary classification)

### What You'll Build

1. **Logistic regression class**: Complete implementation
2. **Decision boundary visualizer**: Plot classification regions
3. **Probability calibrator**: Analyze prediction confidence
4. **Performance evaluator**: Confusion matrix, ROC curve, metrics

---

## Exercise 4: Regularization Techniques

<a href="https://colab.research.google.com/github/jumpingsphinx/jumpingsphinx.github.io/blob/main/notebooks/module2-regression/exercise4-regularization.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

**Time:** 2-3 hours

### What You'll Learn

- Implement L1 regularization (Lasso)
- Implement L2 regularization (Ridge)
- Implement Elastic Net (L1 + L2)
- Understand bias-variance tradeoff
- Tune regularization strength (λ/α)
- Apply regularization to prevent overfitting

### Topics Covered

- Ridge regression (L2 regularization)
- Lasso regression (L1 regularization)
- Elastic Net (combined L1 + L2)
- Regularization strength parameter (α/λ)
- Feature selection with Lasso
- Cross-validation for hyperparameter tuning
- Overfitting vs underfitting
- Regularization path visualization

### Key Concepts

```python
# Ridge: J(w) = MSE + α Σ w²
# Lasso: J(w) = MSE + α Σ |w|
# Elastic Net: J(w) = MSE + α₁ Σ |w| + α₂ Σ w²
```

### Key Functions

```python
sklearn.linear_model.Ridge,
sklearn.linear_model.Lasso,
sklearn.linear_model.ElasticNet,
sklearn.model_selection.cross_val_score
```

### Datasets Used

- Boston Housing (with polynomial features for overfitting demo)
- Diabetes dataset (high-dimensional regression)

### What You'll Build

1. **Ridge regression**: L2 implementation from scratch
2. **Lasso regression**: L1 implementation (using optimization library)
3. **Regularization path plotter**: Visualize coefficient shrinkage
4. **Cross-validator**: Find optimal regularization strength

---

## Solutions

After completing each exercise, review the solutions to:
- Compare your approach with the reference implementation
- Learn alternative methods
- Understand best practices
- Debug any issues

!!! info "Solutions Coming Soon"
    Solution notebooks will be added soon. Work through the exercises independently first!

!!! warning "Use Solutions Wisely"
    Try to complete exercises independently first. Looking at solutions too early prevents deep learning. Use them for verification and learning alternative approaches.

## Assessment Questions

After completing all exercises, test your understanding:

### Conceptual

1. **When would you use the normal equation vs gradient descent?**
   - Consider dataset size, invertibility, computational cost

2. **Why does gradient descent require feature scaling?**
   - Think about how different feature magnitudes affect convergence

3. **What's the difference between MSE and cross-entropy loss?**
   - When is each appropriate?

4. **How does regularization prevent overfitting?**
   - What's happening to the weights?

5. **Why does Lasso perform feature selection but Ridge doesn't?**
   - Think about the geometry of L1 vs L2 penalties

### Practical

1. **How do you choose a learning rate?**
   - What happens if it's too large or too small?

2. **When should you use Ridge vs Lasso vs Elastic Net?**
   - Consider feature correlations and sparsity needs

3. **How do you diagnose overfitting in regression?**
   - What metrics and visualizations help?

4. **Why might logistic regression fail?**
   - Consider linearly separable vs non-separable data

### Debugging

1. **Gradient descent diverges (loss increases). What's wrong?**
   - Likely learning rate too high, or features not normalized

2. **Your model has R² = 1.0 on training data. Good or bad?**
   - Probably overfitting! Check test performance

3. **Logistic regression predicts 0.5 for everything. Why?**
   - Likely uninitialized weights or broken gradient

## Reflection Questions

Think deeply about what you've learned:

1. How are linear and logistic regression similar? Different?
2. What surprised you most about gradient descent?
3. When would you NOT use regularization?
4. How does this relate to neural networks?
5. What real-world problems could you solve with these techniques?

## Common Mistakes to Avoid

!!! warning "Watch Out For"
    - **Forgetting to add bias term**: Always add column of 1s to X
    - **Not normalizing features**: Gradient descent won't converge well
    - **Wrong loss function**: MSE for regression, cross-entropy for classification
    - **Data leakage**: Normalize using training stats only, apply to test
    - **Ignoring regularization**: Always try it, especially with many features
    - **Wrong gradient**: Double-check your calculus!

## Going Further

### Challenge Exercises

Want more practice? Try these:

1. **Polynomial regression**: Extend linear regression with polynomial features
2. **Coordinate descent for Lasso**: Implement Lasso without optimization libraries
3. **Multinomial logistic regression**: Extend to multi-class classification
4. **Early stopping**: Implement regularization through early stopping
5. **Feature engineering**: Create new features and test their impact

### Real-World Projects

Apply your skills:

1. **House price prediction**: Build a practical pricing model
2. **Medical diagnosis**: Binary classification for disease detection
3. **Customer churn**: Predict which customers will leave
4. **Spam detection**: Classify emails as spam/ham
5. **Credit scoring**: Predict loan default risk

## Next Module

Once you're comfortable with regression, you're ready for decision trees and ensemble methods!

[Continue to Module 3: Decision Trees and Ensembles](../module3-trees/index.md){ .md-button .md-button--primary }

---

## Help and Support

**Stuck on an exercise?**

1. Re-read the relevant lesson
2. Check the hints in the notebook
3. Print intermediate values to debug
4. Search the error message online
5. Look at the solution for that specific part
6. Open an issue on [GitHub](https://github.com/jumpingsphinx/jumpingsphinx.github.io/issues)

**Found a bug or have a suggestion?**

Please [open an issue](https://github.com/jumpingsphinx/jumpingsphinx.github.io/issues) or submit a pull request!

---

Good luck with the exercises! Remember: **implementation is the key to understanding**. Don't just read—code!
