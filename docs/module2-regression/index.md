# Module 2: Regression Algorithms

## Overview

Regression is one of the most fundamental tasks in machine learning - predicting a continuous value based on input features. In this module, you'll learn regression algorithms from the ground up, starting with simple linear regression and progressing to advanced regularization techniques.

## Why Regression Matters in Machine Learning

Regression algorithms are everywhere:
- **Predicting prices**: House prices, stock prices, product demand
- **Forecasting**: Sales, weather, energy consumption
- **Risk assessment**: Credit scores, insurance premiums
- **Scientific modeling**: Relationships between variables
- **Foundation for other algorithms**: Many ML techniques build on regression

!!! quote "Core ML Skill"
    "Linear regression might seem simple, but it's the foundation of modern deep learning. Every neural network performs learned linear transformations followed by non-linearities."

## Learning Objectives

By the end of this module, you will be able to:

- ✅ Implement linear regression from scratch using NumPy
- ✅ Understand and implement gradient descent optimization
- ✅ Apply logistic regression for classification problems
- ✅ Use L1 and L2 regularization to prevent overfitting
- ✅ Evaluate model performance with appropriate metrics
- ✅ Use scikit-learn for production-ready implementations

## Prerequisites

- **Module 1 completed**: Vector and matrix operations
- **Python**: Comfortable with NumPy and basic plotting
- **Basic calculus**: Understanding derivatives helps but not required

## Module Structure

### Lesson 1: Linear Regression
**Time: 60 minutes**

Learn the foundation of supervised learning with linear regression.

- Problem formulation and assumptions
- Simple linear regression (one feature)
- Multiple linear regression
- Normal equation (closed-form solution)
- Implementation with NumPy and scikit-learn
- Model evaluation metrics (MSE, MAE, R²)

[Start Lesson 1](01-linear-regression.md){ .md-button .md-button--primary }

---

### Lesson 2: Gradient Descent
**Time: 60 minutes**

Master the optimization algorithm that powers most of machine learning.

- Cost functions (MSE, MAE, RMSE)
- Gradient descent intuition and mathematics
- Batch, stochastic, and mini-batch variants
- Learning rate selection and tuning
- Convergence criteria and monitoring
- Visualization of optimization landscape

[Start Lesson 2](02-gradient-descent.md){ .md-button }

---

### Lesson 3: Logistic Regression
**Time: 60 minutes**

Extend regression to classification problems with logistic regression.

- From regression to classification
- Sigmoid function and probability interpretation
- Log loss (binary cross-entropy)
- Gradient descent for logistic regression
- Multi-class classification (One-vs-Rest, Softmax)
- Decision boundaries and visualization

[Start Lesson 3](03-logistic-regression.md){ .md-button }

---

### Lesson 4: Regularization
**Time: 60 minutes**

Learn to control model complexity and prevent overfitting.

- Overfitting and underfitting
- Bias-variance tradeoff
- L1 regularization (Lasso) for feature selection
- L2 regularization (Ridge) for coefficient shrinkage
- Elastic Net (combining L1 and L2)
- Hyperparameter tuning with cross-validation

[Start Lesson 4](04-regularization.md){ .md-button }

---

### Exercises
**Time: 6-8 hours**

Apply what you've learned through hands-on implementation.

- Exercise 1: Linear regression from scratch and with sklearn
- Exercise 2: Gradient descent visualization and tuning
- Exercise 3: Logistic regression for binary and multi-class classification
- Exercise 4: Comparing regularization techniques

[View Exercises](exercises.md){ .md-button }

## Key Concepts

| Concept | Description | Formula |
|---------|-------------|---------|
| **Linear Model** | Weighted sum of features | $\hat{y} = w^Tx + b$ |
| **Cost Function** | Measure of prediction error | $J(w) = \frac{1}{2m}\sum(y - \hat{y})^2$ |
| **Gradient Descent** | Iterative optimization | $w := w - \alpha\nabla J(w)$ |
| **Sigmoid** | Squash to probability | $\sigma(z) = \frac{1}{1 + e^{-z}}$ |
| **L2 Regularization** | Penalty on weights | $J(w) + \frac{\lambda}{2}\|w\|^2$ |

## What You'll Build

By the end of this module, you'll have implemented:

1. **Linear Regression from Scratch**: Using only NumPy
2. **Gradient Descent Visualizer**: See optimization in action
3. **Logistic Regression Classifier**: Binary and multi-class
4. **Regularization Comparison Tool**: Compare Lasso, Ridge, Elastic Net
5. **Complete ML Pipeline**: Data preprocessing → Training → Evaluation

## Datasets You'll Use

- **California Housing**: Predict house prices (regression)
- **Boston Housing**: Classic regression dataset
- **Iris Flowers**: Multi-class classification
- **Breast Cancer Wisconsin**: Binary classification
- **Synthetic Data**: Understand algorithms with controlled examples

## Tips for Success

!!! tip "Understanding Loss Functions"
    Plot your cost function over iterations. A decreasing curve means learning is working!

!!! tip "Feature Scaling"
    Always normalize/standardize features before applying gradient descent. It makes convergence much faster.

!!! tip "Start Simple"
    Implement algorithms on simple 2D data first, then scale to real datasets.

!!! warning "Common Pitfall"
    Learning rate too high → divergence. Too low → slow convergence. Plot and experiment!

## Estimated Time

- **Reading lessons:** 4-5 hours
- **Completing exercises:** 6-8 hours
- **Total:** 10-13 hours

## Real-World Applications

After this module, you'll understand the algorithms behind:

- **Price Prediction**: Real estate, e-commerce, financial markets
- **Demand Forecasting**: Inventory management, resource planning
- **Risk Modeling**: Credit scoring, insurance pricing
- **Medical Diagnosis**: Disease risk prediction
- **Marketing**: Customer conversion probability
- **Recommendation**: User preference prediction

## From Theory to Practice

This module emphasizes implementation:

1. **Understand the math**: Learn the equations
2. **Code from scratch**: Implement with NumPy
3. **Use libraries**: Apply scikit-learn for production
4. **Compare results**: Verify your implementation matches sklearn
5. **Apply to real data**: Work with actual datasets

## Ready to Start?

Let's begin with the foundation of supervised learning - linear regression!

[Start Lesson 1: Linear Regression](01-linear-regression.md){ .md-button .md-button--primary }

[Or review Module 1 first](../module1-linear-algebra/index.md){ .md-button }

---

**Questions?** Open an issue on [GitHub](https://github.com/jumpingsphinx/ML101/issues).
