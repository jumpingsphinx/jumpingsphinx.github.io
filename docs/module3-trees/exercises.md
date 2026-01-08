# Module 3 Exercises: Tree-Based Algorithms

## Overview

These exercises provide hands-on practice with tree-based machine learning algorithms, from basic decision trees to advanced ensemble methods like Random Forest, Gradient Boosting, and XGBoost. Each exercise notebook corresponds to a lesson in the module and includes both foundational implementations and real-world applications.

## Learning Objectives

By completing these exercises, you will:

- ✅ Implement decision trees from scratch and understand splitting criteria
- ✅ Build and compare different tree algorithms (ID3, C4.5, CART)
- ✅ Master Random Forest and understand the power of ensemble learning
- ✅ Apply boosting algorithms (AdaBoost, Gradient Boosting) to real problems
- ✅ Use XGBoost for high-performance machine learning
- ✅ Tune hyperparameters systematically for optimal performance
- ✅ Interpret models using feature importance and visualization
- ✅ Handle real-world challenges like imbalanced data and missing values

## Exercise Structure

Each exercise notebook contains:

1. **Learning Objectives**: Clear goals for the exercise
2. **Setup**: Import required libraries and load datasets
3. **Guided Exercises**: Step-by-step problems with hints
4. **Challenges**: More difficult problems to test your understanding
5. **Real-World Application**: Apply concepts to practical datasets
6. **Reflection Questions**: Deepen your conceptual understanding

## Exercises

### Exercise 1: Decision Trees (2-3 hours)

**Topics Covered:**
- Building decision trees from scratch
- Understanding Gini impurity and entropy
- Implementing splitting criteria
- Visualizing decision trees
- Analyzing overfitting and pruning

**Datasets:**
- Iris (simple classification)
- Titanic (binary classification with mixed features)
- California Housing (regression)

**Key Skills:**
- Calculate information gain manually
- Implement recursive tree building
- Visualize tree structure and decision boundaries
- Compare depth vs performance

<a href="https://colab.research.google.com/github/jumpingsphinx/jumpingsphinx.github.io/blob/main/notebooks/module3-trees/exercise1-decision-trees.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

---

### Exercise 2: Tree Algorithms (2-3 hours)

**Topics Covered:**
- Implementing ID3 algorithm
- Understanding C4.5 improvements (gain ratio, continuous features)
- Applying CART algorithm
- Comparing different tree algorithms
- Handling missing values

**Datasets:**
- Play Tennis (categorical features)
- Wine Quality (multi-class classification)
- Breast Cancer (binary classification)

**Key Skills:**
- Implement ID3 from scratch for categorical data
- Calculate gain ratio and understand its benefits
- Apply scikit-learn's CART implementation
- Compare algorithm performance
- Handle missing values in tree construction

<a href="https://colab.research.google.com/github/jumpingsphinx/jumpingsphinx.github.io/blob/main/notebooks/module3-trees/exercise2-tree-algorithms.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

---

### Exercise 3: Random Forest (3-4 hours)

**Topics Covered:**
- Understanding bootstrap aggregating (bagging)
- Implementing Random Forest from scratch
- Feature importance analysis
- Out-of-bag (OOB) error estimation
- Comparing Random Forest with single trees

**Datasets:**
- Breast Cancer (classification)
- Wine Quality (multi-class)
- California Housing (regression)
- Credit Default (imbalanced classification)

**Key Skills:**
- Implement bagging manually
- Build Random Forest from components
- Analyze feature importance (MDI vs permutation)
- Use OOB error for model selection
- Tune Random Forest hyperparameters
- Handle imbalanced data

<a href="https://colab.research.google.com/github/jumpingsphinx/jumpingsphinx.github.io/blob/main/notebooks/module3-trees/exercise3-random-forest.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

---

### Exercise 4: Boosting (3-4 hours)

**Topics Covered:**
- Implementing AdaBoost from scratch
- Understanding weight updates and weak learners
- Applying Gradient Boosting
- Comparing boosting with bagging
- Tuning learning rate and number of estimators

**Datasets:**
- Make Moons (non-linear classification)
- Digits (multi-class classification)
- Boston Housing (regression)
- Credit Card Fraud (imbalanced classification)

**Key Skills:**
- Implement AdaBoost algorithm step-by-step
- Understand sample reweighting
- Apply Gradient Boosting for regression and classification
- Visualize learning progression
- Compare AdaBoost, Gradient Boosting, and Random Forest
- Tune boosting hyperparameters

<a href="https://colab.research.google.com/github/jumpingsphinx/jumpingsphinx.github.io/blob/main/notebooks/module3-trees/exercise4-boosting.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

---

### Exercise 5: XGBoost (3-4 hours)

**Topics Covered:**
- Applying XGBoost for classification and regression
- Systematic hyperparameter tuning
- Early stopping and cross-validation
- Feature importance analysis (gain, weight, cover)
- Handling imbalanced data with scale_pos_weight
- Comparing XGBoost with other algorithms

**Datasets:**
- Breast Cancer (classification)
- California Housing (regression)
- Credit Default (imbalanced classification)
- Kaggle-style challenge dataset

**Key Skills:**
- Use XGBoost with scikit-learn API
- Implement early stopping effectively
- Perform grid search and random search
- Analyze different feature importance types
- Handle imbalanced classes
- Save and deploy models
- Build a complete ML pipeline

<a href="https://colab.research.google.com/github/jumpingsphinx/jumpingsphinx.github.io/blob/main/notebooks/module3-trees/exercise5-xgboost.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

---

## Difficulty Levels

- 🟢 **Beginner**: Basic implementation and application
- 🟡 **Intermediate**: Requires understanding of concepts and some problem-solving
- 🔴 **Advanced**: Challenging problems requiring deep understanding

Each exercise contains problems at all three levels.

## Estimated Time

| Exercise | Estimated Time | Difficulty |
|----------|---------------|------------|
| Exercise 1: Decision Trees | 2-3 hours | 🟢 Beginner to 🟡 Intermediate |
| Exercise 2: Tree Algorithms | 2-3 hours | 🟡 Intermediate |
| Exercise 3: Random Forest | 3-4 hours | 🟡 Intermediate to 🔴 Advanced |
| Exercise 4: Boosting | 3-4 hours | 🟡 Intermediate to 🔴 Advanced |
| Exercise 5: XGBoost | 3-4 hours | 🔴 Advanced |
| **Total** | **13-18 hours** | |

## Prerequisites

Before starting these exercises, you should have:

- Completed Module 1 (Linear Algebra) and Module 2 (Regression)
- Basic understanding of Python and NumPy
- Familiarity with scikit-learn API
- Understanding of classification and regression metrics

## Setup Instructions

### Local Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
```

### Google Colab

Click the "Open in Colab" badge on any exercise to run it in Google Colab. All required packages are pre-installed or will be installed automatically.

## Datasets

All exercises use pre-loaded datasets from scikit-learn or publicly available datasets:

- **Iris**: Classic 3-class classification (150 samples, 4 features)
- **Breast Cancer**: Binary classification (569 samples, 30 features)
- **Wine Quality**: Multi-class classification (178 samples, 13 features)
- **California Housing**: Regression (20,640 samples, 8 features)
- **Titanic**: Binary classification with mixed features
- **Digits**: Multi-class classification (1,797 samples, 64 features)
- **Credit Default**: Imbalanced classification

No external downloads required!

## Tips for Success

1. **Read the lessons first**: Each exercise builds on its corresponding lesson
2. **Start simple**: Begin with beginner problems before tackling advanced ones
3. **Experiment**: Try different parameters and observe the effects
4. **Visualize**: Use plots to understand model behavior
5. **Compare**: Run multiple algorithms on the same data to see differences
6. **Document**: Add markdown cells explaining your observations
7. **Debug**: Use print statements to understand what's happening
8. **Ask questions**: Use the reflection questions to deepen understanding

## Common Challenges

### Challenge 1: Overfitting
**Problem**: Model performs well on training data but poorly on test data
**Solution**: Use cross-validation, regularization, pruning, or ensemble methods

### Challenge 2: Slow Training
**Problem**: XGBoost or Random Forest takes too long to train
**Solution**: Reduce n_estimators during development, use tree_method='hist', or subsample data

### Challenge 3: Imbalanced Classes
**Problem**: Model predicts majority class for everything
**Solution**: Use class_weight, scale_pos_weight, or resampling techniques

### Challenge 4: Hyperparameter Tuning
**Problem**: Too many hyperparameters, don't know where to start
**Solution**: Follow systematic tuning order provided in lessons and exercises

## Assessment Criteria

For each exercise, you should be able to:

- ✅ Complete all beginner-level problems
- ✅ Complete at least 75% of intermediate problems
- ✅ Attempt advanced problems (completion not required)
- ✅ Answer reflection questions thoughtfully
- ✅ Apply concepts to the real-world application section
- ✅ Explain your results with visualizations

## Going Further

After completing these exercises, consider:

1. **Kaggle Competitions**: Apply your skills to real competitions
   - [Titanic](https://www.kaggle.com/c/titanic)
   - [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

2. **Advanced Topics**:
   - LightGBM and CatBoost
   - SHAP values for model interpretation
   - Bayesian hyperparameter optimization
   - Stacking ensembles

3. **Real Projects**:
   - Customer churn prediction
   - Credit risk modeling
   - Medical diagnosis
   - Fraud detection

## Resources

- **Scikit-learn Documentation**: [Decision Trees](https://scikit-learn.org/stable/modules/tree.html), [Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)
- **XGBoost Documentation**: [Official Docs](https://xgboost.readthedocs.io/)
- **Books**:
  - "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron
  - "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman
- **Courses**:
  - [Fast.ai Machine Learning Course](https://course.fast.ai/ml)
  - [Kaggle Learn](https://www.kaggle.com/learn)

## Getting Help

If you get stuck:

1. **Review the lesson**: Go back to the corresponding lesson material
2. **Check hints**: Each exercise includes hints for challenging problems
3. **Search documentation**: Use scikit-learn and XGBoost docs
4. **Community**: Ask questions on Stack Overflow or Kaggle forums
5. **GitHub Issues**: Report problems with exercises on our [GitHub repository](https://github.com/jumpingsphinx/jumpingsphinx.github.io/issues)

---

**Ready to start?** Click on Exercise 1 above to begin your journey into tree-based machine learning!

[← Back to Module Overview](index.md)
