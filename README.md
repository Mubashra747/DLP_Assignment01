# Deep Learning for Perception

## Building, Breaking and Fixing a Neural Network

## Overview

This notebook builds a feedforward neural network end to end on **Fashion-MNIST**, deliberately pushes it into overfitting, and then repairs it using regularisation and hyperparameter tuning. The effect of every major design choice is measured and analysed.

## Contents

| Part | Description | Marks |
|------|-------------|-------|
| Setup | Load, normalise, flatten, and perform an 80/20 train/validation split | — |
| Part 1 | Two-layer MLP implemented from scratch in NumPy; gradients verified against PyTorch | 15 |
| Part 2 | Baseline model and activation function study (Sigmoid, Tanh, ReLU, Leaky ReLU) | 15 |
| Part 3 | Loss function comparison (Cross-Entropy vs MSE) + tabular regression task | 10 |
| Part 4 | Optimiser comparison (SGD, SGD + Momentum, RMSProp, Adam) | 10 |
| Part 5 | Deliberately forcing overfitting on a large network / small dataset | 10 |
| Part 6 | Regularisation study: L2, L1, Dropout, Batch Normalisation, Early Stopping, Augmentation, and More Data | 25 |
| Part 7 | Random search + 5-fold cross-validation hyperparameter tuning and final test evaluation | 15 |

Every code cell has a short Markdown note above it explaining what the cell does. This allows the notebook to be followed from top to bottom without needing to read the code first.

## Dataset

**Fashion-MNIST** — [Kaggle Dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist)

The dataset is loaded using the Kaggle CSV files:

- `fashion-mnist_train.csv`
- `fashion-mnist_test.csv`

### Preprocessing

- 43 duplicate rows were removed from the training set.
- 1 duplicate row was removed from the test set.
- Pixel values were normalised to the range `[0, 1]`.
- Images were flattened into 784-dimensional vectors.
- Training data was split 80/20 using a stratified train/validation split.
- The provided test set was kept untouched until the final evaluation in Part 7.

## Key Results

| Metric | Part 2 Baseline | Final Tuned Model |
|--------|-----------------|-------------------|
| Accuracy | 88.97% (Validation) | 90.36% (Test) |
| Macro Precision | — | 90.32% |
| Macro Recall | — | 90.36% |
| Macro F1 | — | 90.29% |

### Selected Configuration

The final configuration selected in **Part 7** was:

- **Learning Rate:** `0.0005`
- **Hidden Width:** `512`
- **Dropout:** `0.2`
- **Search Method:** Random Search
- **Number of Configurations:** 12
- **Cross-Validation:** 5-Fold
- **Mean CV Accuracy:** 84.88%

The selected model was then retrained on the full training set before being evaluated on the untouched test set.

### Most Impactful Change
Increasing the training data from **2,000 to 20,000 samples** reduced the train/validation accuracy gap from **17.25 percentage points to 10.18 percentage points**.

At the same time, training accuracy decreased only slightly:

- **Before:** 100%
- **After:** 98.26%

This provided a better trade-off than any of the explicit regularisation techniques tested.

## How to Reproduce

1. Open the notebook on [Kaggle](https://www.kaggle.com/) with a **GPU T4 x2** accelerator.

2. Add the [Fashion-MNIST dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist) as a notebook input.

3. Run all cells from top to bottom using **Run All**.

4. All random seeds are fixed:
   - `random_state = 42`
   - `np.random.seed(42)`

   Therefore, the results are reproducible.

5. No other files are required. The only external data used besides Fashion-MNIST is:
   `sklearn.datasets.fetch_california_housing`

   This dataset is downloaded automatically during Part 3.

## Notes

- Categorical cross-entropy loss uses one-hot encoded labels throughout the notebook. This maintains consistency between Part 1, which uses NumPy, and the later Keras-based parts.
- The validation set, rather than the test set, was used for every intermediate model-selection decision in Parts 2–6.
- This approach avoids test-set leakage.
- The test set was evaluated exactly once, during the final evaluation in Part 7.

## Conclusion

This assignment demonstrates the complete workflow of building, evaluating, diagnosing, and improving a neural network.

The experiments show that model performance is influenced not only by the network architecture but also by:

- Activation functions
- Loss functions
- Optimisers
- Training data size
- Regularisation techniques
- Hyperparameter selection
- Cross-validation

The results particularly highlight the importance of **sufficient training data** in reducing overfitting and improving generalisation performance.
