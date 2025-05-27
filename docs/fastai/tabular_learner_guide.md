# fastai TabularLearner Documentation

This document provides key information about fastai's TabularLearner, focusing on its structure and feature importance capabilities.

## TabularLearner Overview

TabularLearner is fastai's class for training models on tabular data. It builds on top of PyTorch and provides a high-level API for training deep learning models on structured data.

```python
def tabular_learner(dls, layers=None, emb_szs=None, config=None, n_out=None, y_range=None, **kwargs)
```

### Parameters:
- **dls**: DataLoaders for the tabular data
- **layers**: List of integers representing hidden layer sizes
- **emb_szs**: List of tuple of embeddings sizes, one per categorical variable
- **config**: Dictionary of configuration options for the model
- **n_out**: Number of output neurons
- **y_range**: Tuple with min and max values for regression tasks

## TabularModel Structure

TabularLearner uses a TabularModel internally, which has the following structure:

1. **Embedding layers** for categorical variables
2. **Batch normalization** for continuous variables
3. **Concatenation** of embedded categorical and normalized continuous features
4. **Fully connected layers** with dropout, batch normalization and ReLU activations
5. **Output layer** with appropriate activation for the task

The TabularModel is not directly subscriptable like `model[0]` because it's implemented as a PyTorch `nn.Module` with named components rather than a list-like structure.

## Feature Importance in TabularLearner

Unlike some machine learning libraries (e.g., scikit-learn's RandomForest), TabularLearner/TabularModel does not have a built-in `feature_importance` method or attribute. Instead, there are several approaches for calculating feature importance:

### 1. Using permutation importance

```python
from sklearn.inspection import permutation_importance
import numpy as np

def get_perm_importance(learn, valid_dl=None, n_repeats=5):
    # If no validation dataloader is provided, use the one from the learner
    if valid_dl is None:
        valid_dl = learn.dls.valid
    
    # Put model in evaluation mode and move to CPU
    learn.model.eval()
    device = torch.device('cpu')
    learn.model.to(device)
    
    # Get validation data
    val_xs, val_y = valid_dl.dataset.x, valid_dl.dataset.y
    
    # Function to calculate the metric (e.g., RMSE)
    def score_func(xs, y):
        with torch.no_grad():
            preds = learn.model(xs)
        return torch.sqrt(F.mse_loss(preds, y)).item()
    
    # Calculate baseline score
    baseline_score = score_func(val_xs, val_y)
    
    # Get feature names
    feature_names = learn.dls.x_names
    
    # Calculate importance for each feature
    importances = {}
    for i, feature in enumerate(feature_names):
        scores = []
        for _ in range(n_repeats):
            # Create a copy of the feature data
            xs_permuted = val_xs.copy()
            
            # Permute the feature
            perm_idx = torch.randperm(len(val_y))
            if i < len(learn.dls.cat_names):  # Categorical
                xs_permuted.cats[:, i] = val_xs.cats[perm_idx, i]
            else:  # Continuous
                cont_idx = i - len(learn.dls.cat_names)
                xs_permuted.conts[:, cont_idx] = val_xs.conts[perm_idx, cont_idx]
            
            # Calculate score with permuted feature
            score = score_func(xs_permuted, val_y)
            scores.append(score - baseline_score)
        
        # Mean importance across repeats
        importances[feature] = np.mean(scores)
    
    return importances
```

### 2. Using Random Forest as a proxy

Since TabularLearner doesn't provide direct feature importance, a common approach is to train a Random Forest on the same data and use its feature_importances_ attribute:

```python
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def get_rf_feature_importance(df, cat_names, cont_names, target_name):
    # Prepare data for Random Forest
    X = pd.get_dummies(df, columns=cat_names)
    X = X.drop(columns=[target_name])
    y = df[target_name]
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importance
    feature_names = X.columns
    importances = {name: imp for name, imp in zip(feature_names, rf.feature_importances_)}
    
    # Sort by importance
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    
    return importances
```

### 3. Visualizing Feature Importance

```python
import matplotlib.pyplot as plt

def plot_feature_importance(importances, top_n=10):
    # Get top N features
    top_features = list(importances.keys())[:top_n]
    top_importances = [importances[feature] for feature in top_features]
    
    # Create horizontal bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_features)), top_importances, align='center')
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    plt.show()
```

## Common Errors and Solutions

### 1. TabularModel is not subscriptable

**Error**: `TypeError: 'TabularModel' object is not subscriptable`

This error occurs when trying to access the model with indexing (e.g., `learn.model[0]`) because TabularModel is not a list-like structure.

**Solution**: Access model components using their attribute names:

```python
# Incorrect:
weights = learn.model[0].parameters()

# Correct:
# Access the embeddings
embeddings = learn.model.embeds

# Access the linear layers
linear_layers = learn.model.layers
```

### 2. CUDA tensor conversion errors

**Error**: `TypeError: can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.`

**Solution**: Move the model and tensors to CPU before conversion:

```python
# Move model to CPU
learn.model.cpu()

# For tensors
tensor_cpu = cuda_tensor.cpu()
numpy_array = tensor_cpu.numpy()
```

### 3. Getting feature names

To get the feature names used by the model:

```python
# Get categorical features
cat_names = learn.dls.cat_names

# Get continuous features
cont_names = learn.dls.cont_names

# Get all feature names
all_features = learn.dls.x_names
``` 