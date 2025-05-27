# fastai Coding Style Guide

This document details the coding style used throughout the fastai library. Fastai is known for its distinctive coding style which emphasizes expressiveness, readability, and conciseness.

## Principles

1. **Concise code**: Favors brevity without sacrificing clarity. Often achieves this through careful function and class design.
2. **Expressive naming**: Names reveal intent. Clear, intuitive naming is preferred over documentation.
3. **Functional approach**: Encourages composition of small, single-purpose functions.
4. **Consistent patterns**: Uses consistent patterns across the library to make code more predictable.
5. **Progressive disclosure**: Simple for beginners, powerful for experts.

## Symbol Naming

### Variables and Functions

- **Use meaningful but concise names**: `plot_top_losses` not `plot_images_with_highest_loss_values`
- **Use snake_case for variables and functions**: `acc_thresh` not `accThresh` or `AccThresh`
- **Use Huffman coding principle**: Common concepts get shorter names
  - Very common: 1-3 characters (e.g., `x`, `y`, `df`)
  - Common: 4-8 characters (e.g., `loss`, `metrics`)
  - Less common: 9+ characters (e.g., `regression`, `accuracy`)
- **Abbreviate consistently**: Follow the [abbreviation guide](./abbreviations.md)

### Classes

- **Use CapWords (PascalCase)**: `DataLoader` not `data_loader`
- **Prefer shorter names for frequently used classes**: `Learner` not `ModelTrainer`
- **Name factories after what they create**: `get_preds` returns predictions

### Parameters

- **Be consistent across similar functions**: If one function uses `bs` for batch size, all should use `bs`
- **Order parameters**: Required, optional, very rarely used
- **Use defaults liberally**: Make the API simple for common cases

## Layout

### Whitespace

- **Follow PEP 8 with exceptions**:
  - Allow longer lines (120 characters) when it improves readability
  - Sometimes put multiple short statements on one line when they're closely related

### Imports

- **Organize imports as**:
  1. Standard library imports
  2. Third-party imports
  3. fastai imports
- **Use explicit imports** for public APIs: `from fastai.vision.all import *` is OK in notebooks but not library code
- **Avoid `import *`** inside the library itself

### Code Organization

- **Callbacks over inheritance**: Prefer composition with callbacks over deep inheritance hierarchies
- **Mixins for shared functionality**: When inheritance makes sense, use mixins for reusable behaviors
- **Keep files focused**: Each file should have a single responsibility
- **Progressive disclosure**: Simple entry points, with complexity hidden behind layers of abstraction

## Algorithms

### Tensors

- **Work with tensors, not lists**: Vectorize operations whenever possible
- **Use broadcasting**: Leverage PyTorch's broadcasting rules for concise code
- **Reuse memory**: Use in-place operations (with trailing underscore like `x.mul_(y)`) when appropriate

### Performance

- **Readability first, then performance**: Write clear code first, optimize only if needed
- **Profile before optimizing**: Use `%time`, `%timeit` or dedicated profilers
- **GPU-friendly code**: Minimize CPU-GPU transfers, batch operations

### Error Handling

- **Fail fast and clearly**: Raise exceptions early with clear messages
- **Type annotations**: Use type hints for complex functions (not required for simple ones)
- **Assertions**: Use assertions for internal invariants, proper exceptions for API errors

## Documentation

### Docstrings

- **First line is short description**: One-line summary of the function/class
- **Use examples in docstrings**: Show typical usage
- **Parameters**: Document parameters with types and purpose
- **Returns**: Describe return values clearly

### Comments

- **Comment why, not what**: Code shows what, comments explain why
- **Use comments for complex algorithms**: Especially math transformations

### Notebooks

- **Notebooks are first-class documentation**: fastai's teaching style emphasizes working notebooks
- **Keep notebook code similar to library code**: Maintain consistent style

## Testing

- **Test public APIs**: Focus on testing user-facing APIs
- **Integration over unit**: Prefer testing components working together over isolated units
- **Doctest examples**: Examples in docstrings should work as tests

## Examples

### Good fastai Style

```python
# Simple function with clear purpose
def accuracy(input, target):
    "Compute accuracy with `target` when `input` is bs * n_classes"
    pred = input.argmax(dim=-1)
    return (pred == target).float().mean()

# Class with consistent interface
class Normalize:
    "Normalize x to mean 0 and std 1"
    def __init__(self, mean=None, std=None): self.mean,self.std = mean,std
    
    def setup(self, x):
        "Setup mean and std with `x`"
        if self.mean is None: self.mean = x.mean()
        if self.std is None: self.std = x.std()
        return self
    
    def __call__(self, x): return (x - self.mean) / self.std
```

### Bad Style (Avoid)

```python
# Too verbose, inconsistent naming
def calculate_model_accuracy_score(model_outputs, ground_truth_labels):
    predicted_class_indices = torch.argmax(model_outputs, dim=1)
    correct_predictions_mask = predicted_class_indices.eq(ground_truth_labels)
    accuracy_value = torch.mean(correct_predictions_mask.float())
    return accuracy_value

# Unclear purpose, inconsistent interface
class data_normalizer:
    def __init__(self, normMean=None, normStd=None):
        self.normMean = normMean
        self.normStd = normStd
    
    def train(self, data):
        # Should be setup to match other transforms
        if self.normMean == None:
            self.normMean = torch.mean(data)
        if self.normStd == None:
            self.normStd = torch.std(data)
    
    def normalize(self, x):
        # Should use __call__ for consistency with other transforms
        return (x - self.normMean) / self.normStd
``` 