# Anchor Items in py-irt

## Overview

Anchor items are items with **fixed, pre-determined parameter values** that remain constant during IRT model calibration. This feature is essential for:

- **Test Linking**: Connecting different test forms to a common scale
- **Test Equating**: Ensuring different test versions measure on the same scale
- **Incremental Calibration**: Adding new items while keeping existing items fixed
- **Scale Maintenance**: Maintaining consistent measurement scales over time

## How It Works

When you designate items as anchors:

1. Their parameter values are initialized to specified fixed values
2. During training, their gradients are automatically zeroed out
3. This ensures they remain constant throughout calibration
4. Other items are calibrated relative to these fixed anchor items

The implementation uses PyTorch hooks to zero gradients during the backward pass, ensuring the optimizer cannot update anchor item parameters.

## Quick Start

```python
import pandas as pd
from py_irt.dataset import Dataset
from py_irt.config import IrtConfig
from py_irt.training import IrtModelTrainer

# 1. Load your data
df = pd.read_csv('your_data.csv')
dataset = Dataset.from_pandas(df, subject_column='subject_id')

# 2. Define anchor items with their fixed parameter values
anchor_items = [
    {
        'item_id': 'item_1',
        'difficulty': 0.5,
        'discrimination': 1.2
    },
    {
        'item_id': 'item_3',
        'difficulty': -0.8,
        'discrimination': 0.9
    }
]

# 3. Add anchor items to the dataset
dataset.add_anchor_items(anchor_items)

# 4. Configure model with anchor initializer
config = IrtConfig(
    model_type='2pl',
    priors='vague',
    epochs=100,
    initializers=['anchor_items']  # ← Important!
)

# 5. Train as usual
trainer = IrtModelTrainer(
    data_path=None,
    config=config,
    dataset=dataset
)
trainer.train()

# 6. Get results
params = trainer.best_params
```

## Detailed Usage

### Defining Anchor Items

Anchor items are defined as a list of dictionaries, where each dictionary specifies:

- `item_id` (required): The string identifier of the item
- `difficulty` (optional): Fixed difficulty parameter
- `discrimination` (optional): Fixed discrimination parameter  
- `guessing` (optional): Fixed guessing parameter (for 3PL/4PL models)

**Example: Full anchoring (all parameters fixed)**
```python
anchor_items = [
    {
        'item_id': 'item_1',
        'difficulty': 0.5,
        'discrimination': 1.2
    }
]
```

**Example: Partial anchoring (only difficulty fixed)**
```python
anchor_items = [
    {
        'item_id': 'item_1',
        'difficulty': 0.5
        # discrimination will be estimated
    }
]
```

### Adding Anchor Items to Dataset

```python
dataset.add_anchor_items(anchor_items)
```

This method:
- Validates that all item IDs exist in the dataset
- Creates `AnchorItem` objects with the specified parameters
- Stores them in `dataset.anchor_items`

### Configuring the Model

The key step is to include `'anchor_items'` in the initializers list:

```python
config = IrtConfig(
    model_type='2pl',  # or '1pl', '3pl', '4pl'
    initializers=['anchor_items']
)
```

You can combine it with other initializers:

```python
config = IrtConfig(
    model_type='2pl',
    initializers=[
        'anchor_items',
        {'name': 'difficulty_sign', 'magnitude': 2.0, 'n_to_init': 5}
    ]
)
```

## Supported Models

Anchor items work with all standard IRT models:

| Model | Supported Parameters |
|-------|---------------------|
| 1PL | `difficulty` |
| 2PL | `difficulty`, `discrimination` |
| 3PL | `difficulty`, `discrimination`, `guessing` |
| 4PL | `difficulty`, `discrimination`, `guessing`, `slip` |

## Use Cases

### 1. Test Linking

Link two test forms using common anchor items:

```python
# Form A calibration (reference form)
dataset_A = Dataset.from_pandas(form_A_data, subject_column='subject_id')
# ... train and get parameters for all items

# Form B calibration (new form) with common items as anchors
dataset_B = Dataset.from_pandas(form_B_data, subject_column='subject_id')

# Use Form A parameters for common items as anchors
anchor_items = [
    {
        'item_id': 'common_item_1',
        'difficulty': form_A_params['diff'][item_1_ix],
        'discrimination': form_A_params['disc'][item_1_ix]
    },
    # ... more common items
]

dataset_B.add_anchor_items(anchor_items)
# Now Form B will be calibrated on the same scale as Form A
```

### 2. Incremental Calibration

Add new items to an existing item bank:

```python
# Load existing item bank parameters
item_bank = pd.read_csv('item_bank.csv')

# New data with both old and new items
new_dataset = Dataset.from_pandas(new_data, subject_column='subject_id')

# Use existing items as anchors
anchor_items = [
    {
        'item_id': row['item_id'],
        'difficulty': row['difficulty'],
        'discrimination': row['discrimination']
    }
    for _, row in item_bank.iterrows()
]

new_dataset.add_anchor_items(anchor_items)
# New items will be calibrated relative to the fixed item bank
```

### 3. Test Equating

Ensure different test versions measure on the same scale:

```python
# Calibrate base form
base_form = Dataset.from_pandas(base_data, subject_column='subject_id')
# ... train and get parameters

# For each parallel form, use anchor items
for form_data in parallel_forms:
    form_dataset = Dataset.from_pandas(form_data, subject_column='subject_id')
    
    # Use common items as anchors
    form_dataset.add_anchor_items(anchor_items)
    
    # Calibrate - will be on the same scale as base form
    # ... train
```

## Implementation Details

### Architecture

The anchor items functionality consists of three main components:

1. **Dataset Extensions** (`py_irt/dataset.py`):
   - `AnchorItem` class: Stores anchor item information
   - `add_anchor_items()`: Method to add anchors to a dataset
   - `get_anchor_indices()`: Helper to retrieve anchor item indices

2. **AnchorItemInitializer** (`py_irt/initializers.py`):
   - Sets initial parameter values for anchor items
   - Sets scale parameters to near-zero (very low variance)
   - Registered as `'anchor_items'` initializer

3. **AnchorGradientZeroer** (`py_irt/anchor_utils.py`):
   - Uses PyTorch hooks to zero gradients during backward pass
   - Ensures optimizer cannot update anchor parameters
   - Automatically registered/removed by training loop

### How Gradients are Zeroed

The implementation uses PyTorch's `register_hook()` mechanism:

```python
def _create_grad_hook(self, anchor_indices):
    def hook(grad):
        if grad is not None:
            grad_copy = grad.clone()
            for anchor_ix in anchor_indices:
                grad_copy[anchor_ix] = 0.0
            return grad_copy
        return grad
    return hook
```

This hook is called automatically during the backward pass, before the optimizer step, ensuring anchor parameters never receive gradient updates.

### Training Loop Integration

The training loop in `py_irt/training.py` automatically:

1. Detects if anchor items are present in the dataset
2. Creates an `AnchorGradientZeroer`
3. Registers gradient hooks after parameter initialization
4. Cleans up hooks after training completes

No manual intervention is required beyond adding `'anchor_items'` to the initializers list.

## Validation and Verification

### Checking Anchor Items Stayed Fixed

After training, verify that anchor items maintained their values:

```python
params = trainer.best_params

for anchor in dataset.anchor_items:
    anchor_ix = dataset.item_id_to_ix[anchor.item_id]
    
    if anchor.difficulty is not None:
        estimated = params['diff'][anchor_ix]
        fixed = anchor.difficulty
        error = abs(estimated - fixed)
        print(f"{anchor.item_id} difficulty: fixed={fixed:.4f}, estimated={estimated:.4f}, error={error:.6f}")
        
    if anchor.discrimination is not None:
        estimated = params['disc'][anchor_ix]
        fixed = anchor.discrimination
        error = abs(estimated - fixed)
        print(f"{anchor.item_id} discrimination: fixed={fixed:.4f}, estimated={estimated:.4f}, error={error:.6f}")
```

Errors should be extremely small (< 0.001), confirming parameters stayed fixed.

## Examples

See the following files for complete examples:

- **`examples/anchor_items_example.py`**: Comprehensive Python script with multiple examples
- **`tests/test_anchor_items.py`**: Unit tests demonstrating functionality

Run the example:

```bash
python examples/anchor_items_example.py
```

Run the tests:

```bash
python -m pytest tests/test_anchor_items.py -v
```

## Troubleshooting

### Anchor parameters are changing slightly

**Problem**: Anchor item parameters show small changes (e.g., 0.001-0.01).

**Causes**:
- Numerical precision issues in PyTorch
- Learning rate too high
- Not using the `'anchor_items'` initializer

**Solution**:
```python
# Make sure to include anchor_items initializer
config = IrtConfig(
    initializers=['anchor_items']  # ← Don't forget this!
)
```

### Item ID not found error

**Problem**: `ValueError: Anchor item 'item_x' not found in dataset`

**Cause**: The item ID doesn't exist in the dataset.

**Solution**: Check that item IDs match exactly:
```python
print("Available items:", list(dataset.item_ids))
print("Your anchor ID:", anchor_items[0]['item_id'])
```

### Anchor items have no effect

**Problem**: All items seem to be changing during training.

**Cause**: Forgot to add `'anchor_items'` to initializers.

**Solution**:
```python
config = IrtConfig(
    model_type='2pl',
    initializers=['anchor_items']  # ← Required!
)
```

## API Reference

### Dataset.add_anchor_items(anchor_items)

Add anchor items to the dataset.

**Parameters:**
- `anchor_items` (List[Dict]): List of dictionaries specifying anchor items

**Example:**
```python
dataset.add_anchor_items([
    {'item_id': 'item_1', 'difficulty': 0.5, 'discrimination': 1.2}
])
```

### Dataset.get_anchor_indices()

Get the integer indices of anchor items.

**Returns:**
- `List[int]`: List of anchor item indices

### AnchorItem

Pydantic model representing an anchor item.

**Fields:**
- `item_id` (str): Item identifier
- `item_ix` (int): Item index in the dataset
- `difficulty` (Optional[float]): Fixed difficulty value
- `discrimination` (Optional[float]): Fixed discrimination value
- `guessing` (Optional[float]): Fixed guessing value

### AnchorItemInitializer

Initializer that sets anchor item parameter values.

**Usage:**
```python
config = IrtConfig(initializers=['anchor_items'])
```

### AnchorGradientZeroer

Utility class that zeros gradients for anchor items.

**Note**: This is used automatically by the training loop. You don't need to interact with it directly.

## Limitations

1. **Amortized Models**: Anchor items are not currently supported for amortized models (e.g., `amortized_1pl`).

2. **Hierarchical Priors**: When using hierarchical priors, anchor items fix the item-level parameters but not the hyperparameters (mu, sigma).

3. **MCMC**: Anchor items are designed for variational inference (SVI). MCMC is not supported.

### Technical Note on Parameter Fixing

The implementation uses a combination of gradient hooks and manual parameter reset to ensure anchor items stay fixed:

1. **Gradient Hooks**: Registered on unconstrained parameters (for constrained parameters like `discrimination`) to zero gradients during backward pass
2. **Manual Reset**: After each optimizer step, anchor parameters are reset to their fixed values
3. **Constraint Handling**: For parameters with positive constraints, both the constrained value and its unconstrained representation (log space) are updated

This dual approach ensures high precision for all parameter types:
- **Difficulty** (unconstrained): Typically < 0.001 deviation
- **Discrimination** (positive constraint): Typically < 0.001 deviation  
- **Guessing/Slip** (positive constraint): Typically < 0.001 deviation

The implementation correctly handles Pyro's internal parameter transformations, ensuring that anchors remain stable throughout training.

## References

For more information on anchor items and test linking in IRT:

- Kolen, M. J., & Brennan, R. L. (2014). *Test Equating, Scaling, and Linking* (3rd ed.). Springer.
- von Davier, A. A., Holland, P. W., & Thayer, D. T. (2004). *The Kernel Method of Test Equating*. Springer.

## Contributing

If you find bugs or have suggestions for improving anchor items functionality, please open an issue on GitHub.

