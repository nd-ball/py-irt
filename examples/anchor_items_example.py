#!/usr/bin/env python3
"""
Example: Using Anchor Items in IRT Models

This example demonstrates how to use anchor items (fixed parameter values)
in IRT calibration. Anchor items are useful for:
1. Test linking - maintaining calibration across different test forms
2. Equating - putting different tests on the same scale
3. Pre-calibrated items - using items with known parameters
"""

import pandas as pd
import numpy as np
from py_irt.dataset import Dataset
from py_irt.config import IrtConfig
from py_irt.training import IrtModelTrainer
import pyro

# Set random seed for reproducibility
np.random.seed(42)
torch_seed = 42


def create_example_dataset():
    """Create a synthetic dataset for demonstration"""
    # Simulate 50 subjects and 10 items
    n_subjects = 50
    n_items = 10
    
    # True parameters (for simulation)
    true_abilities = np.random.randn(n_subjects)
    true_difficulties = np.linspace(-2, 2, n_items)
    true_discriminations = np.random.uniform(0.8, 1.5, n_items)
    
    # Generate responses using 2PL model
    data = {}
    data['subject_id'] = [f'subject_{i}' for i in range(n_subjects)]
    
    for j in range(n_items):
        responses = []
        for i in range(n_subjects):
            # 2PL formula: P(correct) = 1 / (1 + exp(-a * (theta - b)))
            prob = 1 / (1 + np.exp(-true_discriminations[j] * (true_abilities[i] - true_difficulties[j])))
            response = 1 if np.random.random() < prob else 0
            responses.append(response)
        data[f'item_{j}'] = responses
    
    df = pd.DataFrame(data)
    return df, true_difficulties, true_discriminations


def example_1_basic_anchor_items():
    """Example 1: Basic usage of anchor items"""
    print("=" * 70)
    print("Example 1: Basic Usage of Anchor Items")
    print("=" * 70)
    
    # Create dataset
    df, true_difficulties, true_discriminations = create_example_dataset()
    dataset = Dataset.from_pandas(df, subject_column='subject_id')
    
    print(f"\nDataset: {len(dataset.subject_ids)} subjects, {len(dataset.item_ids)} items")
    
    # Designate items 0, 1, and 2 as anchor items
    # In a real scenario, these would come from a previous calibration
    anchor_items = [
        {
            'item_id': 'item_0',
            'difficulty': true_difficulties[0],  # Use true value as anchor
            'discrimination': true_discriminations[0]
        },
        {
            'item_id': 'item_1',
            'difficulty': true_difficulties[1],
            'discrimination': true_discriminations[1]
        },
        {
            'item_id': 'item_2',
            'difficulty': true_difficulties[2],
            'discrimination': true_discriminations[2]
        }
    ]
    
    print(f"\nAnchor items:")
    for anchor in anchor_items:
        print(f"  {anchor['item_id']}: diff={anchor['difficulty']:.3f}, disc={anchor['discrimination']:.3f}")
    
    # Add anchor items to dataset
    dataset.add_anchor_items(anchor_items)
    
    # Configure model with anchor initializer
    config = IrtConfig(
        model_type='2pl',
        priors='vague',
        epochs=100,
        lr=0.1,
        lr_decay=0.995,
        initializers=['anchor_items']  # Use anchor items initializer
    )
    
    # Clear Pyro parameter store
    pyro.clear_param_store()
    
    # Train model
    print("\nTraining 2PL model with anchor items...")
    trainer = IrtModelTrainer(
        data_path=None,
        config=config,
        dataset=dataset,
        verbose=True
    )
    
    trainer.train(epochs=100, device='cpu')
    
    # Get results
    params = trainer.best_params
    
    # Check anchor items stayed fixed
    print("\n" + "=" * 70)
    print("Verification: Anchor Items Parameters")
    print("=" * 70)
    for i, anchor in enumerate(anchor_items):
        anchor_ix = dataset.item_id_to_ix[anchor['item_id']]
        estimated_diff = params['diff'][anchor_ix]
        estimated_disc = params['disc'][anchor_ix]
        
        print(f"\n{anchor['item_id']}:")
        print(f"  Fixed difficulty:     {anchor['difficulty']:.4f}")
        print(f"  Estimated difficulty: {estimated_diff:.4f}")
        print(f"  Difference:           {abs(estimated_diff - anchor['difficulty']):.6f}")
        print(f"  Fixed discrimination:     {anchor['discrimination']:.4f}")
        print(f"  Estimated discrimination: {estimated_disc:.4f}")
        print(f"  Difference:               {abs(estimated_disc - anchor['discrimination']):.6f}")
    
    # Show non-anchor items (should have been estimated)
    print("\n" + "=" * 70)
    print("Non-Anchor Items (Estimated)")
    print("=" * 70)
    for i in range(3, 6):  # Show a few non-anchor items
        item_id = f'item_{i}'
        item_ix = dataset.item_id_to_ix[item_id]
        print(f"\n{item_id}:")
        print(f"  True difficulty:      {true_difficulties[i]:.4f}")
        print(f"  Estimated difficulty: {params['diff'][item_ix]:.4f}")
        print(f"  True discrimination:      {true_discriminations[i]:.4f}")
        print(f"  Estimated discrimination: {params['disc'][item_ix]:.4f}")


def example_2_without_anchor_items():
    """Example 2: Training without anchor items for comparison"""
    print("\n\n" + "=" * 70)
    print("Example 2: Training WITHOUT Anchor Items (for comparison)")
    print("=" * 70)
    
    # Create dataset
    df, true_difficulties, true_discriminations = create_example_dataset()
    dataset = Dataset.from_pandas(df, subject_column='subject_id')
    
    # Configure model WITHOUT anchor items
    config = IrtConfig(
        model_type='2pl',
        priors='vague',
        epochs=100,
        lr=0.1,
        lr_decay=0.995,
        initializers=[]  # No initializers
    )
    
    # Clear Pyro parameter store
    pyro.clear_param_store()
    
    # Train model
    print("\nTraining 2PL model without anchor items...")
    trainer = IrtModelTrainer(
        data_path=None,
        config=config,
        dataset=dataset,
        verbose=True
    )
    
    trainer.train(epochs=100, device='cpu')
    
    # Get results
    params = trainer.best_params
    
    # Show first few items
    print("\n" + "=" * 70)
    print("Estimated Parameters (No Anchors)")
    print("=" * 70)
    for i in range(3):
        item_id = f'item_{i}'
        item_ix = dataset.item_id_to_ix[item_id]
        print(f"\n{item_id}:")
        print(f"  True difficulty:      {true_difficulties[i]:.4f}")
        print(f"  Estimated difficulty: {params['diff'][item_ix]:.4f}")
        print(f"  True discrimination:      {true_discriminations[i]:.4f}")
        print(f"  Estimated discrimination: {params['disc'][item_ix]:.4f}")
    
    print("\nNote: Without anchor items, the scale may be different from the true scale.")


def example_3_partial_anchors():
    """Example 3: Anchoring only some parameters (e.g., only difficulty)"""
    print("\n\n" + "=" * 70)
    print("Example 3: Partial Anchoring (Only Difficulty)")
    print("=" * 70)
    
    # Create dataset
    df, true_difficulties, true_discriminations = create_example_dataset()
    dataset = Dataset.from_pandas(df, subject_column='subject_id')
    
    # Anchor only difficulty for some items
    anchor_items = [
        {
            'item_id': 'item_0',
            'difficulty': true_difficulties[0],
            # No discrimination specified - will be estimated
        },
        {
            'item_id': 'item_1',
            'difficulty': true_difficulties[1],
        }
    ]
    
    print(f"\nAnchor items (difficulty only):")
    for anchor in anchor_items:
        print(f"  {anchor['item_id']}: diff={anchor['difficulty']:.3f}")
    
    dataset.add_anchor_items(anchor_items)
    
    config = IrtConfig(
        model_type='2pl',
        priors='vague',
        epochs=100,
        lr=0.1,
        lr_decay=0.995,
        initializers=['anchor_items']
    )
    
    pyro.clear_param_store()
    
    print("\nTraining with partial anchors...")
    trainer = IrtModelTrainer(
        data_path=None,
        config=config,
        dataset=dataset,
        verbose=True
    )
    
    trainer.train(epochs=100, device='cpu')
    
    params = trainer.best_params
    
    print("\n" + "=" * 70)
    print("Results: Partially Anchored Items")
    print("=" * 70)
    for anchor in anchor_items:
        anchor_ix = dataset.item_id_to_ix[anchor['item_id']]
        print(f"\n{anchor['item_id']}:")
        print(f"  Difficulty (anchored):       {params['diff'][anchor_ix]:.4f}")
        print(f"  Discrimination (estimated):  {params['disc'][anchor_ix]:.4f}")


if __name__ == '__main__':
    # Run examples
    example_1_basic_anchor_items()
    example_2_without_anchor_items()
    example_3_partial_anchors()
    
    print("\n\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Anchor items maintain their fixed parameter values during training")
    print("2. This is useful for test linking and equating")
    print("3. You can anchor all parameters or just some (e.g., only difficulty)")
    print("4. Anchor items help maintain scale across different test administrations")

