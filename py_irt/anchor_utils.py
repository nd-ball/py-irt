# MIT License

# Copyright (c) 2019 John Lalor <john.lalor@nd.edu> and Pedro Rodriguez <me@pedro.ai>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Utilities for handling anchor items in IRT models.

Anchor items are items with fixed parameter values that do not change during training.
This is useful for linking tests or maintaining calibrated items across different test forms.
"""

from typing import List, Dict, Any
import torch
from pyro.optim import PyroOptim
import pyro
from rich.console import Console

console = Console()


class AnchorGradientZeroer:
    """
    Utility class that zeros out gradients for anchor items.
    
    This ensures that anchor item parameters remain fixed during training by
    setting their gradients to zero after backward pass but before optimizer step.
    
    Args:
        anchor_indices: List of item indices that should remain fixed
        param_names: List of parameter names to apply anchoring to (e.g., ['loc_diff', 'loc_slope'])
    """
    
    def __init__(
        self, 
        anchor_indices: List[int],
        param_names: List[str] = None
    ):
        self.anchor_indices = anchor_indices
        
        # Default parameter names to anchor (both loc and scale)
        if param_names is None:
            self.param_names = [
                'loc_diff', 'scale_diff',
                'loc_slope', 'scale_slope', 
                'loc_guess', 'scale_guess',
                'loc_slip', 'scale_slip'
            ]
        else:
            self.param_names = param_names
        
        self._hooks = []
        
        if self.anchor_indices:
            console.log(f"AnchorGradientZeroer initialized with {len(anchor_indices)} anchor items")
            console.log(f"Will zero gradients for parameters: {self.param_names}")
    
    def _create_grad_hook(self, anchor_indices: List[int]):
        """Create a hook function that zeros gradients for anchor items."""
        def hook(grad):
            if grad is not None:
                # Clone the gradient to avoid in-place modification issues
                grad_copy = grad.clone()
                for anchor_ix in anchor_indices:
                    if anchor_ix < grad_copy.shape[0]:
                        grad_copy[anchor_ix] = 0.0
                return grad_copy
            return grad
        return hook
    
    def register_hooks(self) -> None:
        """Register backward hooks on anchor item parameters."""
        if not self.anchor_indices:
            return
        
        param_store = pyro.get_param_store()
        
        for param_name in self.param_names:
            if param_name in param_store:
                # Get the parameter - this might be constrained
                param = param_store[param_name]
                
                # Check if parameter has unconstrained version (for constrained parameters)
                if hasattr(param, 'unconstrained') and callable(param.unconstrained):
                    try:
                        # Use unconstrained parameter for hook (this is where gradients actually flow)
                        param_to_hook = param.unconstrained()
                        console.log(f"Registered gradient hook for {param_name} (on unconstrained parameter)")
                    except Exception:
                        # If unconstrained() fails, use regular parameter
                        param_to_hook = param
                        console.log(f"Registered gradient hook for {param_name}")
                else:
                    # Use regular parameter (not constrained)
                    param_to_hook = param
                    console.log(f"Registered gradient hook for {param_name}")
                
                # Register a hook that will be called during backward pass
                hook = param_to_hook.register_hook(self._create_grad_hook(self.anchor_indices))
                self._hooks.append(hook)
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
    
    def zero_anchor_gradients(self) -> None:
        """Manually zero out gradients for anchor item parameters."""
        if not self.anchor_indices:
            return
        
        param_store = pyro.get_param_store()
        
        for param_name in self.param_names:
            if param_name in param_store:
                param = param_store[param_name]
                
                # For constrained parameters, work with unconstrained version
                if hasattr(param, 'unconstrained') and callable(param.unconstrained):
                    try:
                        param_to_zero = param.unconstrained()
                    except Exception:
                        param_to_zero = param
                else:
                    param_to_zero = param
                
                # Check if parameter has gradients
                if param_to_zero.grad is not None:
                    # Zero out gradients for anchor items
                    for anchor_ix in self.anchor_indices:
                        if anchor_ix < param_to_zero.grad.shape[0]:
                            param_to_zero.grad[anchor_ix] = 0.0
    
    def __call__(self):
        """Allows using the zeroer as a callable."""
        self.zero_anchor_gradients()


def create_anchor_gradient_zeroer(dataset, param_names: List[str] = None):
    """
    Create an anchor gradient zeroer from a dataset.
    
    Args:
        dataset: The Dataset object containing anchor item information
        param_names: Optional list of parameter names to anchor
    
    Returns:
        AnchorGradientZeroer: A gradient zeroer that respects anchor items
    """
    anchor_indices = dataset.get_anchor_indices() if hasattr(dataset, 'get_anchor_indices') else []
    
    if not anchor_indices:
        console.log("No anchor items found, gradient zeroer will be a no-op")
    
    return AnchorGradientZeroer(anchor_indices, param_names)

