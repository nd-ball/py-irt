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

import unittest
import pandas as pd
import numpy as np
import pyro
import torch

from py_irt.dataset import Dataset
from py_irt.config import IrtConfig
from py_irt.training import IrtModelTrainer


class TestAnchorItems(unittest.TestCase):
    """Test anchor items functionality"""

    def setUp(self):
        """Set up a simple test dataset"""
        # Create a simple dataset with known structure
        # Each row is a unique subject, columns are items
        self.df = pd.DataFrame({
            'subject_id': ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
                          's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20'],
            'item_1': [1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
            'item_2': [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'item_3': [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            'item_4': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        })
        
        self.dataset = Dataset.from_pandas(self.df, subject_column='subject_id')
    
    def test_add_anchor_items(self):
        """Test adding anchor items to a dataset"""
        anchor_items = [
            {'item_id': 'item_1', 'difficulty': 0.5, 'discrimination': 1.2},
            {'item_id': 'item_3', 'difficulty': -0.8, 'discrimination': 0.9}
        ]
        
        self.dataset.add_anchor_items(anchor_items)
        
        # Check that anchor items were added
        self.assertIsNotNone(self.dataset.anchor_items)
        self.assertEqual(len(self.dataset.anchor_items), 2)
        
        # Check anchor item properties
        anchor_1 = self.dataset.anchor_items[0]
        self.assertEqual(anchor_1.item_id, 'item_1')
        self.assertEqual(anchor_1.difficulty, 0.5)
        self.assertEqual(anchor_1.discrimination, 1.2)
        
        # Check anchor indices
        anchor_indices = self.dataset.get_anchor_indices()
        self.assertEqual(len(anchor_indices), 2)
    
    def test_anchor_items_invalid_id(self):
        """Test that adding anchor items with invalid ID raises error"""
        anchor_items = [
            {'item_id': 'invalid_item', 'difficulty': 0.5}
        ]
        
        with self.assertRaises(ValueError):
            self.dataset.add_anchor_items(anchor_items)
    
    def test_training_with_anchor_items(self):
        """Test training with anchor items"""
        # Add anchor items
        anchor_items = [
            {'item_id': 'item_1', 'difficulty': 0.5, 'discrimination': 1.2},
        ]
        self.dataset.add_anchor_items(anchor_items)
        
        # Clear Pyro param store
        pyro.clear_param_store()
        
        # Create config with anchor initializer
        config = IrtConfig(
            model_type='2pl',
            priors='vague',
            epochs=10,
            lr=0.1,
            initializers=['anchor_items']
        )
        
        # Train model
        trainer = IrtModelTrainer(
            data_path=None,
            config=config,
            dataset=self.dataset,
            verbose=False
        )
        
        trainer.train(epochs=10, device='cpu')
        
        # Get final parameters
        params = trainer.last_params
        
        # Check that anchor item parameters are close to their fixed values
        anchor_ix = self.dataset.anchor_items[0].item_ix
        difficulty = params['diff'][anchor_ix]
        discrimination = params['disc'][anchor_ix]
        
        # Both difficulty and discrimination should stay very close to their fixed values
        print(f"\nAnchor item parameters:")
        print(f"  Difficulty: expected=0.5, got={difficulty:.4f}")
        print(f"  Discrimination: expected=1.2, got={discrimination:.4f}")
        
        # Verify anchor parameters stayed fixed (allow small numerical error)
        self.assertAlmostEqual(difficulty, 0.5, places=2,
                              msg=f"Difficulty should stay at 0.5, got {difficulty}")
        self.assertAlmostEqual(discrimination, 1.2, places=2,
                              msg=f"Discrimination should stay at 1.2, got {discrimination}")
        
        # Verify non-anchor items have different parameters (not all zeros)
        non_anchor_diffs = [params['diff'][i] for i in range(len(params['diff'])) if i != anchor_ix]
        non_anchor_discs = [params['disc'][i] for i in range(len(params['disc'])) if i != anchor_ix]
        
        # At least one non-anchor item should have non-zero difficulty
        has_nonzero_diff = any(abs(d) > 0.01 for d in non_anchor_diffs)
        self.assertTrue(has_nonzero_diff, "At least one non-anchor item should have non-zero difficulty")
    
    def test_anchor_gradient_zeroer(self):
        """Test that anchor gradient zeroer properly zeros gradients"""
        from py_irt.anchor_utils import AnchorGradientZeroer
        import pyro
        
        pyro.clear_param_store()
        
        # Create some test parameters
        loc_diff = pyro.param('loc_diff', torch.zeros(4))
        loc_slope = pyro.param('loc_slope', torch.ones(4))
        
        # Set up gradients
        loc_diff.grad = torch.ones(4)
        loc_slope.grad = torch.ones(4) * 2.0
        
        # Create zeroer for anchor indices [0, 2]
        zeroer = AnchorGradientZeroer(
            anchor_indices=[0, 2],
            param_names=['loc_diff', 'loc_slope']
        )
        
        # Zero anchor gradients
        zeroer.zero_anchor_gradients()
        
        # Check that anchor gradients are zeroed
        self.assertEqual(loc_diff.grad[0].item(), 0.0)
        self.assertEqual(loc_diff.grad[2].item(), 0.0)
        self.assertEqual(loc_slope.grad[0].item(), 0.0)
        self.assertEqual(loc_slope.grad[2].item(), 0.0)
        
        # Check that non-anchor gradients are unchanged
        self.assertEqual(loc_diff.grad[1].item(), 1.0)
        self.assertEqual(loc_diff.grad[3].item(), 1.0)
        self.assertEqual(loc_slope.grad[1].item(), 2.0)
        self.assertEqual(loc_slope.grad[3].item(), 2.0)


if __name__ == '__main__':
    unittest.main()

