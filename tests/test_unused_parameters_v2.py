"""DDP needs the v2 model's unused parameters to stay exactly the known pair.

``AttentionV2`` allocates ``context_norm`` for cross attention, so the
self-attention copy keeps two parameters that never receive gradients. Single
GPU training never noticed; DDP stops unless it is told to expect them.
"""
import unittest

import torch

from merraflow.model_v2 import UNetV2, regression_v2


EXPECTED = {'self_attention.context_norm.weight', 'self_attention.context_norm.bias'}


def small_model(mean_condition):
    return UNetV2(3, base_channels=8, channel_mult=(1, 2), time_dim=8, blocks_per_level=1,
                  attention_heads=2, mean_condition=mean_condition)


def without_gradients(model, backward):
    model.train()
    for parameter in model.parameters():
        parameter.grad = None
    backward(model)
    return {name for name, p in model.named_parameters() if p.grad is None}


class UnusedParameterTests(unittest.TestCase):
    def test_flow_stage_leaves_only_self_attention_context_norm(self):
        model = small_model(True)
        def backward(model):
            batch = torch.zeros(2, 5, 16, 16)
            velocity = model(batch, torch.rand(2), torch.zeros(2, 3, 16, 16),
                             torch.zeros(2, 3, 16, 16), torch.zeros(2, 5, 16, 16))
            velocity.square().mean().backward()
        self.assertEqual(without_gradients(model, backward), EXPECTED)

    def test_regression_stage_leaves_only_self_attention_context_norm(self):
        model = small_model(False)
        def backward(model):
            batch = {'target': torch.zeros(2, 5, 16, 16), 'condition': torch.zeros(2, 3, 16, 16),
                     'context': torch.zeros(2, 3, 16, 16)}
            regression_v2(model, batch).square().mean().backward()
        self.assertEqual(without_gradients(model, backward), EXPECTED)
