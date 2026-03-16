#!/usr/bin/env python3
"""Debug script to check model dimensions."""

import sys
sys.path.insert(0, '..')
import paddle

from ppmat.models import SolvGNN

# Create model with hidden_dim=64
model = SolvGNN(
    in_dim=75,
    hidden_dim=64,
    n_classes=1,
    mlp_dropout_rate=0.1,
    mlp_activation='softplus',
    mpnn_activation='relu',
    num_step_message_passing=6,
    pinn_lambda=1.0
)

print("Model created successfully")
print()

# Check classifier dimensions
print("classify1 layers:")
for i, layer in enumerate(model.classify1):
    if hasattr(layer, 'weight'):
        print(f"  Layer {i} ({type(layer).__name__}):")
        print(f"    Weight shape: {layer.weight.shape}")
        print(f"    Input dims: {layer.weight.shape[1]}")
        print(f"    Output dims: {layer.weight.shape[0]}")
print()

print("classify2 layers:")
for i, layer in enumerate(model.classify2):
    if hasattr(layer, 'weight'):
        print(f"  Layer {i} ({type(layer).__name__}):")
        print(f"    Weight shape: {layer.weight.shape}")
        print(f"    Input dims: {layer.weight.shape[1]}")
        print(f"    Output dims: {layer.weight.shape[0]}")
print()

# Test with fake input
h1 = paddle.randn([32, 64])  # [batch_size, hidden_dim]
h2 = paddle.randn([32, 64])
h_combined = paddle.concat([h1, h2], axis=1)
print(f"h_combined shape: {h_combined.shape}")
print()

try:
    output = model.classify1(h_combined)
    print(f"✓ classify1 output shape: {output.shape}")
except Exception as e:
    print(f"✗ classify1 error: {e}")
