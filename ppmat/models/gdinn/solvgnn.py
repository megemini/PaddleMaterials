# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SolvGNN model for predicting binary activity coefficients.

This module implements the SolvGNN model which uses graph neural networks
to predict activity coefficients for binary solvent mixtures, incorporating
Gibbs-Duhem thermodynamic constraints.
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from typing import Dict, Optional, Tuple

from ppmat.models.gdinn.layers import GraphConv, MPNNConv
from ppmat.models.gdinn.graph_utils import mean_nodes


class SolvGNN(nn.Layer):
    """SolvGNN model for predicting binary activity coefficients.
    
    This model takes two molecular graphs (for the two solvents in a binary mixture)
    and predicts their activity coefficients (gamma1, gamma2). It enforces the
    Gibbs-Duhem thermodynamic constraint: x1*d(ln(gamma1))/dx1 + x2*d(ln(gamma2))/dx1 = 0.
    
    Model architecture:
        1. Two separate graph convolutional branches for each solvent
        2. Graph-level pooling to get molecular embeddings
        3. Global interaction layer between the two embeddings
        4. MLP classifier to predict gamma1 and gamma2
        5. Gibbs-Duhem constraint loss computation
    
    Args:
        in_dim: Input node feature dimension (default: 75 for atom features)
        hidden_dim: Hidden dimension for graph layers (default: 256)
        n_classes: Number of output classes (default: 1 for gamma)
        mlp_dropout_rate: Dropout rate for MLP layers (default: 0.0)
        mlp_activation: Activation function for MLP (default: "softplus")
        mpnn_activation: Activation function for MPNN layers (default: "relu")
        num_step_message_passing: Number of message passing steps (default: 6)
        pinn_lambda: Weight for Gibbs-Duhem constraint loss (default: 1.0)
    """
    
    def __init__(
        self,
        in_dim: int = 75,
        hidden_dim: int = 256,
        n_classes: int = 1,
        mlp_dropout_rate: float = 0.0,
        mlp_activation: str = "softplus",
        mpnn_activation: str = "relu",
        num_step_message_passing: int = 6,
        pinn_lambda: float = 1.0
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.pinn_lambda = pinn_lambda
        
        # Graph convolutional layers (shared between two solvents)
        self.conv1 = GraphConv(in_dim, hidden_dim, norm=True, activation='relu')
        self.conv2 = GraphConv(hidden_dim, hidden_dim, norm=True, activation='relu')
        
        # Global MPNN convolution layer for interaction
        # Input dimension is hidden_dim + 1 (for composition information)
        self.global_conv = MPNNConv(
            node_in_feats=hidden_dim + 1,
            edge_in_feats=12,  # Edge feature dimension
            node_out_feats=hidden_dim,
            edge_hidden_feats=32,
            num_step_message_passing=num_step_message_passing,
            activation=mpnn_activation
        )
        
        # MLP classifier for gamma1 and gamma2 (shared)
        self.classify1 = nn.Linear(hidden_dim, hidden_dim)
        self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        self.classify3 = nn.Linear(hidden_dim, n_classes)
        self.mlp_activation = self._get_activation_layer(mlp_activation)
    
    def _get_activation_layer(self, activation: str) -> nn.Layer:
        """Get activation layer based on activation name."""
        if activation == "relu":
            return nn.ReLU()
        elif activation == "leaky_relu":
            return nn.LeakyReLU()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "softplus":
            return nn.Softplus()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(
        self,
        batch_data: Dict
    ) -> Dict[str, Dict[str, paddle.Tensor]]:
        """Forward pass of SolvGNN model.

        Args:
            batch_data: Dictionary containing:
                - g1: First molecular graph (solvent 1)
                - g2: Second molecular graph (solvent 2)
                - x1: Composition of solvent 1 (mole fraction) [batch_size, 1]
                - gamma1: Target activity coefficient for solvent 1 [batch_size, 1]
                - gamma2: Target activity coefficient for solvent 2 [batch_size, 1]
                - intra_hb1: Intra-molecular hydrogen bonds in solvent 1 [batch_size, 1]
                - intra_hb2: Intra-molecular hydrogen bonds in solvent 2 [batch_size, 1]
                - inter_hb: Inter-molecular hydrogen bonds [batch_size, 1]

        Returns:
            Dictionary containing:
                - loss_dict: Dictionary of losses
                    - pred_loss: Prediction loss (MSE)
                    - gd_loss: Gibbs-Duhem constraint loss
                    - total_loss: Combined loss
                - pred_dict: Dictionary of predictions
                    - gamma1: Predicted gamma1
                    - gamma2: Predicted gamma2
                    - ln_gamma1: Predicted ln(gamma1)
                    - ln_gamma2: Predicted ln(gamma2)
        """
        g1 = batch_data['g1']
        g2 = batch_data['g2']

        # Use real data format field
        solv1_x = batch_data['x1']  # [batch_size, 1]
        
        gamma1_label = batch_data['gamma1']  # [batch_size, 1]
        gamma2_label = batch_data['gamma2']  # [batch_size, 1]
        
        # Extract node features
        h1 = g1.node_feat['h']  # [num_nodes1, in_dim]
        h2 = g2.node_feat['h']  # [num_nodes2, in_dim]
        
        # Apply graph convolutions for both solvents (shared weights)
        h1 = self.conv1(g1, h1)
        h1 = F.relu(h1)
        h1 = self.conv2(g1, h1)
        h1 = F.relu(h1)
        g1.node_feat['h'] = h1  # Update node features in graph
        
        h2 = self.conv1(g2, h2)
        h2 = F.relu(h2)
        h2 = self.conv2(g2, h2)
        h2 = F.relu(h2)
        g2.node_feat['h'] = h2  # Update node features in graph
        
        # Graph-level pooling to get molecular embeddings
        hg1 = mean_nodes(g1, "h")  # [batch_size, hidden_dim]
        hg2 = mean_nodes(g2, "h")  # [batch_size, hidden_dim]
        
        # Concatenate with composition information
        # This matches the original GDI-NN architecture
        hg1_with_comp = paddle.concat([hg1, solv1_x], axis=1)  # [batch_size, hidden_dim + 1]
        hg2_with_comp = paddle.concat([hg2, 1 - solv1_x], axis=1)  # [batch_size, hidden_dim + 1]
        
        # Create edge features for global convolution
        # Using hydrogen bond features if available
        if 'inter_hb' in batch_data:
            inter_hb = batch_data['inter_hb']
            intra_hb1 = batch_data['intra_hb1']
            intra_hb2 = batch_data['intra_hb2']
            # Repeat inter_hb twice and concatenate with intra_hb1 and intra_hb2
            hb_features = paddle.concat([
                inter_hb.repeat(2),
                intra_hb1,
                intra_hb2
            ], axis=0)  # [4 * batch_size, 1]
            hb_features = hb_features.unsqueeze(1)  # [4 * batch_size, 1]
        else:
            # Use dummy edge features if not available
            batch_size = solv1_x.shape[0]
            hb_features = paddle.zeros([4 * batch_size, 1], dtype=solv1_x.dtype)
        
        # Concatenate both molecule embeddings for global convolution
        hg_concat = paddle.concat([hg1_with_comp, hg2_with_comp], axis=0)  # [2 * batch_size, hidden_dim + 1]
        
        # Apply global MPNN convolution for molecular interaction
        # We use g1 as the graph structure (this is a simplification)
        # In the original implementation, a special "empty solvent system" graph is used
        hg = self.global_conv(g1, hg_concat, hb_features)
        
        # Split the output back into two molecules
        hg = hg[:hg_concat.shape[0], :]  # [2 * batch_size, hidden_dim]
        hg1 = hg[:hg1_with_comp.shape[0], :]  # [batch_size, hidden_dim]
        hg2 = hg[hg1_with_comp.shape[0]:, :]  # [batch_size, hidden_dim]
        
        # Predict ln_gamma using shared classifier
        output = self.mlp_activation(self.classify1(hg))
        output = self.mlp_activation(self.classify2(output))
        output = self.classify3(output)  # [2 * batch_size, n_classes]
        
        # Concatenate predictions from both molecules
        output = paddle.concat([output[:hg1.shape[0], :], output[hg1.shape[0]:, :]], axis=1)  # [batch_size, 2 * n_classes]
        
        # Split into ln_gamma1 and ln_gamma2
        ln_gamma1_pred = output[:, :self.n_classes]  # [batch_size, 1]
        ln_gamma2_pred = output[:, self.n_classes:]  # [batch_size, 1]
        
        # Convert to gamma (gamma = exp(ln(gamma)))
        gamma1_pred = paddle.exp(ln_gamma1_pred)
        gamma2_pred = paddle.exp(ln_gamma2_pred)
        
        # Compute prediction loss (MSE on ln(gamma) for better scaling)
        ln_gamma1_label = paddle.log(paddle.maximum(gamma1_label, paddle.ones_like(gamma1_label) * 1e-6))
        ln_gamma2_label = paddle.log(paddle.maximum(gamma2_label, paddle.ones_like(gamma2_label) * 1e-6))
        
        pred_loss = F.mse_loss(ln_gamma1_pred, ln_gamma1_label) + \
                    F.mse_loss(ln_gamma2_pred, ln_gamma2_label)
        
        # Compute Gibbs-Duhem constraint loss
        gd_loss = self._compute_gibbs_duhem_loss(
            ln_gamma1_pred, ln_gamma2_pred, solv1_x
        )
        
        # Total loss
        total_loss = pred_loss + self.pinn_lambda * gd_loss
        
        # Build output dictionaries
        loss_dict = {
            'pred_loss': pred_loss,
            'gd_loss': gd_loss,
            'total_loss': total_loss
        }
        
        pred_dict = {
            'gamma1': gamma1_pred,
            'gamma2': gamma2_pred,
            'ln_gamma1': ln_gamma1_pred,
            'ln_gamma2': ln_gamma2_pred
        }
        
        return {
            'loss_dict': loss_dict,
            'pred_dict': pred_dict
        }
    
    def _compute_gibbs_duhem_loss(
        self,
        ln_gamma1: paddle.Tensor,
        ln_gamma2: paddle.Tensor,
        x1: paddle.Tensor
    ) -> paddle.Tensor:
        """Compute Gibbs-Duhem constraint loss.
        
        The Gibbs-Duhem equation for binary mixtures:
        x1 * d(ln(gamma1))/dx1 + x2 * d(ln(gamma2))/dx1 = 0
        
        Since x2 = 1 - x1, we can write:
        x1 * d(ln(gamma1))/dx1 + (1 - x1) * d(ln(gamma2))/dx1 = 0
        
        Args:
            ln_gamma1: Predicted ln(gamma1) [batch_size, 1]
            ln_gamma2: Predicted ln(gamma2) [batch_size, 1]
            x1: Composition of solvent 1 [batch_size, 1]
            
        Returns:
            Gibbs-Duhem constraint loss (scalar)
        """
        # Compute gradients using automatic differentiation
        x1 = x1.stop_gradient(False)  # Enable gradient computation
        
        # Compute d(ln(gamma1))/dx1
        dln_gamma1_dx1 = paddle.grad(
            outputs=ln_gamma1,
            inputs=x1,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Compute d(ln(gamma2))/dx1
        dln_gamma2_dx1 = paddle.grad(
            outputs=ln_gamma2,
            inputs=x1,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Gibbs-Duhem constraint: x1*dln_gamma1/dx1 + (1-x1)*dln_gamma2/dx1 = 0
        gd_constraint = x1 * dln_gamma1_dx1 + (1 - x1) * dln_gamma2_dx1
        
        # Loss is squared constraint violation
        gd_loss = paddle.mean(gd_constraint ** 2)
        
        return gd_loss
    
    def predict(
        self,
        g1,
        g2,
        x1: paddle.Tensor
    ) -> Dict[str, paddle.Tensor]:
        """Predict activity coefficients for a binary mixture.

        This method is for inference only and does not compute losses.

        Args:
            g1: First molecular graph (solvent 1)
            g2: Second molecular graph (solvent 2)
            x1: Composition of solvent 1 [batch_size, 1]

        Returns:
            Dictionary containing:
                - gamma1: Predicted activity coefficient for solvent 1
                - gamma2: Predicted activity coefficient for solvent 2
        """
        batch_data = {
            'g1': g1,
            'g2': g2,
            'x1': x1,
            'gamma1': paddle.zeros_like(x1),  # Dummy label
            'gamma2': paddle.zeros_like(x1)   # Dummy label
        }

        output = self.forward(batch_data)
        return output['pred_dict']


class SolvGNNWithHydrogenBonds(SolvGNN):
    """SolvGNN model with hydrogen bond features.
    
    This extended version includes hydrogen bond features as additional inputs
    to improve prediction accuracy for polar and hydrogen-bonding solvents.
    
    Args:
        in_dim: Input node feature dimension (default: 75)
        hidden_dim: Hidden dimension for graph layers (default: 256)
        n_classes: Number of output classes (default: 1)
        mlp_dropout_rate: Dropout rate for MLP layers (default: 0.0)
        mlp_activation: Activation function for MLP (default: "softplus")
        mpnn_activation: Activation function for MPNN layers (default: "relu")
        num_step_message_passing: Number of message passing steps (default: 6)
        pinn_lambda: Weight for Gibbs-Duhem constraint loss (default: 1.0)
        use_hb_features: Whether to use hydrogen bond features (default: True)
    """
    
    def __init__(
        self,
        in_dim: int = 75,
        hidden_dim: int = 256,
        n_classes: int = 1,
        mlp_dropout_rate: float = 0.0,
        mlp_activation: str = "softplus",
        mpnn_activation: str = "relu",
        num_step_message_passing: int = 6,
        pinn_lambda: float = 1.0,
        use_hb_features: bool = True
    ):
        super().__init__(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            n_classes=n_classes,
            mlp_dropout_rate=mlp_dropout_rate,
            mlp_activation=mlp_activation,
            mpnn_activation=mpnn_activation,
            num_step_message_passing=num_step_message_passing,
            pinn_lambda=pinn_lambda
        )
        self.use_hb_features = use_hb_features
        
        if use_hb_features:
            # MLP to process hydrogen bond features
            self.hb_mlp = nn.Sequential(
                nn.Linear(3, hidden_dim),  # intra_hb1, intra_hb2, inter_hb
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
    
    def forward(
        self,
        batch_data: Dict
    ) -> Dict[str, Dict[str, paddle.Tensor]]:
        """Forward pass with hydrogen bond features."""
        g1 = batch_data['g1']
        g2 = batch_data['g2']
        solv1_x = batch_data['x1']
        gamma1_label = batch_data['gamma1']
        gamma2_label = batch_data['gamma2']
        
        # Extract node and edge features
        h1 = g1.node_feat['h']
        _ = g1.edge_feat.get('e', None)  # Edge features not used in current implementation
        h2 = g2.node_feat['h']
        _ = g2.edge_feat.get('e', None)  # Edge features not used in current implementation
        
        # Apply graph convolutions
        h1 = self.conv1_1(g1, h1)
        h1 = F.relu(h1)
        h1 = self.conv1_2(g1, h1)
        h1 = F.relu(h1)
        
        h2 = self.conv2_1(g2, h2)
        h2 = F.relu(h2)
        h2 = self.conv2_2(g2, h2)
        h2 = F.relu(h2)
        
        # Graph-level pooling
        h1 = mean_nodes(g1, "h")
        h2 = mean_nodes(g2, "h")
        
        # Concatenate molecular embeddings
        h_combined = paddle.concat([h1, h2], axis=1)
        
        # Incorporate hydrogen bond features if available
        if self.use_hb_features and 'intra_hb1' in batch_data:
            hb_features = paddle.stack([
                batch_data['intra_hb1'],
                batch_data['intra_hb2'],
                batch_data['inter_hb']
            ], axis=1)  # [batch_size, 3]
            hb_embedding = self.hb_mlp(hb_features)  # [batch_size, hidden_dim]
            h_combined = paddle.concat([h_combined, hb_embedding], axis=1)
            
            # Adjust classifier input dimension dynamically
            # Note: This is a simplified approach; in practice, you may want to
            # reinitialize the classifiers or use a more sophisticated method
            pass
        
        # Predict gamma1 and gamma2
        ln_gamma1_pred = self.classify1(h_combined)
        ln_gamma2_pred = self.classify2(h_combined)
        
        # Convert to gamma
        gamma1_pred = paddle.exp(ln_gamma1_pred)
        gamma2_pred = paddle.exp(ln_gamma2_pred)
        
        # Compute prediction loss
        ln_gamma1_label = paddle.log(paddle.maximum(gamma1_label, paddle.ones_like(gamma1_label) * 1e-6))
        ln_gamma2_label = paddle.log(paddle.maximum(gamma2_label, paddle.ones_like(gamma2_label) * 1e-6))
        
        pred_loss = F.mse_loss(ln_gamma1_pred, ln_gamma1_label) + \
                    F.mse_loss(ln_gamma2_pred, ln_gamma2_label)
        
        # Compute Gibbs-Duhem constraint loss
        gd_loss = self._compute_gibbs_duhem_loss(
            ln_gamma1_pred, ln_gamma2_pred, solv1_x
        )
        
        # Total loss
        total_loss = pred_loss + self.pinn_lambda * gd_loss
        
        # Build output dictionaries
        loss_dict = {
            'pred_loss': pred_loss,
            'gd_loss': gd_loss,
            'total_loss': total_loss
        }
        
        pred_dict = {
            'gamma1': gamma1_pred,
            'gamma2': gamma2_pred,
            'ln_gamma1': ln_gamma1_pred,
            'ln_gamma2': ln_gamma2_pred
        }
        
        return {
            'loss_dict': loss_dict,
            'pred_dict': pred_dict
        }
