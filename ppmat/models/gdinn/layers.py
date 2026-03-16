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
Graph neural network layers for GDI-NN.

This module provides PaddlePaddle implementations of graph neural network layers
that replace DGL's NNConv and GraphConv layers.
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from typing import Optional, Union, Callable

from ppmat.models.gdinn.graph_utils import segment_sum, segment_mean


class NNConv(nn.Layer):
    """Edge-conditioned graph convolution layer.
    
    This layer implements neural network convolution where edge features
    are transformed by an MLP to produce edge-specific message weights.
    This replaces dgl.nn.NNConv for PaddlePaddle.
    
    Reference: Gilmer et al., "Neural Message Passing for Quantum Chemistry"
    (https://arxiv.org/abs/1704.01212)
    
    Args:
        in_feats: Number of input node features
        out_feats: Number of output node features
        edge_func: MLP that transforms edge features to edge weights
        aggregator_type: Aggregation method ('sum', 'mean', 'max')
        bias: Whether to include bias term
    """
    
    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        edge_func: nn.Layer,
        aggregator_type: str = "sum",
        bias: bool = True
    ):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.edge_func = edge_func
        self.aggregator_type = aggregator_type
        
        if aggregator_type not in ["sum", "mean", "max"]:
            raise ValueError(f"Unsupported aggregator_type: {aggregator_type}")
        
        # Bias term
        if bias:
            self.bias = self.create_parameter(
                shape=[out_feats],
                default_initializer=nn.initializer.Constant(0.0)
            )
        else:
            self.bias = None
    
    def forward(
        self,
        graph,
        feat: paddle.Tensor,
        edge_feat: paddle.Tensor
    ) -> paddle.Tensor:
        """Forward pass.
        
        Args:
            graph: MolecularGraph or pgl.Graph object
            feat: Node features of shape [num_nodes, in_feats]
            edge_feat: Edge features of shape [num_edges, edge_feat_dim]
            
        Returns:
            Updated node features of shape [num_nodes, out_feats]
        """
        # Get edge information
        if hasattr(graph, 'edges'):
            # MolecularGraph: edges is a tuple (src, dst)
            src, dst = graph.edges
            num_nodes = graph.num_nodes
            num_edges = len(src)
        else:
            # pgl.Graph: edges is a tensor of shape [num_edges, 2]
            edge_tensor = graph.edges
            src = edge_tensor[:, 0]
            dst = edge_tensor[:, 1]
            num_nodes = graph.num_nodes
            num_edges = len(src)
        
        # Transform edge features to edge weights
        # edge_func(e_ij) returns [num_edges, in_feats * out_feats]
        edge_weights = self.edge_func(edge_feat)
        edge_weights = edge_weights.reshape([num_edges, self.in_feats, self.out_feats])
        
        # Get source node features
        src_feat = feat[src]  # [num_edges, in_feats]
        
        # Compute messages: m_ij = (edge_func(e_ij) * h_j)
        # Using einsum: [num_edges, in_feats] @ [num_edges, in_feats, out_feats] -> [num_edges, out_feats]
        messages = paddle.einsum('ni,nio->no', src_feat, edge_weights)
        
        # Aggregate messages at destination nodes
        if self.aggregator_type == "sum":
            out = segment_sum(messages, dst, num_nodes)
        elif self.aggregator_type == "mean":
            out = segment_mean(messages, dst, num_nodes)
        elif self.aggregator_type == "max":
            # For max aggregation, use unsorted_segment_max (requires custom implementation)
            out = self._segment_max(messages, dst, num_nodes)
        else:
            # Default to sum aggregation
            out = segment_sum(messages, dst, num_nodes)
        
        # Add bias if needed
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    def _segment_max(
        self,
        data: paddle.Tensor,
        segment_ids: paddle.Tensor,
        num_segments: int
    ) -> paddle.Tensor:
        """Custom implementation of segment max."""
        result = paddle.full([num_segments] + list(data.shape[1:]), -float('inf'), dtype=data.dtype)
        
        for i in range(num_segments):
            mask = segment_ids == i
            if paddle.any(mask):
                result[i] = paddle.max(data[mask], axis=0)
        
        return result


class GraphConv(nn.Layer):
    """Standard graph convolution layer (GCN).
    
    This replaces dgl.nn.GraphConv for PaddlePaddle.
    
    Args:
        in_feats: Number of input node features
        out_feats: Number of output node features
        norm: Whether to apply symmetric normalization
        bias: Whether to include bias term
        activation: Activation function (None, 'relu', 'leaky_relu', etc.)
    """
    
    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        norm: bool = True,
        bias: bool = True,
        activation: Optional[str] = None
    ):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.norm = norm
        self.activation = activation
        
        self.weight = self.create_parameter(
            shape=[in_feats, out_feats],
            default_initializer=nn.initializer.XavierUniform()
        )
        
        if bias:
            self.bias = self.create_parameter(
                shape=[out_feats],
                default_initializer=nn.initializer.Constant(0.0)
            )
        else:
            self.bias = None
    
    def forward(self, graph, feat: paddle.Tensor) -> paddle.Tensor:
        """Forward pass.
        
        Args:
            graph: MolecularGraph or pgl.Graph object
            feat: Node features of shape [num_nodes, in_feats]
            
        Returns:
            Updated node features of shape [num_nodes, out_feats]
        """
        # Get edge information
        if hasattr(graph, 'edges'):
            # MolecularGraph: edges is a tuple (src, dst)
            src, dst = graph.edges
            num_nodes = graph.num_nodes
        else:
            # pgl.Graph: edges is a tensor of shape [num_edges, 2]
            edge_tensor = graph.edges
            src = edge_tensor[:, 0]
            dst = edge_tensor[:, 1]
            num_nodes = graph.num_nodes
        
        # Transform node features
        feat = feat @ self.weight
        
        # Message passing
        if self.norm:
            # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
            if hasattr(graph, 'in_degrees'):
                deg = graph.in_degrees().cast('float32')
            else:
                deg = graph.indegree().cast('float32')
            norm_coeff = paddle.pow(deg, -0.5)
            norm_coeff = paddle.where(
                paddle.isinf(norm_coeff),
                paddle.zeros_like(norm_coeff),
                norm_coeff
            )
            
            # Get source features and normalize them
            src_feat = feat[src]  # [num_edges, out_feats]
            src_norm = norm_coeff[src]  # [num_edges]
            src_feat = src_feat * src_norm.unsqueeze(-1)  # [num_edges, out_feats]
            
            # Aggregate normalized messages
            out = segment_sum(src_feat, dst, num_nodes)
            
            # Normalize by destination degrees
            dst_norm = norm_coeff  # [num_nodes]
            out = out * dst_norm.unsqueeze(-1)  # [num_nodes, out_feats]
        else:
            # No normalization
            src_feat = feat[src]
            out = segment_sum(src_feat, dst, num_nodes)
        
        # Add bias if needed
        if self.bias is not None:
            out = out + self.bias
        
        # Apply activation
        if self.activation == 'relu':
            out = F.relu(out)
        elif self.activation == 'leaky_relu':
            out = F.leaky_relu(out)
        elif self.activation == 'sigmoid':
            out = F.sigmoid(out)
        elif self.activation == 'tanh':
            out = F.tanh(out)
        
        return out


class MPNNConv(nn.Layer):
    """Message Passing Neural Network convolution layer.
    
    This is the core component of GDI-NN, combining NNConv with GRU
    for iterative message passing over multiple steps.
    
    Args:
        node_in_feats: Input node feature dimension
        edge_in_feats: Input edge feature dimension
        node_out_feats: Output node feature dimension
        edge_hidden_feats: Hidden dimension for edge function
        num_step_message_passing: Number of message passing steps
        activation: Activation function ('relu', 'leaky_relu', 'sigmoid', 'tanh', 'softplus')
        dropout: Dropout rate (0.0 means no dropout)
    """
    
    def __init__(
        self,
        node_in_feats: int,
        edge_in_feats: int,
        node_out_feats: int = 128,
        edge_hidden_feats: int = 32,
        num_step_message_passing: int = 6,
        activation: str = "relu",
        dropout: float = 0.0
    ):
        super().__init__()
        self.node_in_feats = node_in_feats
        self.edge_in_feats = edge_in_feats
        self.node_out_feats = node_out_feats
        self.num_step_message_passing = num_step_message_passing
        self.activation = activation
        
        # Input projection if node_in_feats != node_out_feats
        # (needed because iterative message passing reuses output as input)
        if node_in_feats != node_out_feats:
            self.input_proj = nn.Linear(node_in_feats, node_out_feats)
        else:
            self.input_proj = None

        # Edge function MLP: transforms edge features to edge weights
        # Uses node_out_feats for both in/out since iterative steps use projected features
        self.edge_func = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hidden_feats),
            self._get_activation_layer(),
            nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats)
        )

        # NNConv layer
        self.gnn_layer = NNConv(
            in_feats=node_out_feats,
            out_feats=node_out_feats,
            edge_func=self.edge_func,
            aggregator_type="sum"
        )
        
        # GRU for updating node representations
        self.gru = nn.GRU(node_out_feats, node_out_feats)
        
        # Dropout
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(node_out_feats)
    
    def _get_activation_layer(self) -> nn.Layer:
        """Get activation layer based on activation name."""
        if self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "leaky_relu":
            return nn.LeakyReLU()
        elif self.activation == "sigmoid":
            return nn.Sigmoid()
        elif self.activation == "tanh":
            return nn.Tanh()
        elif self.activation == "softplus":
            return nn.Softplus()
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
    
    def forward(
        self,
        graph,
        node_feats: paddle.Tensor,
        edge_feats: paddle.Tensor
    ) -> paddle.Tensor:
        """Forward pass with iterative message passing.
        
        Args:
            graph: MolecularGraph or pgl.Graph object
            node_feats: Node features of shape [num_nodes, node_in_feats]
            edge_feats: Edge features of shape [num_edges, edge_in_feats]
            
        Returns:
            Updated node features of shape [num_nodes, node_out_feats]
        """
        num_nodes = graph.num_nodes if hasattr(graph, 'num_nodes') else graph.graph.num_nodes
        
        # Project input features if needed
        if self.input_proj is not None:
            node_feats = self.input_proj(node_feats)

        # Initialize hidden state for GRU: [num_layers=1, num_nodes, node_out_feats]
        hidden = paddle.zeros([1, num_nodes, self.node_out_feats])

        # Message passing for multiple steps
        for step in range(self.num_step_message_passing):
            # Apply GNN layer
            new_h = self.gnn_layer(graph, node_feats, edge_feats)

            # Apply layer norm and activation
            new_h = self.layer_norm(new_h)
            new_h = self._get_activation_layer()(new_h)

            # Apply dropout
            if self.dropout is not None:
                new_h = self.dropout(new_h)

            # GRU update: input [batch=num_nodes, seq_len=1, feat], hidden [1, num_nodes, feat]
            new_h = new_h.unsqueeze(1)  # [num_nodes, 1, node_out_feats]
            out, hidden = self.gru(new_h, hidden)
            node_feats = out.squeeze(1)  # [num_nodes, node_out_feats]

        return node_feats
