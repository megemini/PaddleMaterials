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
Molecular graph data structure compatible with PGL (Paddle Graph Learning).

This module provides graph utilities that replace DGL functionality for PaddlePaddle.
"""

import paddle
import pgl
import numpy as np
from typing import List, Dict, Optional, Union


class MolecularGraph:
    """Molecular graph data structure compatible with PGL.
    
    This class wraps PGL's Graph to provide a DGL-like interface for molecular graphs.
    It stores node features, edge features, and graph topology.
    
    Attributes:
        num_nodes: Number of nodes in the graph
        edges: Edge indices as tuple (src_nodes, dst_nodes)
        node_feat: Dictionary of node features
        edge_feat: Dictionary of edge features
        batch_num_nodes: Number of nodes per graph (for batching)
        batch_num_edges: Number of edges per graph (for batching)
    """
    
    def __init__(
        self,
        num_nodes: int,
        edges: tuple,
        node_feat: Optional[Dict[str, paddle.Tensor]] = None,
        edge_feat: Optional[Dict[str, paddle.Tensor]] = None
    ):
        """Initialize MolecularGraph.
        
        Args:
            num_nodes: Number of nodes in the graph
            edges: Tuple of (src_nodes, dst_nodes) arrays
            node_feat: Dictionary mapping feature names to node feature tensors
            edge_feat: Dictionary mapping feature names to edge feature tensors
        """
        self.num_nodes = num_nodes
        self.edges = edges
        self.node_feat = node_feat if node_feat is not None else {}
        self.edge_feat = edge_feat if edge_feat is not None else {}
        self._graph = None
        
        # Batch information (set by batch_graphs)
        self.batch_num_nodes = None
        self.batch_num_edges = None
    
    @property
    def graph(self) -> pgl.Graph:
        """Get underlying PGL Graph object."""
        if self._graph is None:
            src, dst = self.edges
            self._graph = pgl.Graph(
                num_nodes=self.num_nodes,
                edges=list(zip(src.tolist(), dst.tolist())),
                node_feat=self.node_feat,
                edge_feat=self.edge_feat
            )
        return self._graph
    
    def set_batch_info(self, batch_num_nodes: np.ndarray, batch_num_edges: np.ndarray):
        """Set batch information for batched graphs.
        
        Args:
            batch_num_nodes: Number of nodes in each graph
            batch_num_edges: Number of edges in each graph
        """
        self.batch_num_nodes = batch_num_nodes
        self.batch_num_edges = batch_num_edges
    
    def num_edges(self) -> int:
        """Return number of edges."""
        return len(self.edges[0])
    
    def in_degrees(self) -> paddle.Tensor:
        """Return in-degree of each node."""
        return self.graph.indegree()
    
    def out_degrees(self) -> paddle.Tensor:
        """Return out-degree of each node."""
        return self.graph.outdegree()


def batch_graphs(graphs: List[MolecularGraph]) -> MolecularGraph:
    """Batch multiple molecular graphs into a single graph.
    
    This function replaces dgl.batch() by concatenating multiple graphs with
    node and edge index offsets to maintain graph separation.
    
    Args:
        graphs: List of MolecularGraph objects to batch
        
    Returns:
        Batched MolecularGraph object
        
    Example:
        >>> g1 = MolecularGraph(num_nodes=3, edges=(src1, dst1), ...)
        >>> g2 = MolecularGraph(num_nodes=2, edges=(src2, dst2), ...)
        >>> batched = batch_graphs([g1, g2])
    """
    if len(graphs) == 0:
        raise ValueError("Cannot batch empty list of graphs")
    
    # Track node and edge counts for each graph
    batch_num_nodes = np.array([g.num_nodes for g in graphs])
    batch_num_edges = np.array([g.num_edges() for g in graphs])
    
    # Compute offsets
    node_offset = np.cumsum([0] + batch_num_nodes[:-1].tolist())
    edge_offset = np.cumsum([0] + batch_num_edges[:-1].tolist())
    
    # Concatenate edges with node offsets
    all_src = []
    all_dst = []
    for i, g in enumerate(graphs):
        src, dst = g.edges
        # Convert offsets to the same type as edges
        offset = paddle.to_tensor(node_offset[i], dtype=src.dtype)
        all_src.append(src + offset)
        all_dst.append(dst + offset)
    
    edges = (
        paddle.concat(all_src),
        paddle.concat(all_dst)
    )
    
    # Concatenate node features
    node_feat = {}
    if graphs[0].node_feat:
        for feat_name in graphs[0].node_feat.keys():
            all_feat = [g.node_feat[feat_name] for g in graphs]
            node_feat[feat_name] = paddle.concat(all_feat, axis=0)
    
    # Concatenate edge features
    edge_feat = {}
    if graphs[0].edge_feat:
        for feat_name in graphs[0].edge_feat.keys():
            all_feat = [g.edge_feat[feat_name] for g in graphs]
            edge_feat[feat_name] = paddle.concat(all_feat, axis=0)
    
    # Create batched graph
    total_num_nodes = int(np.sum(batch_num_nodes))
    batched = MolecularGraph(
        num_nodes=total_num_nodes,
        edges=edges,
        node_feat=node_feat,
        edge_feat=edge_feat
    )
    
    # Set batch information
    batched.set_batch_info(batch_num_nodes, batch_num_edges)
    
    return batched


def mean_nodes(graph: MolecularGraph, feat_name: str = "h") -> paddle.Tensor:
    """Compute mean of node features per graph in batch.
    
    This function replaces dgl.mean_nodes() by averaging node features
    within each graph in the batch using batch_num_nodes information.
    
    Args:
        graph: Batched MolecularGraph object
        feat_name: Name of node feature to average (default: "h")
        
    Returns:
        Tensor of shape [batch_size, feat_dim] with averaged features
        
    Example:
        >>> batched = batch_graphs([g1, g2])
        >>> batched.node_feat["h"]  # [total_nodes, feat_dim]
        >>> graph_means = mean_nodes(batched, "h")  # [batch_size, feat_dim]
    """
    if graph.batch_num_nodes is None:
        raise ValueError("Graph is not batched. batch_num_nodes is None.")
    
    node_feats = graph.node_feat[feat_name]  # [total_nodes, feat_dim]
    batch_num_nodes = graph.batch_num_nodes  # [batch_size]
    
    # Split node features by graph
    split_feats = paddle.split(node_feats, batch_num_nodes.tolist())
    
    # Compute mean for each graph
    means = []
    for feats in split_feats:
        means.append(paddle.mean(feats, axis=0, keepdim=True))
    
    result = paddle.concat(means, axis=0)  # [batch_size, feat_dim]
    
    return result


def segment_sum(data: paddle.Tensor, segment_ids: paddle.Tensor, num_segments: int) -> paddle.Tensor:
    """Sum data along segments defined by segment_ids.
    
    This is a utility function for message passing aggregation.
    
    Args:
        data: Input tensor of shape [N, ...]
        segment_ids: Segment indices of shape [N], values in [0, num_segments)
        num_segments: Number of segments
        
    Returns:
        Tensor of shape [num_segments, ...] with summed data per segment
    """
    original_shape = data.shape
    ndim = len(original_shape)
    
    # Flatten all dimensions except the first (segment dimension)
    if ndim > 1:
        feat_dim = int(np.prod(original_shape[1:]))
        data = data.reshape([original_shape[0], feat_dim])
    else:
        feat_dim = 1
    
    segment_ids = segment_ids.reshape([-1])
    
    # Create one-hot encoding
    one_hot = paddle.nn.functional.one_hot(segment_ids, num_segments).cast('float32')
    
    # Sum using matrix multiplication: [num_segments, N] @ [N, feat_dim] -> [num_segments, feat_dim]
    result = paddle.matmul(one_hot.T, data)
    
    # Reshape back to original dimensions (excluding the segment dimension)
    if ndim > 1:
        result = result.reshape([num_segments] + list(original_shape[1:]))
    
    return result


def segment_mean(data: paddle.Tensor, segment_ids: paddle.Tensor, num_segments: int) -> paddle.Tensor:
    """Compute mean of data along segments defined by segment_ids.
    
    Args:
        data: Input tensor of shape [N, ...]
        segment_ids: Segment indices of shape [N], values in [0, num_segments)
        num_segments: Number of segments
        
    Returns:
        Tensor of shape [num_segments, ...] with mean data per segment
    """
    sums = segment_sum(data, segment_ids, num_segments)
    
    # Count elements per segment
    counts = paddle.zeros([num_segments], dtype=data.dtype)
    for i in range(num_segments):
        counts[i] = paddle.sum(segment_ids == i).cast(data.dtype)
    
    # Avoid division by zero
    counts = paddle.maximum(counts, paddle.ones_like(counts))
    
    result = sums / counts.unsqueeze(-1) if len(sums.shape) > 1 else sums / counts
    return result


def segment_max(data: paddle.Tensor, segment_ids: paddle.Tensor, num_segments: int) -> paddle.Tensor:
    """Compute max of data along segments defined by segment_ids.
    
    Args:
        data: Input tensor of shape [N, ...]
        segment_ids: Segment indices of shape [N], values in [0, num_segments)
        num_segments: Number of segments
        
    Returns:
        Tensor of shape [num_segments, ...] with max data per segment
    """
    result = paddle.full([num_segments] + list(data.shape[1:]), -float('inf'), dtype=data.dtype)
    
    for i in range(num_segments):
        mask = segment_ids == i
        if paddle.any(mask):
            result[i] = paddle.max(data[mask], axis=0)
    
    return result


def generate_empty_solvsys(batch_size: int) -> MolecularGraph:
    """Generate an empty solvent system graph for global interaction.
    
    This creates a bipartite graph connecting solvent 1 and solvent 2 representations
    for each batch sample, matching the original GDI-NN architecture.
    
    The graph has:
    - 2 * batch_size nodes (two solvent nodes per batch)
    - Bidirectional edges between solvent pairs
    - Self-loops on each node
    
    Args:
        batch_size: Number of samples in the batch
        
    Returns:
        MolecularGraph with the solvent system topology
    """
    n_solv = 2
    num_nodes = n_solv * batch_size

    # Create edges matching original DGL order:
    #   src = arange(batch_size)           -> [0, 1, ..., batch-1]
    #   dst = arange(batch_size, 2*batch)  -> [batch, batch+1, ..., 2*batch-1]
    #   add_edges(cat(src, dst), cat(dst, src))  -> all src->dst then all dst->src
    #   add_edges(arange(2*batch), arange(2*batch))  -> self-loops
    #
    # Edge order matters because hb_features are indexed by position:
    #   [0..batch-1]: inter_hb (solv1->solv2)
    #   [batch..2*batch-1]: inter_hb (solv2->solv1)
    #   [2*batch..3*batch-1]: intra_hb1 (self-loops on solv1)
    #   [3*batch..4*batch-1]: intra_hb2 (self-loops on solv2)
    src_range = paddle.arange(batch_size, dtype='int64')
    dst_range = paddle.arange(batch_size, num_nodes, dtype='int64')
    all_range = paddle.arange(num_nodes, dtype='int64')

    # Bidirectional edges: cat(src, dst) -> cat(dst, src)
    edge_src = paddle.concat([paddle.concat([src_range, dst_range]), all_range])
    edge_dst = paddle.concat([paddle.concat([dst_range, src_range]), all_range])

    edges = (edge_src, edge_dst)
    
    # Create the graph
    graph = MolecularGraph(
        num_nodes=num_nodes,
        edges=edges,
        node_feat={'h': paddle.zeros([num_nodes, 1])},  # Dummy features
        edge_feat={'e': paddle.zeros([edge_src.shape[0], 1])}  # Dummy edge features
    )
    
    return graph
