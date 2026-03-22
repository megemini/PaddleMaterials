from typing import Callable
from typing import Optional

import paddle
from paddle.nn.functional import swish

from ppmat.models.common.initializer import he_orthogonal_init
from ppmat.models.common.initializer import xavier_uniform_
from ppmat.models.common.initializer import zeros_
from ppmat.models.spherenet.features import dist_emb
from ppmat.models.spherenet.features import angle_emb
from ppmat.models.spherenet.features import torsion_emb
from ppmat.utils.crystal import get_pbc_distances
from ppmat.utils.scatter import scatter

"""This module is adapted from https://github.com/divelab/DIG/tree/dig-stable/dig/threedgraph/method/spherenet
"""


class ResidualLayer(paddle.nn.Layer):
    def __init__(self, hidden_channels: int, act: Callable):
        super().__init__()
        self.act = act
        self.lin1 = paddle.nn.Linear(
            in_features=hidden_channels, out_features=hidden_channels
        )
        self.lin2 = paddle.nn.Linear(
            in_features=hidden_channels, out_features=hidden_channels
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class init(paddle.nn.Layer):
    def __init__(self, num_radial, hidden_channels, act, use_node_features=True):
        super().__init__()
        self.act = act
        self.use_node_features = use_node_features
        if self.use_node_features:
            self.emb = paddle.nn.Embedding(
                num_embeddings=95, embedding_dim=hidden_channels
            )
        else:
            self.node_emb = paddle.create_parameter(
                shape=[1, hidden_channels],
                dtype='float32',
                default_initializer=paddle.nn.initializer.XavierUniform(),
            )
            self.node_emb.stop_gradient = False
        self.lin_rbf_0 = paddle.nn.Linear(
            in_features=num_radial, out_features=hidden_channels
        )
        self.lin = paddle.nn.Linear(
            in_features=3 * hidden_channels, out_features=hidden_channels
        )
        self.lin_rbf_1 = paddle.nn.Linear(
            in_features=num_radial, out_features=hidden_channels, bias_attr=False
        )

    def forward(self, x, emb, i, j):
        rbf, _, _ = emb
        if self.use_node_features:
            x = self.emb(x)
        else:
            x = self.node_emb.expand([x.shape[0], -1])
        rbf0 = self.act(self.lin_rbf_0(rbf))
        e1 = self.act(self.lin(paddle.concat(x=[x[i], x[j], rbf0], axis=-1)))
        e2 = self.lin_rbf_1(rbf) * e1
        return e1, e2


class update_e(paddle.nn.Layer):
    def __init__(
        self,
        hidden_channels,
        int_emb_size,
        basis_emb_size_dist,
        basis_emb_size_angle,
        basis_emb_size_torsion,
        num_spherical,
        num_radial,
        num_before_skip,
        num_after_skip,
        act,
    ):
        super().__init__()
        self.act = act
        self.hidden_channels = hidden_channels

        # dist
        self.lin_dist1 = paddle.nn.Linear(
            in_features=num_radial, out_features=basis_emb_size_dist, bias_attr=False
        )
        self.lin_dist2 = paddle.nn.Linear(
            in_features=basis_emb_size_dist, out_features=hidden_channels, bias_attr=False
        )

        # angle
        self.lin_angle1 = paddle.nn.Linear(
            in_features=num_spherical * num_radial,
            out_features=basis_emb_size_angle,
            bias_attr=False,
        )
        self.lin_angle2 = paddle.nn.Linear(
            in_features=basis_emb_size_angle,
            out_features=int_emb_size,
            bias_attr=False,
        )

        # torsion
        self.lin_torsion1 = paddle.nn.Linear(
            in_features=num_spherical * num_spherical * num_radial,
            out_features=basis_emb_size_torsion,
            bias_attr=False,
        )
        self.lin_torsion2 = paddle.nn.Linear(
            in_features=basis_emb_size_torsion,
            out_features=int_emb_size,
            bias_attr=False,
        )

        # message layers
        self.lin_kj = paddle.nn.Linear(
            in_features=hidden_channels, out_features=hidden_channels
        )
        self.lin_ji = paddle.nn.Linear(
            in_features=hidden_channels, out_features=hidden_channels
        )

        self.lin_down = paddle.nn.Linear(
            in_features=hidden_channels, out_features=int_emb_size, bias_attr=False
        )
        self.lin_up = paddle.nn.Linear(
            in_features=int_emb_size, out_features=hidden_channels, bias_attr=False
        )

        self.layers_before_skip = paddle.nn.LayerList(
            sublayers=[
                ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)
            ]
        )
        self.lin_skip = paddle.nn.Linear(
            in_features=hidden_channels, out_features=hidden_channels
        )
        self.layers_after_skip = paddle.nn.LayerList(
            sublayers=[
                ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)
            ]
        )
        self.lin_rbf = paddle.nn.Linear(
            in_features=num_radial, out_features=hidden_channels, bias_attr=False
        )

    def forward(self, e, emb, idx_kj, idx_ji):
        rbf0, abf, tbf = emb
        x1, _ = e

        x_ji = self.act(self.lin_ji(x1))
        x_kj = self.act(self.lin_kj(x1))

        # dist interaction
        rbf = self.lin_dist1(rbf0)
        rbf = self.lin_dist2(rbf)
        x_kj = x_kj * rbf

        x_kj = self.act(self.lin_down(x_kj))

        # angle interaction
        abf = self.lin_angle1(abf)
        abf = self.lin_angle2(abf)
        x_kj = x_kj[idx_kj] * abf

        # torsion interaction
        tbf = self.lin_torsion1(tbf)
        tbf = self.lin_torsion2(tbf)
        x_kj = x_kj * tbf

        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x1.shape[0])
        x_kj = self.act(self.lin_up(x_kj))

        e1 = x_ji + x_kj
        for layer in self.layers_before_skip:
            e1 = layer(e1)
        e1 = self.act(self.lin_skip(e1)) + x1
        for layer in self.layers_after_skip:
            e1 = layer(e1)
        e2 = self.lin_rbf(rbf0) * e1

        return e1, e2


class update_v(paddle.nn.Layer):
    def __init__(
        self,
        hidden_channels,
        out_emb_channels,
        out_channels,
        num_output_layers,
        act,
        output_init,
    ):
        super().__init__()
        self.act = act
        self.output_init = output_init

        self.lin_up = paddle.nn.Linear(
            in_features=hidden_channels, out_features=out_emb_channels, bias_attr=True
        )
        self.lins = paddle.nn.LayerList()
        for _ in range(num_output_layers):
            self.lins.append(
                paddle.nn.Linear(
                    in_features=out_emb_channels, out_features=out_emb_channels
                )
            )
        self.lin = paddle.nn.Linear(
            in_features=out_emb_channels, out_features=out_channels, bias_attr=False
        )

        self.reset_parameters()

    def reset_parameters(self):
        if self.output_init == 'GlorotOrthogonal':
            he_orthogonal_init(self.lin.weight)
        elif self.output_init == 'zeros':
            zeros_(self.lin.weight)

    def forward(self, e, i):
        _, e2 = e
        v = scatter(e2, i, dim=0)
        v = self.lin_up(v)
        for lin in self.lins:
            v = self.act(lin(v))
        v = self.lin(v)
        return v


class update_u(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, u, v, batch):
        u += scatter(v, batch, dim=0)
        return u


class emb(paddle.nn.Layer):
    def __init__(self, num_spherical, num_radial, cutoff, envelope_exponent):
        super().__init__()
        self.dist_emb = dist_emb(num_radial, cutoff, envelope_exponent)
        self.angle_emb = angle_emb(
            num_spherical, num_radial, cutoff, envelope_exponent
        )
        self.torsion_emb = torsion_emb(
            num_spherical, num_radial, cutoff, envelope_exponent
        )

    def forward(self, dist, angle, torsion, idx_kj):
        dist_emb_out = self.dist_emb(dist)
        angle_emb_out = self.angle_emb(dist, angle, idx_kj)
        torsion_emb_out = self.torsion_emb(dist, angle, torsion, idx_kj)
        return dist_emb_out, angle_emb_out, torsion_emb_out


def xyz_to_dat(pos, edge_index, num_nodes, offsets=None, use_torsion=True):
    j, i = edge_index

    # compute distances (already done externally, but we need vectors)
    if offsets is not None:
        dist_vec = pos[j] - pos[i] + offsets
    else:
        dist_vec = pos[j] - pos[i]
    dist = dist_vec.norm(axis=-1)

    # build triplet indices
    value = paddle.arange(end=j.shape[0], dtype='int64')
    # for each node, gather the edge indices pointing to it
    n = j.shape[0]
    rows = paddle.arange(end=n).unsqueeze(1)
    cols = paddle.arange(end=n).unsqueeze(0)
    mask = (i.unsqueeze(1) == i.unsqueeze(0)) & (cols <= rows)
    col_ = mask.astype('int64').sum(axis=1) - 1
    adj_value = value + 1  # shift by 1 to distinguish from 0
    mat = paddle.scatter_nd(
        paddle.stack(x=[i, col_], axis=1),
        adj_value,
        shape=[num_nodes, col_.max().item() + 1],
    )

    idx_kj = mat[j][mat[j] > 0] - 1
    tmp = paddle.nonzero(mat[j], as_tuple=False)
    idx_ji = tmp[:, 0]

    idx_k = paddle.index_select(j, idx_kj, axis=0)
    idx_j = paddle.index_select(j, idx_ji, axis=0)
    idx_i = paddle.index_select(i, idx_ji, axis=0)

    # remove self-loops in triplets
    mask2 = idx_i != idx_k
    idx_kj = idx_kj[mask2]
    idx_ji = idx_ji[mask2]
    idx_k = idx_k[mask2]
    idx_j = idx_j[mask2]
    idx_i = idx_i[mask2]

    # compute angles
    if offsets is not None:
        pos_ji = pos[idx_i] - pos[idx_j] + offsets[idx_ji]
        pos_kj = pos[idx_k] - pos[idx_j] + offsets[idx_kj]
    else:
        pos_ji = pos[idx_i] - pos[idx_j]
        pos_kj = pos[idx_k] - pos[idx_j]

    a = (pos_ji * pos_kj).sum(axis=-1)
    b = paddle.cross(pos_ji, pos_kj).norm(axis=-1)
    angle = paddle.atan2(b, a)

    # compute torsion angles
    if use_torsion:
        # compute dihedral/torsion angle
        dist_ji = pos_ji.norm(axis=-1, keepdim=True) + 1e-7
        pos_ji_normalized = pos_ji / dist_ji
        # plane normal of (ji, kj)
        plane1 = paddle.cross(pos_ji, pos_kj)
        plane1_norm = plane1.norm(axis=-1, keepdim=True) + 1e-7
        plane1 = plane1 / plane1_norm
        # for torsion we need another triplet layer
        # use the cross product with the normalized ji direction
        plane2 = paddle.cross(pos_ji_normalized, plane1)
        # torsion = atan2 of projections
        cos_torsion = (pos_kj * plane1).sum(axis=-1)
        sin_torsion = (pos_kj * plane2).sum(axis=-1)
        torsion = paddle.atan2(sin_torsion, cos_torsion)
        torsion = torsion + paddle.to_tensor(
            3.141592653589793, dtype=torsion.dtype)
        return dist, angle, torsion, i, j, idx_kj, idx_ji
    else:
        return dist, angle, i, j, idx_kj, idx_ji


class SphereNet(paddle.nn.Layer):
    """
    Spherical Message Passing for 3D Graph Networks,
    https://arxiv.org/abs/2102.05013

    Args:
        out_channels (int): The number of output channels for the final prediction.
        hidden_channels (int, optional): The dimensionality of hidden feature
            vectors in each convolutional layer. Defaults to 128.
        num_blocks (int, optional): The number of interaction blocks to stack.
            Defaults to 4.
        int_emb_size (int, optional): The size of the intermediate embedding.
            Defaults to 64.
        basis_emb_size_dist (int, optional): The size of the distance basis
            embedding. Defaults to 8.
        basis_emb_size_angle (int, optional): The size of the angle basis
            embedding. Defaults to 8.
        basis_emb_size_torsion (int, optional): The size of the torsion basis
            embedding. Defaults to 8.
        out_emb_channels (int, optional): The number of channels after the final
            embedding layer before readout. Defaults to 256.
        num_spherical (int, optional): The number of spherical basis functions to use.
            Defaults to 7.
        num_embeddings (int, optional): The number of distinct atom types to embed.
            Defaults to 95.
        num_radial (int, optional): The number of radial basis functions to use.
            Defaults to 6.
        otf_graph (bool, optional): Whether to construct the interaction graph
            on-the-fly during training. Defaults to False.
        cutoff (float, optional): The cutoff distance for neighbor interactions.
            Defaults to 5.0.
        max_num_neighbors (int, optional): The maximum number of neighbors to consider
            for each atom. Defaults to 50.
        envelope_exponent (int, optional): The exponent used in the cutoff envelope
            function. Defaults to 5.
        num_before_skip (int, optional): The number of residual layers before skip
            connection. Defaults to 1.
        num_after_skip (int, optional): The number of residual layers after skip
            connection. Defaults to 2.
        num_output_layers (int, optional): The number of output layers. Defaults to 3.
        readout (str, optional): Readout method ("mean" or "sum"). Defaults to "mean".
        property_names (Optional[str], optional): Target property names to predict.
            Defaults to "formation_energy_per_atom".
        data_mean (float, optional): The mean used for normalizing target values.
            Defaults to 0.0.
        data_std (float, optional): The standard deviation used for normalizing
            target values. Defaults to 1.0.
        loss_type (str, optional): Loss type, can be 'mse_loss' or 'l1_loss'.
            Defaults to "l1_loss".
        act (str, optional): The activation function. Defaults to "swish".
        output_init (str, optional): Weight initialization method for output
            layers. Defaults to "GlorotOrthogonal".
    """

    def __init__(
        self,
        out_channels: int,
        hidden_channels: int = 128,
        num_blocks: int = 4,
        int_emb_size: int = 64,
        basis_emb_size_dist: int = 8,
        basis_emb_size_angle: int = 8,
        basis_emb_size_torsion: int = 8,
        out_emb_channels: int = 256,
        num_spherical: int = 7,
        num_embeddings: int = 95,
        num_radial: int = 6,
        otf_graph: bool = False,
        cutoff: float = 5.0,
        max_num_neighbors: int = 50,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        readout: str = "mean",
        property_names: Optional[str] = "formation_energy_per_atom",
        data_mean: float = 0.0,
        data_std: float = 1.0,
        loss_type: str = "l1_loss",
        act: str = "swish",
        output_init: str = "GlorotOrthogonal",
    ):
        super().__init__()
        # store hyperparams
        self.out_channels = out_channels
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.otf_graph = otf_graph
        self.readout = readout

        if isinstance(property_names, list):
            self.property_names = property_names[0]
        else:
            assert isinstance(property_names, str)
            self.property_names = property_names

        self.register_buffer(
            tensor=paddle.to_tensor(data_mean), name="data_mean"
        )
        self.register_buffer(
            tensor=paddle.to_tensor(data_std), name="data_std"
        )

        # act func
        if act == "swish":
            act_fn = swish
        else:
            raise ValueError(f"Invalid activation function: {act}")

        # embedding layers
        self.emb_layer = emb(num_spherical, num_radial, self.cutoff, envelope_exponent)

        self.init_e = init(
            num_radial, hidden_channels, act_fn, use_node_features=True
        )
        self.init_v = update_v(
            hidden_channels, out_emb_channels, out_channels,
            num_output_layers, act_fn, output_init
        )
        self.init_u = update_u()

        self.update_vs = paddle.nn.LayerList(
            sublayers=[
                update_v(
                    hidden_channels, out_emb_channels, out_channels,
                    num_output_layers, act_fn, output_init
                )
                for _ in range(num_blocks)
            ]
        )
        self.update_es = paddle.nn.LayerList(
            sublayers=[
                update_e(
                    hidden_channels, int_emb_size,
                    basis_emb_size_dist, basis_emb_size_angle,
                    basis_emb_size_torsion,
                    num_spherical, num_radial,
                    num_before_skip, num_after_skip, act_fn
                )
                for _ in range(num_blocks)
            ]
        )
        self.update_us = paddle.nn.LayerList(
            sublayers=[update_u() for _ in range(num_blocks)]
        )

        if loss_type == "mse_loss":
            self.loss_fn = paddle.nn.functional.mse_loss
        elif loss_type == "l1_loss":
            self.loss_fn = paddle.nn.functional.l1_loss
        else:
            raise ValueError(f"Unknown loss type {loss_type}.")

        self.reset_parameters()

    def reset_parameters(self):
        self.init_v.reset_parameters()
        for update_v_layer in self.update_vs:
            update_v_layer.reset_parameters()

    def triplets(self, edge_index, num_nodes):
        row, col = edge_index
        value = paddle.arange(1, row.shape[0] + 1, dtype="int64")
        n = col.shape[0]
        rows = paddle.arange(end=n).unsqueeze(1)
        cols = paddle.arange(end=n).unsqueeze(0)
        mask = (col.unsqueeze(1) == col.unsqueeze(0)) & (cols <= rows)
        col_ = mask.astype("int64").sum(axis=1) - 1
        mat = paddle.scatter_nd(
            paddle.stack(x=[col, col_], axis=1),
            value,
            shape=[num_nodes, col_.max().item() + 1],
        )
        idx_kj = mat[row][mat[row] > 0] - 1
        tmp = paddle.nonzero(mat[row], as_tuple=False)
        idx_ji = tmp[:, 0]
        idx_k = paddle.index_select(row, idx_kj, axis=0)
        idx_j = paddle.index_select(row, idx_ji, axis=0)
        idx_i = paddle.index_select(col, idx_ji, axis=0)
        mask2 = idx_i != idx_k
        return (
            col,
            row,
            idx_i[mask2],
            idx_j[mask2],
            idx_k[mask2],
            idx_kj[mask2],
            idx_ji[mask2],
        )

    def normalize(self, tensor):
        return (tensor - self.data_mean) / self.data_std

    def unnormalize(self, tensor):
        return tensor * self.data_std + self.data_mean

    def _forward(self, data):
        data["graph"] = data["graph"].tensor()

        graph = data["graph"]
        batch = graph.graph_node_id
        lattices = graph.node_feat["lattice"]
        pos = graph.node_feat["cart_coords"]
        frac = graph.node_feat["frac_coords"]
        edge_index = graph.edges
        to_jimages = graph.edge_feat["pbc_offset"]
        num_atoms = graph.node_feat["num_atoms"]
        num_bonds = graph.edge_feat["num_edges"]
        atom_types = graph.node_feat["atom_types"]

        out = get_pbc_distances(
            frac,
            edge_index.T,
            lattices,
            to_jimages,
            num_atoms,
            num_bonds,
            return_offsets=True,
        )
        edge_index = out["edge_index"]
        dist = out["distances"]
        offsets = out["offsets"]

        j, i, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
            edge_index, num_nodes=atom_types.shape[0]
        )

        # compute angles
        pos_i = pos[idx_i]
        pos_j = pos[idx_j]
        pos_ji = pos_j - pos_i + offsets[idx_ji]
        pos_kj = pos[idx_k] - pos_j + offsets[idx_kj]
        a = (pos_ji * pos_kj).sum(axis=-1)
        b = paddle.cross(pos_ji, pos_kj).norm(axis=-1)
        angle = paddle.atan2(b, a)

        # compute torsion angles
        dist_ji = pos_ji.norm(axis=-1, keepdim=True) + 1e-7
        pos_ji_normalized = pos_ji / dist_ji
        plane1 = paddle.cross(pos_ji, pos_kj)
        plane1_norm = plane1.norm(axis=-1, keepdim=True) + 1e-7
        plane1 = plane1 / plane1_norm
        plane2 = paddle.cross(pos_ji_normalized, plane1)
        cos_torsion = (pos_kj * plane1).sum(axis=-1)
        sin_torsion = (pos_kj * plane2).sum(axis=-1)
        torsion = paddle.atan2(sin_torsion, cos_torsion)
        torsion = torsion + paddle.to_tensor(
            3.141592653589793, dtype=torsion.dtype
        )

        # basis embeddings
        emb_out = self.emb_layer(dist, angle, torsion, idx_kj)

        # initial edge, node, graph features
        e = self.init_e(atom_types, emb_out, i, j)
        v = self.init_v(e, i)
        u = self.init_u(
            paddle.zeros_like(x=scatter(v, batch, dim=0)), v, batch
        )

        for update_e_layer, update_v_layer, update_u_layer in zip(
            self.update_es, self.update_vs, self.update_us
        ):
            e = update_e_layer(e, emb_out, idx_kj, idx_ji)
            v = update_v_layer(e, i)
            u = update_u_layer(u, v, batch)

        return u

    def forward(self, data, return_loss=True, return_prediction=True):
        assert (
            return_loss or return_prediction
        ), "At least one of return_loss or return_prediction must be True."
        pred = self._forward(data)

        loss_dict = {}
        if return_loss:
            label = data[self.property_names]
            label = self.normalize(label)
            loss = self.loss_fn(
                input=pred,
                label=label,
            )
            loss_dict["loss"] = loss

        prediction = {}
        if return_prediction:
            pred = self.unnormalize(pred)
            prediction[self.property_names] = pred
        return {"loss_dict": loss_dict, "pred_dict": prediction}

    @paddle.no_grad()
    def predict(self, graphs):
        if isinstance(graphs, list):
            results = []
            for graph in graphs:
                result = self._forward(
                    {
                        "graph": graph,
                    }
                )
                result = self.unnormalize(result).numpy()[0, 0]
                result = {self.property_names: result}
                results.append(result)
            return results

        else:
            data = {
                "graph": graphs,
            }
            result = self._forward(data)
            result = self.unnormalize(result).numpy()[0, 0]
            result = {self.property_names: result}
            return result
