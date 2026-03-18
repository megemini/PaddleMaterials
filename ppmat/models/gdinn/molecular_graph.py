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
Molecular graph construction utilities.

This module provides functions to convert RDKit molecules to graph representations
compatible with PGL, replacing DGL's mol_to_bigraph functionality.
"""

from typing import Dict, List, Optional, Union, Callable
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import AllChem

import paddle
import numpy as np

from ppmat.models.gdinn.graph_utils import MolecularGraph


class CanonicalAtomFeaturizer:
    """Atom feature encoder that generates 75-dimensional atom features.

    This class provides comprehensive atom features including atom type, degree,
    formal charge, valence, hybridization, and other chemical properties.

    Features (75 dimensions total):
        - Atom type (one-hot, 45 types)
        - Degree (one-hot, 11 types)
        - Formal charge (1)
        - Radical electrons (1)
        - Number of hydrogen atoms (one-hot, 5 types)
        - Hybridization (one-hot, 5 types)
        - Aromatic (1)
        - Mass (1)
        - Valence (one-hot, 5 types)
    """
    
    def __init__(self):
        """Initialize atom featurizer with allowable feature values."""
        # 45 atom types: H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl,
        # Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Ga, Ge, As, Se, Br,
        # Kr, Rb, Sr, Y, Zr, Nb, Mo, Tc, Ru, Rh
        self.allowable_atom_types = [
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
            'Nb', 'Mo', 'Tc', 'Ru', 'Rh'
        ]
        
        # 11 possible degrees (0-10)
        self.allowable_degree = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        # 5 possible numbers of hydrogens (0-4)
        self.allowable_num_hs = [0, 1, 2, 3, 4]
        
        # 5 possible valences (0-4)
        self.allowable_valence = [0, 1, 2, 3, 4]
        
        # 5 hybridization types
        self.allowable_hybridization = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2
        ]
    
    def __call__(self, mol: 'Chem.Mol') -> Dict[str, paddle.Tensor]:
        """Generate atom features for a molecule.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary with "h" key containing atom features of shape [num_atoms, 75]
        """
        num_atoms = mol.GetNumAtoms()
        features = np.zeros((num_atoms, 75), dtype=np.float32)
        
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            feature = self._featurize_atom(atom)
            features[i] = feature
        
        return {"h": paddle.to_tensor(features)}
    
    def _featurize_atom(self, atom: 'Chem.Atom') -> np.ndarray:
        """Generate features for a single atom.
        
        Args:
            atom: RDKit atom object
            
        Returns:
            Feature vector of shape [75]
        """
        feature = np.zeros(75, dtype=np.float32)
        idx = 0
        
        # 1. Atom type (one-hot, 45)
        atom_type = atom.GetSymbol()
        for i, t in enumerate(self.allowable_atom_types):
            if atom_type == t:
                feature[idx + i] = 1.0
        idx += len(self.allowable_atom_types)
        
        # 2. Degree (one-hot, 11)
        degree = atom.GetDegree()
        for i, d in enumerate(self.allowable_degree):
            if degree == d:
                feature[idx + i] = 1.0
        idx += len(self.allowable_degree)
        
        # 3. Formal charge (1)
        feature[idx] = float(atom.GetFormalCharge())
        idx += 1
        
        # 4. Radical electrons (1)
        feature[idx] = float(atom.GetNumRadicalElectrons())
        idx += 1
        
        # 5. Number of hydrogens (one-hot, 5)
        num_hs = atom.GetTotalNumHs()
        for i, h in enumerate(self.allowable_num_hs):
            if num_hs == h:
                feature[idx + i] = 1.0
        idx += len(self.allowable_num_hs)
        
        # 6. Hybridization (one-hot, 5)
        hybridization = atom.GetHybridization()
        for i, h in enumerate(self.allowable_hybridization):
            if hybridization == h:
                feature[idx + i] = 1.0
        idx += len(self.allowable_hybridization)
        
        # 7. Aromatic (1)
        feature[idx] = float(atom.GetIsAromatic())
        idx += 1
        
        # 8. Mass (1)
        feature[idx] = float(atom.GetMass())
        idx += 1
        
        # 9. Valence (one-hot, 5)
        valence = atom.GetTotalValence()
        for i, v in enumerate(self.allowable_valence):
            if valence == v:
                feature[idx + i] = 1.0
        idx += len(self.allowable_valence)
        
        return feature


class CanonicalBondFeaturizer:
    """Bond feature encoder.
    
    This class provides comprehensive bond features including bond type,
    conjugation, ring membership, and stereochemistry.
    
    Features (12 dimensions total):
        - Bond type (one-hot, 4 types: single, double, triple, aromatic)
        - Conjugated (1)
        - In ring (1)
        - Stereo (one-hot, 6 types)
    """
    
    def __init__(self, self_loop: bool = False):
        """Initialize bond featurizer.
        
        Args:
            self_loop: Whether to include self-loops
        """
        self.self_loop = self_loop
    
    def __call__(self, mol: 'Chem.Mol') -> Dict[str, paddle.Tensor]:
        """Generate bond features for a molecule.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary with "e" key containing bond features of shape [num_edges, 12]
        """
        num_atoms = mol.GetNumAtoms()
        
        # Collect bond information
        bonds = []
        for bond in mol.GetBonds():
            src = bond.GetBeginAtomIdx()
            dst = bond.GetEndAtomIdx()
            features = self._featurize_bond(bond)
            bonds.append((src, dst, features))
            bonds.append((dst, src, features))
        
        # Add self-loops if needed
        if self.self_loop:
            for i in range(num_atoms):
                self_loop_feat = np.zeros(12, dtype=np.float32)
                self_loop_feat[0] = 1.0  # Single bond
                bonds.append((i, i, self_loop_feat))
        
        # Convert to arrays
        if len(bonds) == 0:
            # Handle edge case of single atom
            src = np.array([0], dtype=np.int64)
            dst = np.array([0], dtype=np.int64)
            edge_feat = np.zeros((1, 12), dtype=np.float32)
        else:
            src = np.array([b[0] for b in bonds], dtype=np.int64)
            dst = np.array([b[1] for b in bonds], dtype=np.int64)
            edge_feat = np.stack([b[2] for b in bonds], axis=0)
        
        return {
            "src": paddle.to_tensor(src),
            "dst": paddle.to_tensor(dst),
            "e": paddle.to_tensor(edge_feat)
        }
    
    def _featurize_bond(self, bond: 'Chem.Bond') -> np.ndarray:
        """Generate features for a single bond.
        
        Args:
            bond: RDKit bond object
            
        Returns:
            Feature vector of shape [12]
        """
        feature = np.zeros(12, dtype=np.float32)
        idx = 0
        
        # 1. Bond type (one-hot, 4)
        bond_type = bond.GetBondType()
        if bond_type == Chem.rdchem.BondType.SINGLE:
            feature[idx] = 1.0
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            feature[idx + 1] = 1.0
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            feature[idx + 2] = 1.0
        elif bond_type == Chem.rdchem.BondType.AROMATIC:
            feature[idx + 3] = 1.0
        idx += 4
        
        # 2. Conjugated (1)
        feature[idx] = float(bond.GetIsConjugated())
        idx += 1
        
        # 3. In ring (1)
        feature[idx] = float(bond.IsInRing())
        idx += 1
        
        # 4. Stereo (one-hot, 6)
        stereo = bond.GetStereo()
        if stereo == Chem.rdchem.BondStereo.STEREONONE:
            feature[idx] = 1.0
        elif stereo == Chem.rdchem.BondStereo.STEREOZ:
            feature[idx + 1] = 1.0
        elif stereo == Chem.rdchem.BondStereo.STEREOE:
            feature[idx + 2] = 1.0
        elif stereo == Chem.rdchem.BondStereo.STEREOCIS:
            feature[idx + 3] = 1.0
        elif stereo == Chem.rdchem.BondStereo.STEREOTRANS:
            feature[idx + 4] = 1.0
        elif stereo == Chem.rdchem.BondStereo.STEREOANY:
            feature[idx + 5] = 1.0
        idx += 6
        
        return feature


def mol_to_bigraph(
    mol: 'Chem.Mol',
    add_self_loop: bool = False,
    node_featurizer: Optional[Callable] = None,
    edge_featurizer: Optional[Callable] = None,
    canonical_atom_order: bool = True,
    explicit_hydrogens: bool = False,
    num_virtual_nodes: int = 0
) -> MolecularGraph:
    """Convert RDKit molecule to MolecularGraph.

    This function replaces DGL's mol_to_bigraph for PaddlePaddle.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to False.
    node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to None.
    edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph. Default to None.
    canonical_atom_order : bool
        Whether to use a canonical order of atoms returned by RDKit. Setting it
        to true might change the order of atoms in the graph constructed. Default
        to True.
    explicit_hydrogens : bool
        Whether to explicitly represent hydrogens as nodes in the graph. If True,
        it will call rdkit.Chem.AddHs(mol). Default to False.
    num_virtual_nodes : int
        The number of virtual nodes to add. The virtual nodes will be connected to
        all real nodes with virtual edges. If the returned graph has any node/edge
        feature, an additional column of binary values will be used for each feature
        to indicate the identity of virtual node/edges. The features of the virtual
        nodes/edges will be zero vectors except for the additional column. Default to 0.

    Returns
    -------
    DGLGraph or None
        Bi-directed DGLGraph for the molecule if :attr:`mol` is valid and None otherwise.
    """
    if mol is None:
        raise ValueError("Input molecule is None")

    # Whether to have hydrogen atoms as explicit nodes
    if explicit_hydrogens:
        mol = Chem.AddHs(mol)

    # Apply canonical ordering if requested
    if canonical_atom_order:
        mol = Chem.RenumberAtoms(mol, list(range(mol.GetNumAtoms())))

    # Default featurizers
    if node_featurizer is None:
        node_featurizer = CanonicalAtomFeaturizer()
    if edge_featurizer is None:
        edge_featurizer = CanonicalBondFeaturizer(self_loop=add_self_loop)

    # Generate node features
    node_feat = node_featurizer(mol)

    # Generate edge features
    edge_data = edge_featurizer(mol)
    src = edge_data["src"]
    dst = edge_data["dst"]
    edge_feat = {"e": edge_data["e"]}

    # Handle virtual nodes
    if num_virtual_nodes > 0:
        num_real_nodes = mol.GetNumAtoms()
        real_nodes = list(range(num_real_nodes))

        # Add virtual node indices to node features
        for feat_name in node_feat.keys():
            real_feat = node_feat[feat_name]
            feat_dim = real_feat.shape[1]
            virtual_feat = paddle.zeros((num_virtual_nodes, feat_dim), dtype=real_feat.dtype)
            # Add indicator column for virtual nodes
            real_indicator = paddle.zeros((num_real_nodes, 1), dtype=real_feat.dtype)
            virtual_indicator = paddle.ones((num_virtual_nodes, 1), dtype=real_feat.dtype)
            real_feat = paddle.concat([real_feat, real_indicator], axis=1)
            virtual_feat = paddle.concat([virtual_feat, virtual_indicator], axis=1)
            node_feat[feat_name] = paddle.concat([real_feat, virtual_feat], axis=0)

        # Add virtual edges
        virtual_src = []
        virtual_dst = []
        for count in range(num_virtual_nodes):
            virtual_node = num_real_nodes + count
            virtual_node_copy = [virtual_node] * num_real_nodes
            virtual_src.extend(real_nodes)
            virtual_src.extend(virtual_node_copy)
            virtual_dst.extend(virtual_node_copy)
            virtual_dst.extend(real_nodes)

        # Concatenate edges
        src = paddle.concat([src, paddle.to_tensor(np.array(virtual_src, dtype=np.int64))])
        dst = paddle.concat([dst, paddle.to_tensor(np.array(virtual_dst, dtype=np.int64))])

        # Add indicator column to edge features
        for feat_name in edge_feat.keys():
            real_feat = edge_feat[feat_name]
            num_real_edges = real_feat.shape[0]
            feat_dim = real_feat.shape[1]
            num_virtual_edges = len(virtual_src)
            real_indicator = paddle.zeros((num_real_edges, 1), dtype=real_feat.dtype)
            virtual_indicator = paddle.ones((num_virtual_edges, 1), dtype=real_feat.dtype)
            real_feat = paddle.concat([real_feat, real_indicator], axis=1)
            virtual_feat = paddle.zeros((num_virtual_edges, feat_dim + 1), dtype=real_feat.dtype)
            virtual_feat[:, :-1] = 0
            virtual_feat[:, -1] = 1
            edge_feat[feat_name] = paddle.concat([real_feat, virtual_feat], axis=0)

    # Create MolecularGraph
    num_nodes = mol.GetNumAtoms() + num_virtual_nodes
    graph = MolecularGraph(
        num_nodes=num_nodes,
        edges=(src, dst),
        node_feat=node_feat,
        edge_feat=edge_feat
    )

    return graph


def smiles_to_bigraph(
    smiles: str,
    add_self_loop: bool = False,
    node_featurizer: Optional[Callable] = None,
    edge_featurizer: Optional[Callable] = None,
    canonical_atom_order: bool = True,
    explicit_hydrogens: bool = False,
    num_virtual_nodes: int = 0
) -> MolecularGraph:
    """Convert a SMILES into a bi-directed DGLGraph and featurize for it.

    Parameters
    ----------
    smiles : str
        String of SMILES
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to False.
    node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to None.
    edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph. Default to None.
    canonical_atom_order : bool
        Whether to use a canonical order of atoms returned by RDKit. Setting it
        to true might change the order of atoms in the graph constructed. Default
        to True.
    explicit_hydrogens : bool
        Whether to explicitly represent hydrogens as nodes in the graph. If True,
        it will call rdkit.Chem.AddHs(mol). Default to False.
    num_virtual_nodes : int
        The number of virtual nodes to add. The virtual nodes will be connected to
        all real nodes with virtual edges. If the returned graph has any node/edge
        feature, an additional column of binary values will be used for each feature
        to indicate the identity of virtual node/edges. The features of the virtual
        nodes/edges will be zero vectors except for the additional column. Default to 0.

    Returns
    -------
    DGLGraph or None
        Bi-directed DGLGraph for the molecule if :attr:`smiles` is valid and None otherwise.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    return mol_to_bigraph(mol, add_self_loop, node_featurizer, edge_featurizer,
                          canonical_atom_order, explicit_hydrogens, num_virtual_nodes)


def compute_hydrogen_bond_features(
    mol1: 'Chem.Mol',
    mol2: Optional['Chem.Mol'] = None
) -> Dict[str, float]:
    """Compute hydrogen bond features for molecules using RDKit descriptors.

    This matches the GDI-NN implementation which uses RDKit's hydrogen bond
    acceptor and donor descriptors to quantify hydrogen bonding capacity.

    Args:
        mol1: First RDKit molecule (solvent 1)
        mol2: Second RDKit molecule (solvent 2), optional

    Returns:
        Dictionary with hydrogen bond features:
            - 'intra_hb1': Intra-molecular hydrogen bonding capacity for mol1
                          Calculated as min(HBA, HBD) for solvent 1
            - 'intra_hb2': Intra-molecular hydrogen bonding capacity for mol2 (if provided)
                          Calculated as min(HBA, HBD) for solvent 2
            - 'inter_hb': Inter-molecular hydrogen bonding capacity (if mol2 provided)
                          Calculated as: min(HBA1, HBD2) + min(HBD1, HBA2)

    Reference:
        GDI-NN: https://git.rwth-aachen.de/avt-svt/public/GDI-NN
    """
    from rdkit.Chem import rdMolDescriptors

    # Compute hydrogen bond acceptors and donors for mol1
    hba1 = rdMolDescriptors.CalcNumHBA(mol1)
    hbd1 = rdMolDescriptors.CalcNumHBD(mol1)

    # Intra-molecular hydrogen bonding capacity for mol1
    # Represents self-association capability within the solvent
    intra_hb1 = float(min(hba1, hbd1))

    if mol2 is not None:
        # Compute hydrogen bond acceptors and donors for mol2
        hba2 = rdMolDescriptors.CalcNumHBA(mol2)
        hbd2 = rdMolDescriptors.CalcNumHBD(mol2)

        # Intra-molecular hydrogen bonding capacity for mol2
        intra_hb2 = float(min(hba2, hbd2))

        # Inter-molecular hydrogen bonding capacity
        # Represents cross-interaction potential between two solvents
        # Solvent1 can donate to Solvent2 (HBD1 with HBA2) + Solvent2 can donate to Solvent1 (HBD2 with HBA1)
        inter_hb = float(min(hba1, hbd2) + min(hbd1, hba2))

        return {
            'intra_hb1': intra_hb1,
            'intra_hb2': intra_hb2,
            'inter_hb': inter_hb
        }
    else:
        return {
            'intra_hb1': intra_hb1,
            'intra_hb2': 0.0,
            'inter_hb': 0.0
        }
