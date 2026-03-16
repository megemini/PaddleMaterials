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

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None

import paddle
import numpy as np

from ppmat.models.gdinn.graph_utils import MolecularGraph


class CanonicalAtomFeaturizer:
    """Atom feature encoder that generates 74-dimensional atom features.

    This class provides comprehensive atom features including atom type, degree,
    formal charge, valence, hybridization, and other chemical properties.

    Features (74 dimensions total):
        - Atom type (one-hot, 44 types)
        - Degree (one-hot, 11 types)
        - Formal charge (1)
        - Radical electrons (1)
        - Number of hydrogen atoms (one-hot, 5 types)
        - Hybridization (one-hot, 5 types)
        - Aromatic (1)
        - Mass (1)
        - Valence (one-hot, 5 types)
    """
    """
    
    def __init__(self):
        """Initialize atom featurizer with allowable feature values."""
        # 44 atom types: H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl,
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
            Dictionary with "h" key containing atom features of shape [num_atoms, 74]
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for molecular graph construction. "
                            "Please install it with: pip install rdkit")
        
        num_atoms = mol.GetNumAtoms()
        features = np.zeros((num_atoms, 74), dtype=np.float32)
        
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
            Feature vector of shape [74]
        """
        feature = np.zeros(74, dtype=np.float32)
        idx = 0
        
        # 1. Atom type (one-hot, 44)
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
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for molecular graph construction. "
                            "Please install it with: pip install rdkit")
        
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
    add_self_loop: bool = True,
    node_featurizer: Optional[Callable] = None,
    edge_featurizer: Optional[Callable] = None,
    canonical_atom_order: bool = False
) -> MolecularGraph:
    """Convert RDKit molecule to MolecularGraph.
    
    This function replaces DGL's mol_to_bigraph for PaddlePaddle.
    
    Args:
        mol: RDKit molecule object
        add_self_loop: Whether to add self-loops
        node_featurizer: Atom featurizer (default: CanonicalAtomFeaturizer)
        edge_featurizer: Bond featurizer (default: CanonicalBondFeaturizer)
        canonical_atom_order: Whether to use canonical atom ordering
        
    Returns:
        MolecularGraph object
        
    Example:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles('CCO')
        >>> graph = mol_to_bigraph(mol)
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for molecular graph construction. "
                        "Please install it with: pip install rdkit")
    
    if mol is None:
        raise ValueError("Input molecule is None")
    
    # Sanitize molecule
    Chem.SanitizeMol(mol)
    
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
    
    # Create MolecularGraph
    num_nodes = mol.GetNumAtoms()
    graph = MolecularGraph(
        num_nodes=num_nodes,
        edges=(src, dst),
        node_feat=node_feat,
        edge_feat=edge_feat
    )
    
    return graph


def smiles_to_bigraph(
    smiles: str,
    add_self_loop: bool = True,
    node_featurizer: Optional[Callable] = None,
    edge_featurizer: Optional[Callable] = None
) -> MolecularGraph:
    """Convert SMILES string to MolecularGraph.
    
    Args:
        smiles: SMILES string
        add_self_loop: Whether to add self-loops
        node_featurizer: Atom featurizer
        edge_featurizer: Bond featurizer
        
    Returns:
        MolecularGraph object
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for molecular graph construction. "
                        "Please install it with: pip install rdkit")
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    return mol_to_bigraph(mol, add_self_loop, node_featurizer, edge_featurizer)


def compute_hydrogen_bond_features(
    mol1: 'Chem.Mol',
    mol2: Optional['Chem.Mol'] = None
) -> Dict[str, float]:
    """Compute hydrogen bond features for molecules.
    
    Args:
        mol1: First RDKit molecule (solvent 1)
        mol2: Second RDKit molecule (solvent 2), optional
        
    Returns:
        Dictionary with hydrogen bond features:
            - 'intra_hb1': Intra-molecular hydrogen bonds in mol1
            - 'intra_hb2': Intra-molecular hydrogen bonds in mol2 (if provided)
            - 'inter_hb': Inter-molecular hydrogen bonds (if mol2 provided)
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for hydrogen bond feature computation. "
                        "Please install it with: pip install rdkit")
    
    features = {}
    
    # Compute intra-molecular hydrogen bonds for mol1
    features['intra_hb1'] = float(_count_hydrogen_bonds(mol1))
    
    if mol2 is not None:
        # Compute intra-molecular hydrogen bonds for mol2
        features['intra_hb2'] = float(_count_hydrogen_bonds(mol2))
        
        # Compute inter-molecular hydrogen bonds
        features['inter_hb'] = float(_count_inter_hydrogen_bonds(mol1, mol2))
    else:
        features['intra_hb2'] = 0.0
        features['inter_hb'] = 0.0
    
    return features


def _count_hydrogen_bonds(mol: 'Chem.Mol') -> int:
    """Count intra-molecular hydrogen bonds in a molecule.
    
    Args:
        mol: RDKit molecule
        
    Returns:
        Number of hydrogen bonds
    """
    if mol is None:
        return 0
    
    count = 0
    
    # Find hydrogen bond donors (N-H, O-H)
    donors = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in [7, 8]:  # N or O
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 1:  # H
                    donors.append(atom.GetIdx())
                    break
    
    # Find hydrogen bond acceptors (N, O, F with lone pairs)
    acceptors = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in [7, 8, 9]:  # N, O, F
            acceptors.append(atom.GetIdx())
    
    # Count donor-acceptor pairs (simplified)
    for donor_idx in donors:
        for acceptor_idx in acceptors:
            if donor_idx != acceptor_idx:
                # Check distance (require 3D coordinates)
                try:
                    conf = mol.GetConformer()
                    donor_pos = conf.GetAtomPosition(donor_idx)
                    acceptor_pos = conf.GetAtomPosition(acceptor_idx)
                    dist = np.sqrt(
                        (donor_pos.x - acceptor_pos.x)**2 +
                        (donor_pos.y - acceptor_pos.y)**2 +
                        (donor_pos.z - acceptor_pos.z)**2
                    )
                    # Hydrogen bond distance threshold (Å)
                    if dist < 3.5:
                        count += 1
                except:
                    # No 3D coordinates available, skip
                    pass
    
    return count


def _count_inter_hydrogen_bonds(mol1: 'Chem.Mol', mol2: 'Chem.Mol') -> int:
    """Count inter-molecular hydrogen bonds between two molecules.
    
    Args:
        mol1: First RDKit molecule
        mol2: Second RDKit molecule
        
    Returns:
        Number of inter-molecular hydrogen bonds
    """
    if mol1 is None or mol2 is None:
        return 0
    
    # For now, return 0 (requires more sophisticated analysis)
    # This is a placeholder that can be improved with proper 3D analysis
    return 0
