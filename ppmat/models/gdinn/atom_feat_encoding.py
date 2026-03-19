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
Atom and bond feature encoding utilities.

This module provides featurizers for encoding atoms and bonds in molecular graphs.
"""

from typing import Dict

from rdkit import Chem

import paddle
import numpy as np


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
