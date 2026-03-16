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
Binary activity coefficient dataset for GDI-NN.

This module provides a dataset class for loading binary solvent mixture data
with activity coefficients (gamma1, gamma2) and converting molecules to graphs.
"""

import os
import csv
from typing import Optional, Dict, List, Union
from pathlib import Path

import paddle
import numpy as np
from paddle.io import Dataset

from ppmat.models.gdinn.graph_utils import MolecularGraph
from ppmat.datasets.build_molecule import BuildMolecule
from ppmat.utils.molecular_graph import (
    mol_to_bigraph,
    smiles_to_bigraph,
    compute_hydrogen_bond_features
)


class BinaryActivityDataset(Dataset):
    """Binary activity coefficient dataset.

    This dataset loads binary solvent mixture data from a CSV file and converts
    molecules to graph representations. Each sample contains two molecular graphs,
    composition (x1), activity coefficients (gamma1, gamma2), temperature (T), and hydrogen bond features.

    Supported CSV format:
        SMILES_x, SMILES_y, temperature (K), x(1), x(2), ln_gamma_1, ln_gamma_2
        "CCO", "CC(C)O", 298.15, 0.5, 0.5, 0.182, -0.095
        ...

    Note: The dataset expects ln_gamma values and will convert them to gamma using exp().

    Args:
        data_path: Path to CSV file containing binary mixture data
        solvent_list_path: Path to file containing list of solvent SMILES (optional)
        graph_converter: Function to convert molecules to graphs (default: mol_to_bigraph)
        add_self_loop: Whether to add self-loops to graphs (default: True)
        preload_graphs: Whether to preload all graphs into memory (default: False)
        compute_hb: Whether to compute hydrogen bond features (default: False)
    """
    
    def __init__(
        self,
        data_path: str,
        solvent_list_path: Optional[str] = None,
        graph_converter: Optional[callable] = None,
        add_self_loop: bool = True,
        preload_graphs: bool = False,
        compute_hb: bool = False
    ):
        super().__init__()
        self.data_path = data_path
        self.solvent_list_path = solvent_list_path
        self.add_self_loop = add_self_loop
        self.preload_graphs = preload_graphs
        self.compute_hb = compute_hb
        
        # Set default graph converter
        if graph_converter is None:
            self.graph_converter = lambda mol: mol_to_bigraph(
                mol, add_self_loop=add_self_loop
            )
        else:
            self.graph_converter = graph_converter
        
        # Initialize BuildMolecule factory for processing SMILES
        self.build_molecule = BuildMolecule(
            format="smiles",
            sanitize=True,
            add_hs=False,
            remove_hs=False,
            kekulize=False
        )
        
        # Load data
        self.data = self._load_csv(data_path)

        # Convert ln_gamma to gamma
        self._convert_ln_gamma_to_gamma()

        # Load solvent list if provided
        self.solvent_cache = {}
        if solvent_list_path and os.path.exists(solvent_list_path):
            self._load_solvent_list(solvent_list_path)
        
        # Preload graphs if requested
        self.graph_cache = {}
        if preload_graphs:
            self._preload_graphs()
        
        # Check required columns
        self._validate_data()
    
    def _load_csv(self, data_path: str) -> List[Dict]:
        """Load CSV data file.
        
        Args:
            data_path: Path to CSV file
            
        Returns:
            List of dictionaries, each representing a row
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        data = []
        with open(data_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        
        return data
    
    def _load_solvent_list(self, solvent_list_path: str):
        """Load solvent list for caching molecular graphs.
        
        Args:
            solvent_list_path: Path to solvent list file
                Format: SMILES (one per line)
        """
        try:
            with open(solvent_list_path, 'r') as f:
                for line in f:
                    smiles = line.strip()
                    if smiles and not smiles.startswith('#'):
                        self.solvent_cache[smiles] = None
        except Exception as e:
            print(f"Warning: Failed to load solvent list: {e}")
    
    def _preload_graphs(self):
        """Preload all molecular graphs into memory."""
        print("Preloading molecular graphs...")

        # Collect all unique SMILES
        all_smiles = set()
        for row in self.data:
            smiles_x = row.get('SMILES_x', '')
            smiles_y = row.get('SMILES_y', '')
            if smiles_x:
                all_smiles.add(smiles_x)
            if smiles_y:
                all_smiles.add(smiles_y)

        # Precompute graphs using BuildMolecule factory
        for smiles in all_smiles:
            if smiles not in self.graph_cache:
                try:
                    mol = self.build_molecule(smiles)
                    if mol is not None:
                        self.graph_cache[smiles] = self.graph_converter(mol)
                except Exception as e:
                    print(f"Warning: Failed to convert SMILES to graph: {smiles}, {e}")

        print(f"Preloaded {len(self.graph_cache)} molecular graphs")

    def _convert_ln_gamma_to_gamma(self):
        """Convert ln_gamma to gamma in-place."""
        if len(self.data) == 0:
            raise ValueError("Data file is empty")

        for row in self.data:
            ln_gamma1 = float(row['ln_gamma_1'])
            ln_gamma2 = float(row['ln_gamma_2'])

            # Handle infinite values
            if np.isinf(ln_gamma1) or np.isnan(ln_gamma1):
                row['ln_gamma_1'] = str(1e6)
            else:
                row['ln_gamma_1'] = str(float(np.exp(ln_gamma1)))

            if np.isinf(ln_gamma2) or np.isnan(ln_gamma2):
                row['ln_gamma_2'] = str(1e6)
            else:
                row['ln_gamma_2'] = str(float(np.exp(ln_gamma2)))

        print("ln_gamma to gamma conversion completed")
    
    def _validate_data(self):
        """Validate that required columns exist in the data."""
        if len(self.data) == 0:
            raise ValueError("Data file is empty")

        required_columns = ['SMILES_x', 'SMILES_y', 'x(1)', 'x(2)', 'ln_gamma_1', 'ln_gamma_2']
        available_columns = list(self.data[0].keys())

        # Check if temperature column exists
        self.has_temperature = 'temperature (K)' in available_columns

        missing_columns = [col for col in required_columns if col not in available_columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in data: {missing_columns}. "
                f"Available columns: {available_columns}"
            )

        # Check for hydrogen bond columns
        self.has_hb_columns = all(col in available_columns for col in ['intra_hb1', 'intra_hb2', 'inter_hb'])
    
    def _get_molecular_graph(self, smiles: str) -> MolecularGraph:
        """Get molecular graph for a SMILES string.
        
        Args:
            smiles: SMILES string
            
        Returns:
            MolecularGraph object
        """
        if self.preload_graphs and smiles in self.graph_cache:
            return self.graph_cache[smiles]
        
        try:
            # Use BuildMolecule factory to build molecule from SMILES
            mol = self.build_molecule(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            
            graph = self.graph_converter(mol)
            
            if self.preload_graphs:
                self.graph_cache[smiles] = graph
            
            return graph
        except Exception as e:
            raise ValueError(f"Failed to convert SMILES to graph: {smiles}, {e}")
    
    def _parse_value(self, value: str) -> float:
        """Parse string value to float, handling special cases.
        
        Args:
            value: String value
            
        Returns:
            Float value
        """
        try:
            return float(value)
        except (ValueError, TypeError):
            # Handle special cases like 'inf', 'nan'
            value_lower = str(value).lower()
            if value_lower == 'inf':
                return float('inf')
            elif value_lower == '-inf':
                return float('-inf')
            elif value_lower == 'nan':
                return float('nan')
            else:
                return 0.0
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - g1: Molecular graph for solvent 1
                - g2: Molecular graph for solvent 2
                - x1: Composition of solvent 1 (mole fraction)
                - x2: Composition of solvent 2 (mole fraction)
                - gamma1: Activity coefficient for solvent 1
                - gamma2: Activity coefficient for solvent 2
                - T: Temperature (if available)
                - intra_hb1: Intra-molecular hydrogen bonds in solvent 1 (if compute_hb)
                - intra_hb2: Intra-molecular hydrogen bonds in solvent 2 (if compute_hb)
                - inter_hb: Inter-molecular hydrogen bonds (if compute_hb)
        """
        row = self.data[idx]

        # Get SMILES strings
        smiles_x = row['SMILES_x']
        smiles_y = row['SMILES_y']

        # Parse numeric values
        x1 = self._parse_value(row['x(1)'])
        x2 = self._parse_value(row['x(2)'])
        gamma1 = self._parse_value(row['ln_gamma_1'])
        gamma2 = self._parse_value(row['ln_gamma_2'])

        # Handle infinite values (clip to reasonable range)
        gamma1 = self._clip_gamma(gamma1)
        gamma2 = self._clip_gamma(gamma2)

        # Convert SMILES to molecular graphs
        g1 = self._get_molecular_graph(smiles_x)
        g2 = self._get_molecular_graph(smiles_y)

        # Build sample dictionary
        sample = {
            'g1': g1,
            'g2': g2,
            'x1': np.array([[x1]], dtype=np.float32),
            'x2': np.array([[x2]], dtype=np.float32),
            'gamma1': np.array([[gamma1]], dtype=np.float32),
            'gamma2': np.array([[gamma2]], dtype=np.float32)
        }

        # Add temperature if available
        if self.has_temperature:
            T = self._parse_value(row['temperature (K)'])
            sample['T'] = np.array([[T]], dtype=np.float32)
        
        # Add hydrogen bond features if available
        if self.has_hb_columns:
            sample['intra_hb1'] = np.array([[self._parse_value(row['intra_hb1'])]], dtype=np.float32)
            sample['intra_hb2'] = np.array([[self._parse_value(row['intra_hb2'])]], dtype=np.float32)
            sample['inter_hb'] = np.array([[self._parse_value(row['inter_hb'])]], dtype=np.float32)
        elif self.compute_hb:
            # Compute hydrogen bond features on the fly using BuildMolecule factory
            mol1 = self.build_molecule(smiles_x)
            mol2 = self.build_molecule(smiles_y)
            hb_features = compute_hydrogen_bond_features(mol1, mol2)
            sample['intra_hb1'] = np.array([[hb_features['intra_hb1']]], dtype=np.float32)
            sample['intra_hb2'] = np.array([[hb_features['intra_hb2']]], dtype=np.float32)
            sample['inter_hb'] = np.array([[hb_features['inter_hb']]], dtype=np.float32)
        
        return sample
    
    def _clip_gamma(self, gamma: float, min_val: float = 1e-6, max_val: float = 1e6) -> float:
        """Clip gamma values to a reasonable range.
        
        Args:
            gamma: Gamma value
            min_val: Minimum value
            max_val: Maximum value
            
        Returns:
            Clipped gamma value
        """
        if np.isinf(gamma) or np.isnan(gamma):
            return max_val
        return max(min_val, min(max_val, gamma))
    
    def get_solvent_list(self) -> List[str]:
        """Get list of unique solvents in dataset.

        Returns:
            List of unique solvent SMILES strings
        """
        solvents = set()
        for row in self.data:
            solvents.add(row['SMILES_x'])
            solvents.add(row['SMILES_y'])
        return list(solvents)

    def get_composition_range(self) -> Dict[str, float]:
        """Get the range of compositions in dataset.

        Returns:
            Dictionary with 'min' and 'max' values
        """
        compositions = [self._parse_value(row['x(1)']) for row in self.data]
        return {
            'min': min(compositions),
            'max': max(compositions)
        }


class BinaryActivityDatasetFromSMILES(BinaryActivityDataset):
    """Binary activity coefficient dataset from SMILES lists.

    This is a convenience class that creates a dataset from separate lists of
    solvents 1 and 2, compositions, and activity coefficients.

    Args:
        solvents1: List of SMILES for solvent 1
        solvents2: List of SMILES for solvent 2
        compositions: List of compositions (x1)
        gamma1: List of activity coefficients for solvent 1
        gamma2: List of activity coefficients for solvent 2
        intra_hb1: List of intra-molecular hydrogen bonds for solvent 1 (optional)
        intra_hb2: List of intra-molecular hydrogen bonds for solvent 2 (optional)
        inter_hb: List of inter-molecular hydrogen bonds (optional)
        graph_converter: Function to convert molecules to graphs
        add_self_loop: Whether to add self-loops to graphs
        preload_graphs: Whether to preload all graphs
    """

    def __init__(
        self,
        solvents1: List[str],
        solvents2: List[str],
        compositions: List[float],
        gamma1: List[float],
        gamma2: List[float],
        intra_hb1: Optional[List[float]] = None,
        intra_hb2: Optional[List[float]] = None,
        inter_hb: Optional[List[float]] = None,
        graph_converter: Optional[callable] = None,
        add_self_loop: bool = True,
        preload_graphs: bool = False
    ):
        # Validate input lengths
        n_samples = len(solvents1)
        if not all(len(lst) == n_samples for lst in [solvents2, compositions, gamma1, gamma2]):
            raise ValueError("All input lists must have the same length")

        if intra_hb1 is not None:
            if not all(len(lst) == n_samples for lst in [intra_hb1, intra_hb2, inter_hb]):
                raise ValueError("Hydrogen bond lists must have the same length as other inputs")

        # Build data list with real data format
        data = []
        for i in range(n_samples):
            row = {
                'SMILES_x': solvents1[i],
                'SMILES_y': solvents2[i],
                'x(1)': str(compositions[i]),
                'x(2)': str(1.0 - compositions[i]),
                'ln_gamma_1': str(np.log(gamma1[i])),
                'ln_gamma_2': str(np.log(gamma2[i])),
                'temperature (K)': '298.15'
            }
            if intra_hb1 is not None:
                row['intra_hb1'] = str(intra_hb1[i])
                row['intra_hb2'] = str(intra_hb2[i])
                row['inter_hb'] = str(inter_hb[i])
            data.append(row)

        # Initialize parent class
        super().__init__(
            data_path="",  # Not used
            solvent_list_path=None,
            graph_converter=graph_converter,
            add_self_loop=add_self_loop,
            preload_graphs=preload_graphs,
            compute_hb=False
        )

        # Override data with constructed data
        self.data = data
        self.has_hb_columns = intra_hb1 is not None
