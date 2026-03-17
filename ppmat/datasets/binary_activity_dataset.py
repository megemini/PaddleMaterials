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
with activity coefficients (gamma1, gamma2) in GDI-NN format.

GDI-NN Format (input_file):
    - Required columns: solv1, solv2, solv1_x, solv2_x, solv1_gamma, solv2_gamma
    - Optional columns: solv1_smiles, solv2_smiles, solv1_name, solv2_name, intra_hb1, intra_hb2, inter_hb

Solvent List Format:
    - Required columns: solvent_name, solvent_id, smiles_can

Reference: GDI-NN (https://git.rwth-aachen.de/avt-svt/public/GDI-NN)
"""

import os
import csv
from typing import Optional, Dict, List
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
    """Binary activity coefficient dataset in GDI-NN format.

    This dataset loads binary solvent mixture data from a CSV file and converts
    molecules to graph representations. Each sample contains two molecular graphs,
    composition (x1), activity coefficients (gamma1, gamma2), and hydrogen bond features.

    GDI-NN CSV Format (input_file):
        solv1, solv2, solv1_x, solv2_x, solv1_gamma, solv2_gamma, solv1_smiles, solv2_smiles, solv1_name, solv2_name, ...
        "solvent_587", "solvent_604", 0.1, 0.9, 0.47175935, 0.00025148, "CN", "CC(=O)CC(C)C", "METHYL AMINE", "METHYL ISOBUTYL KETONE", ...

    Solvent List Format:
        solvent_name, solvent_id, smiles_can
        "1,1,1-TRICHLOROETHANE", solvent_1, CC(Cl)(Cl)Cl
        ...

    Note: The dataset expects ln_gamma values (natural log) and will convert them to gamma using exp().

    Args:
        data_path: Path to CSV file containing binary mixture data
        solvent_list_path: Path to file containing list of solvents
            Format: solvent_name, solvent_id, smiles_can
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
        """Initialize Binary Activity Dataset.

        Args:
            data_path: Path to CSV file containing binary mixture data (GDI-NN format)
            solvent_list_path: Path to file containing list of solvents
                Format: solvent_name, solvent_id, smiles_can
            graph_converter: Function to convert molecules to graphs (default: mol_to_bigraph)
            add_self_loop: Whether to add self-loops to graphs (default: True)
            preload_graphs: Whether to preload all graphs into memory (default: False)
            compute_hb: Whether to compute hydrogen bond features (default: False)
        """
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

        # Load solvent list for caching molecular graphs
        self.solvent_info = {}  # solvent_id -> {name, smiles}
        self.solvent_smiles = {}  # solvent_id -> smiles
        if solvent_list_path and os.path.exists(solvent_list_path):
            self._load_solvent_list(solvent_list_path)
        elif solvent_list_path:
            print(f"Warning: Solvent list not found: {solvent_list_path}")

        # Load data
        self.data = self._load_csv(data_path)

        # Validate data format
        self._validate_data()

        # Preload graphs if requested
        self.graph_cache = {}
        if preload_graphs:
            self._preload_graphs()

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

        GDI-NN format: solvent_name, solvent_id, smiles_can

        Args:
            solvent_list_path: Path to solvent list file
        """
        try:
            with open(solvent_list_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    solvent_id = row.get('solvent_id', '')
                    solvent_name = row.get('solvent_name', '')
                    smiles = row.get('smiles_can', '').strip()
                    if solvent_id and smiles:
                        self.solvent_info[solvent_id] = {
                            'name': solvent_name,
                            'smiles': smiles
                        }
                        self.solvent_smiles[solvent_id] = smiles
            print(f"Loaded {len(self.solvent_info)} solvents from solvent list")
        except Exception as e:
            print(f"Warning: Failed to load solvent list: {e}")

    def _preload_graphs(self):
        """Preload all molecular graphs into memory."""
        print("Preloading molecular graphs...")

        # Use solvent_smiles dictionary to preload
        if self.solvent_smiles:
            for solvent_id, smiles in self.solvent_smiles.items():
                if smiles and smiles not in self.graph_cache:
                    try:
                        mol = self.build_molecule(smiles)
                        if mol is not None:
                            self.graph_cache[smiles] = self.graph_converter(mol)
                            self.graph_cache[solvent_id] = self.graph_cache[smiles]
                    except Exception as e:
                        print(f"Warning: Failed to convert SMILES to graph: {smiles}, {e}")
        else:
            # Fallback: collect all unique SMILES from data
            all_smiles = set()
            for row in self.data:
                smiles1 = row.get('solv1_smiles', '')
                smiles2 = row.get('solv2_smiles', '')
                if smiles1:
                    all_smiles.add(smiles1)
                if smiles2:
                    all_smiles.add(smiles2)

            for smiles in all_smiles:
                if smiles not in self.graph_cache:
                    try:
                        mol = self.build_molecule(smiles)
                        if mol is not None:
                            self.graph_cache[smiles] = self.graph_converter(mol)
                    except Exception as e:
                        print(f"Warning: Failed to convert SMILES to graph: {smiles}, {e}")

        print(f"Preloaded {len(self.graph_cache)} molecular graphs")

    def _validate_data(self):
        """Validate that required columns exist in the data."""
        if len(self.data) == 0:
            raise ValueError("Data file is empty")

        available_columns = list(self.data[0].keys())

        # GDI-NN format validation
        required_columns = ['solv1', 'solv2', 'solv1_x', 'solv2_x', 'solv1_gamma', 'solv2_gamma']
        missing_columns = [col for col in required_columns if col not in available_columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in GDI-NN format data: {missing_columns}. "
                f"Available columns: {available_columns}"
            )

        # Check if SMILES columns exist in data
        self.has_smiles_in_data = 'solv1_smiles' in available_columns and 'solv2_smiles' in available_columns

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
            mol = self.build_molecule(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")

            graph = self.graph_converter(mol)

            if self.preload_graphs:
                self.graph_cache[smiles] = graph

            return graph
        except Exception as e:
            raise ValueError(f"Failed to convert SMILES to graph: {smiles}, {e}")

    def _get_smiles(self, solvent_id: str) -> str:
        """Get SMILES for a solvent ID.

        Args:
            solvent_id: Solvent ID (e.g., 'solvent_587')

        Returns:
            SMILES string
        """
        # First try to get from solvent_list
        if solvent_id in self.solvent_smiles:
            return self.solvent_smiles[solvent_id]

        # Fallback to data if available
        if self.has_smiles_in_data:
            for row in self.data:
                if row.get('solv1') == solvent_id:
                    return row.get('solv1_smiles', '')
                elif row.get('solv2') == solvent_id:
                    return row.get('solv2_smiles', '')

        raise ValueError(f"Cannot find SMILES for solvent ID: {solvent_id}")

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
                - x1: Composition of solvent 1 (mole fraction, solv1_x)
                - x2: Composition of solvent 2 (mole fraction, solv2_x)
                - gamma1: Activity coefficient for solvent 1 (converted from ln_gamma)
                - gamma2: Activity coefficient for solvent 2 (converted from ln_gamma)
                - intra_hb1: Intra-molecular hydrogen bonds in solvent 1 (if compute_hb or available)
                - intra_hb2: Intra-molecular hydrogen bonds in solvent 2 (if compute_hb or available)
                - inter_hb: Inter-molecular hydrogen bonds (if compute_hb or available)
                - solv1_id: Solvent 1 ID
                - solv2_id: Solvent 2 ID
        """
        row = self.data[idx]

        # Get solvent IDs
        solv1_id = row['solv1']
        solv2_id = row['solv2']

        # Get SMILES strings
        smiles1 = self._get_smiles(solv1_id)
        smiles2 = self._get_smiles(solv2_id)

        # Parse composition values
        x1 = self._parse_value(row['solv1_x'])
        x2 = self._parse_value(row['solv2_x'])

        # Parse ln_gamma values and convert to gamma using exp()
        ln_gamma1 = self._parse_value(row['solv1_gamma'])
        ln_gamma2 = self._parse_value(row['solv2_gamma'])

        # Convert ln_gamma to gamma
        gamma1 = self._convert_ln_gamma(ln_gamma1)
        gamma2 = self._convert_ln_gamma(ln_gamma2)

        # Convert SMILES to molecular graphs
        g1 = self._get_molecular_graph(smiles1)
        g2 = self._get_molecular_graph(smiles2)

        # Build sample dictionary
        sample = {
            'g1': g1,
            'g2': g2,
            'x1': np.array([[x1]], dtype=np.float32),
            'x2': np.array([[x2]], dtype=np.float32),
            'gamma1': np.array([[gamma1]], dtype=np.float32),
            'gamma2': np.array([[gamma2]], dtype=np.float32),
            'solv1_id': solv1_id,
            'solv2_id': solv2_id
        }

        # Add hydrogen bond features if available in data
        if self.has_hb_columns:
            sample['intra_hb1'] = np.array([[self._parse_value(row['intra_hb1'])]], dtype=np.float32)
            sample['intra_hb2'] = np.array([[self._parse_value(row['intra_hb2'])]], dtype=np.float32)
            sample['inter_hb'] = np.array([[self._parse_value(row['inter_hb'])]], dtype=np.float32)
        elif self.compute_hb:
            # Compute hydrogen bond features on the fly
            mol1 = self.build_molecule(smiles1)
            mol2 = self.build_molecule(smiles2)
            hb_features = compute_hydrogen_bond_features(mol1, mol2)
            sample['intra_hb1'] = np.array([[hb_features['intra_hb1']]], dtype=np.float32)
            sample['intra_hb2'] = np.array([[hb_features['intra_hb2']]], dtype=np.float32)
            sample['inter_hb'] = np.array([[hb_features['inter_hb']]], dtype=np.float32)

        return sample

    def _convert_ln_gamma(self, ln_gamma: float) -> float:
        """Convert ln_gamma to gamma, handling infinite and invalid values.

        Args:
            ln_gamma: Natural log of activity coefficient

        Returns:
            Activity coefficient (gamma)
        """
        if np.isinf(ln_gamma) or np.isnan(ln_gamma):
            # Handle extreme values - return a large but finite value
            return 1e6
        return np.exp(ln_gamma)

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
        """Get list of unique solvent IDs in dataset.

        Returns:
            List of unique solvent IDs
        """
        solvents = set()
        for row in self.data:
            solvents.add(row['solv1'])
            solvents.add(row['solv2'])
        return list(solvents)

    def get_composition_range(self) -> Dict[str, float]:
        """Get the range of compositions in dataset.

        Returns:
            Dictionary with 'min' and 'max' values
        """
        compositions = [self._parse_value(row['solv1_x']) for row in self.data]
        return {
            'min': min(compositions),
            'max': max(compositions)
        }

    def search_chemical(self, chemical_name: str) -> List:
        """Search for a chemical by name.

        Args:
            chemical_name: Name of the chemical to search for

        Returns:
            List containing solvent_id and indices of matching rows
        """
        for solvent_id, info in self.solvent_info.items():
            if chemical_name.lower() == info['name'].lower():
                print(f"{solvent_id}, {info['name']}, {info['smiles']}")
                indices = [
                    i for i, row in enumerate(self.data)
                    if row['solv1'] == solvent_id or row['solv2'] == solvent_id
                ]
                return [solvent_id, indices]
        return [None, []]

    def search_chemical_pair(self, chemical_list: List[str]) -> List:
        """Search for a pair of chemicals.

        Args:
            chemical_list: List of two chemical names

        Returns:
            List containing solvent IDs and indices of matching rows
        """
        solv1_match = self.search_chemical(chemical_list[0])[0]
        solv2_match = self.search_chemical(chemical_list[1])[0]

        if solv1_match is None or solv2_match is None:
            return [[None, None], []]

        indices = [
            i for i, row in enumerate(self.data)
            if (row['solv1'] == solv1_match and row['solv2'] == solv2_match) or
               (row['solv1'] == solv2_match and row['solv2'] == solv1_match)
        ]

        return [[solv1_match, solv2_match], indices]


def collate_solvent_binary(batch: List[Dict]) -> Dict:
    """Collate function for batching binary solvent data.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary
    """
    keys = list(batch[0].keys())
    samples = list(map(lambda sample: sample.values(), batch))
    samples = list(map(list, zip(*samples)))

    batched_sample = {}

    # Handle molecular graphs (g1, g2)
    batched_sample['g1'] = paddle.stack(samples[0]) if isinstance(samples[0][0], paddle.Tensor) else samples[0]
    batched_sample['g2'] = paddle.stack(samples[1]) if isinstance(samples[1][0], paddle.Tensor) else samples[1]

    # Handle scalar values
    for i, key in enumerate(keys[2:]):
        batched_sample[key] = paddle.to_tensor(samples[i + 2], dtype='float32')

    return batched_sample
