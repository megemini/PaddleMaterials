#!/usr/bin/env python
"""调试脚本：检查CO分子的原子特征"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rdkit import Chem
from ppmat.utils.molecular_graph import CanonicalAtomFeaturizer

# 测试CO分子
smiles = "CO"
mol = Chem.MolFromSmiles(smiles)

print(f"SMILES: {smiles}")
print(f"Number of atoms: {mol.GetNumAtoms()}")
print()

# 显示每个原子的信息
print("Atom details:")
for i in range(mol.GetNumAtoms()):
    atom = mol.GetAtomWithIdx(i)
    print(f"  Atom {i}:")
    print(f"    Symbol: {atom.GetSymbol()}")
    print(f"    Atomic number: {atom.GetAtomicNum()}")
    print(f"    Degree: {atom.GetDegree()}")
    print(f"    Total valence: {atom.GetTotalValence()}")
    print(f"    Formal charge: {atom.GetFormalCharge()}")
    print(f"    Radical electrons: {atom.GetNumRadicalElectrons()}")
    print(f"    Total Hs: {atom.GetTotalNumHs()}")
    print(f"    Hybridization: {atom.GetHybridization()}")
    print(f"    Is aromatic: {atom.GetIsAromatic()}")
    print(f"    Mass: {atom.GetMass()}")
    print()

# 尝试创建特征
featurizer = CanonicalAtomFeaturizer()
try:
    features = featurizer(mol)
    print("Featurization successful!")
    print(f"Feature shape: {features['h'].shape}")
    print()

    # 检查每个原子的特征
    print("Atom features:")
    for i in range(mol.GetNumAtoms()):
        print(f"  Atom {i}:")
        print(f"    Feature vector length: {len(features['h'][i])}")
        print(f"    Non-zero indices: {[j for j, v in enumerate(features['h'][i]) if v > 0]}")
except Exception as e:
    print(f"Featurization failed: {e}")
    import traceback
    traceback.print_exc()

# 检查允许的值范围
print("\n" + "=" * 80)
print("Featurizer allowable values:")
print("=" * 80)
print(f"allowable_atom_types: {len(featurizer.allowable_atom_types)} types")
print(f"allowable_degree: {featurizer.allowable_degree}")
print(f"allowable_num_hs: {featurizer.allowable_num_hs}")
print(f"allowable_valence: {featurizer.allowable_valence}")
print(f"allowable_hybridization: {len(featurizer.allowable_hybridization)} types")

# 计算总维度
total_dim = (
    len(featurizer.allowable_atom_types) +
    len(featurizer.allowable_degree) +
    1 +  # formal charge
    1 +  # radical electrons
    len(featurizer.allowable_num_hs) +
    len(featurizer.allowable_hybridization) +
    1 +  # aromatic
    1 +  # mass
    len(featurizer.allowable_valence)
)
print(f"\nTotal expected dimension: {total_dim}")
print(f"Actual dimension in code: 74")
print(f"Match: {total_dim == 74}")
