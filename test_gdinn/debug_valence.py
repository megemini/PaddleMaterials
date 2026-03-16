#!/usr/bin/env python3
"""Debug script to check atom valence values."""

from rdkit import Chem

# Test with methanol
smiles = 'CO'
mol = Chem.MolFromSmiles(smiles)

print(f"SMILES: {smiles}")
print(f"Number of atoms: {mol.GetNumAtoms()}")
print()

for i in range(mol.GetNumAtoms()):
    atom = mol.GetAtomWithIdx(i)
    print(f"Atom {i}:")
    print(f"  Symbol: {atom.GetSymbol()}")
    print(f"  Atomic number: {atom.GetAtomicNum()}")
    print(f"  Degree: {atom.GetDegree()}")
    print(f"  Total valence: {atom.GetTotalValence()}")
    print(f"  Formal charge: {atom.GetFormalCharge()}")
    print(f"  Radical electrons: {atom.GetNumRadicalElectrons()}")
    print(f"  Total Hs: {atom.GetTotalNumHs()}")
    print(f"  Hybridization: {atom.GetHybridization()}")
    print(f"  Is aromatic: {atom.GetIsAromatic()}")
    print(f"  Mass: {atom.GetMass()}")
    print()
