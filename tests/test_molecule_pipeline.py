import math
import sys
import unittest
from pathlib import Path

from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from inference.molecule_inference import smiles_to_qm9_graph


MOLECULES = (
    "O", "C", "N", "CCO", "CC(=O)O", "c1ccccc1", "n1ccccc1",
    "C1CCCCC1", "C=C", "C#C", "CC(O)C", "CC(=O)C", "CCl", "[NH4+]",
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "[2H]O[2H]",
)


class MoleculePipelineTests(unittest.TestCase):
    def test_display_view_hides_hydrogens_without_reindexing_atoms(self):
        for smiles, expected_heavy_atoms, expected_heavy_bonds in (
            ("O", 1, 0),
            ("CCO", 3, 2),
            ("CC(=O)O", 4, 3),
            ("c1ccccc1", 6, 6),
            ("n1ccccc1", 6, 6),
            ("C=C", 2, 1),
        ):
            with self.subTest(smiles=smiles):
                data, _ = smiles_to_qm9_graph(smiles)
                self.assertEqual(len(data.display_heavy_atoms), expected_heavy_atoms)
                self.assertEqual(len(data.display_heavy_bonds), expected_heavy_bonds)
                self.assertTrue(all(atom["atomic_num"] != 1 for atom in data.display_heavy_atoms))
                visible_ids = {atom["id"] for atom in data.display_heavy_atoms}
                self.assertTrue(all(
                    bond["source"] in visible_ids and bond["target"] in visible_ids
                    for bond in data.display_heavy_bonds
                ))

    def test_supported_molecules_preserve_structure_and_model_shapes(self):
        for smiles in MOLECULES:
            with self.subTest(smiles=smiles):
                data, atomic_nums = smiles_to_qm9_graph(smiles)
                expected_mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                expected_atoms = expected_mol.GetNumAtoms()
                expected_bonds = expected_mol.GetNumBonds()
                expected_pairs = {
                    tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
                    for bond in expected_mol.GetBonds()
                }
                display_bonds = data.display_bonds
                displayed_pairs = {
                    tuple(sorted((bond["source"], bond["target"])))
                    for bond in display_bonds
                }
                edge_pairs = {
                    tuple(sorted(pair))
                    for pair in data.edge_index.t().tolist()
                }

                self.assertEqual(data.x.shape, (expected_atoms, 9))
                self.assertEqual(len(atomic_nums), expected_atoms)
                self.assertEqual(len(display_bonds), expected_bonds)
                self.assertEqual(len(displayed_pairs), expected_bonds)
                self.assertEqual(displayed_pairs, expected_pairs)
                self.assertEqual(len(edge_pairs), expected_bonds)
                self.assertEqual(data.edge_index.shape, (2, expected_bonds * 2))
                self.assertEqual(tuple(data.edge_attr.shape), (expected_bonds * 2, 4))
                self.assertTrue(bool(data.x.isfinite().all()))
                self.assertTrue(bool(data.edge_attr.isfinite().all()))
                self.assertEqual(len(data.display_pos), expected_atoms)
                self.assertEqual({atom["id"] for atom in data.display_atoms}, set(range(expected_atoms)))
                self.assertTrue(all(0 <= index < expected_atoms for index in range(expected_atoms)))

    def test_bond_features_retain_chemical_types(self):
        cases = {
            "C=C": (2.0, False),
            "C#C": (3.0, False),
            "c1ccccc1": (12.0, True),
        }
        for smiles, (bond_type, aromatic) in cases.items():
            with self.subTest(smiles=smiles):
                data, _ = smiles_to_qm9_graph(smiles)
                self.assertTrue(any(
                    math.isclose(bond["bond_type"], bond_type) and bond["is_aromatic"] == aromatic
                    for bond in data.display_bonds
                ))

    def test_empty_invalid_and_unsupported_inputs_fail_cleanly(self):
        for smiles, expected_text in (
            ("", "cannot be empty"),
            ("not_a_smiles", "Invalid SMILES"),
            ("[Na+]", "Unsupported element"),
        ):
            with self.subTest(smiles=smiles):
                with self.assertRaisesRegex(ValueError, expected_text):
                    smiles_to_qm9_graph(smiles)


if __name__ == "__main__":
    unittest.main()
