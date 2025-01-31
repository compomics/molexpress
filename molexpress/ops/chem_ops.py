from __future__ import annotations

import numpy as np
from rdkit import Chem

from molexpress import types


def get_molecule(
    input_molecule: types.Molecule | types.SMILES | types.InChI,
    catch_errors: bool = False,
) -> Chem.Mol | None:
    """Generates an molecule object."""

    if isinstance(input_molecule, Chem.Mol):
        return input_molecule

    if input_molecule.startswith("InChI"):
        molecule = Chem.MolFromInchi(input_molecule, sanitize=False)
    else:
        molecule = Chem.MolFromSmiles(input_molecule, sanitize=False)

    if not molecule:
        raise ValueError(f"{input_molecule!r} is invalid.")

    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        if catch_errors:
            raise ValueError(f"{input_molecule!r} is invalid.")
        else:
            # Sanitize molecule again, without the sanitization step that caused
            # the error previously. Unrealistic molecules might pass without an error.
            Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)

    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True, flagPossibleStereoCenters=True)

    return molecule


def get_adjacency(
    molecule: types.Molecule,
    self_loops: bool = False,
    sparse: bool = True,
    dtype: str = "int32",
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Computes the (sparse) adjacency matrix of the molecule"""

    adjacency_matrix: np.ndarray = Chem.GetAdjacencyMatrix(molecule)

    if self_loops:
        adjacency_matrix += np.eye(adjacency_matrix.shape[0], dtype=adjacency_matrix.dtype)

    if not sparse:
        return adjacency_matrix.astype(dtype)

    edge_src, edge_dst = np.where(adjacency_matrix)
    return edge_src.astype(dtype), edge_dst.astype(dtype)
