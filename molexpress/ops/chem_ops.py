import numpy as np
from rdkit import Chem 


def get_molecule(
    molecule: Chem.Mol | str,
    catch_errors: bool = False,
) -> Chem.Mol | None:

    """Generates an molecule object."""

    if isinstance(molecule, Chem.Mol):
        return molecule

    string = molecule 

    if string.startswith('InChI'):
        molecule = Chem.MolFromInchi(string, sanitize=False)
    else:
        molecule = Chem.MolFromSmiles(string, sanitize=False)

    if molecule is None:
        raise ValueError(f"{string!r} is invalid.")

    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        if not catch_errors:
            return None
        # Sanitize molecule again, without the sanitization step that caused 
        # the error previously. Unrealistic molecules might pass without an error.
        Chem.SanitizeMol(
            molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^flag)

    Chem.AssignStereochemistry(
        molecule, cleanIt=True, force=True, flagPossibleStereoCenters=True)

    return molecule

def get_adjacency(
    molecule: Chem.Mol,
    self_loops: bool = False,
    sparse: bool = True,
    dtype: str = 'int32',
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    
    """Computes the (sparse) adjacency matrix of the molecule"""

    adjacency_matrix = Chem.GetAdjacencyMatrix(molecule)

    if self_loops:
        adjacency_matrix += np.eye(
            adjacency_matrix.shape[0], dtype=adjacency_matrix.dtype
        )

    if not sparse:
        return adjacency_matrix.astype(dtype)

    edge_src, edge_dst = np.where(adjacency_matrix)
    return edge_src.astype(dtype), edge_dst.astype(dtype)


