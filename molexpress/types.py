from __future__ import annotations

from typing import (
    Any,  # noqa: F401
    Protocol,  # noqa: F401
    TypedDict,
    TypeVar,
)

from rdkit import Chem

Array = TypeVar("Array")
Variable = TypeVar("Variable")

Shape = TypeVar("Shape")
DType = TypeVar("DType")

Molecule = Chem.Mol
Atom = Chem.Atom
Bond = Chem.Bond

SMILES = TypeVar("SMILES", bound=str)
InChI = TypeVar("InChI", bound=str)


class MolecularGraph(TypedDict):
    node_state: Array
    edge_src: Array
    edge_dst: Array
    edge_state: Array | None
    edge_weight: Array | None
    graph_indicator: Array | None
