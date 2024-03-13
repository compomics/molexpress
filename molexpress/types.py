from typing import TypedDict
from typing import Protocol
from typing import TypeVar 
from typing import Any 

from rdkit import Chem


Scalar = TypeVar("Scalar")
Array = TypeVar("Array")
Variable = TypeVar("Variable")

Shape = TypeVar("Shape")
DType = TypeVar("DType")

Molecule = Chem.Mol
Atom = Chem.Atom 
Bond = Chem.Bond

SMILES = TypeVar("SMILES", bound=str)
InChI = TypeVar("InChI", bound=str) 

class Featurizer(Protocol):

    def __call__(self, x: Atom | Bond) -> Array:
        ...

    @property
    def dim(self) -> int:
        ...

    @property
    def dtype(self) -> DType:
        ...


class MolecularGraph(TypedDict):
    node_state: Array 
    edge_src: Array
    edge_dst: Array 
    edge_state: Array | None 
    edge_weight: Array | None 
    graph_indicator: Array | None
