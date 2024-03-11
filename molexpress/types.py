from typing import Any
from typing import TypedDict

Array = Any
Variable = Any


class MolecularGraph(TypedDict):
    node_state: Array 
    edge_src: Array
    edge_dst: Array 
    edge_state: Array | None 
    edge_weight: Array | None 
    graph_indicator: Array | None