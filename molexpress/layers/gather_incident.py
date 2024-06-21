import keras

from molexpress import types
from molexpress.ops import gnn_ops


class GatherIncident(keras.layers.Layer):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def call(self, inputs: types.MolecularGraph):
        node_state_src = gnn_ops.gather(
            inputs['node_state'], inputs['edge_src']
        )
        node_state_dst = gnn_ops.gather(
            inputs['node_state'], inputs['edge_dst']
        )
        return keras.ops.concatenate([node_state_src, node_state_dst], axis=1)