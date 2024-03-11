import keras

from molexpress import types
from molexpress.ops import gnn_ops 


class Readout(keras.layers.Layer):

    def __init__(self, mode: str = 'mean', **kwargs) -> None:
        super().__init__(**kwargs)
        self.mode = mode
        if self.mode == 'max':
            self._readout_fn = keras.ops.segment_max 
        elif self.mode == 'sum':
            self._readout_fn = keras.ops.segment_sum
        else:
            self._readout_fn = gnn_ops.segment_mean

    def build(self, input_shape: dict[str, tuple[int, ...]]) -> None:
        if 'graph_indicator' not in input_shape:
            raise ValueError(
                "Cannot perform readout: 'graph_indicator' not found.")

    def call(self, inputs: types.MolecularGraph) -> types.Array:
        graph_indicator = keras.ops.cast(inputs['graph_indicator'], 'int32')
        return self._readout_fn(
            data=inputs['node_state'],
            segment_ids=graph_indicator,
            num_segments=None,
            sorted=False, 
        )