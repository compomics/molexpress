from __future__ import annotations

import keras

from molexpress import types
from molexpress.ops import gnn_ops


class PeptideReadout(keras.layers.Layer):
    def __init__(self, mode: str = "mean", **kwargs) -> None:
        super().__init__(**kwargs)
        self.mode = mode
        if self.mode == "max":
            self._readout_fn = keras.ops.segment_max
        elif self.mode == "sum":
            self._readout_fn = keras.ops.segment_sum
        else:
            self._readout_fn = gnn_ops.segment_mean

    def build(self, input_shape: dict[str, tuple[int, ...]]) -> None:
        if "peptide_size" not in input_shape:
            raise ValueError("Cannot perform readout: 'peptide_size' not found.")

    def call(self, inputs: types.MolecularGraph) -> types.Array:
        peptide_size = keras.ops.cast(inputs['peptide_size'], 'int32')
        residue_size = keras.ops.cast(inputs['residue_size'], 'int32')
        n_peptides = keras.ops.shape(peptide_size)[0]
        repeats = keras.ops.segment_sum(
            residue_size, 
            keras.ops.repeat(range(n_peptides), peptide_size)
        )
        segment_ids = keras.ops.repeat(range(n_peptides), repeats)
        return self._readout_fn(
            data=inputs["node_state"],
            segment_ids=segment_ids,
            num_segments=None,
            sorted=False,
        )
