from __future__ import annotations

import keras

from molexpress import types
from molexpress.ops import gnn_ops


class ResidueReadout(keras.layers.Layer):
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
        if "residue_size" not in input_shape:
            raise ValueError("Cannot perform readout: 'residue_size' not found.")

    def call(self, inputs: types.MolecularGraph) -> types.Array:
        peptide_size = keras.ops.cast(inputs['peptide_size'], 'int32')
        residue_size = keras.ops.cast(inputs['residue_size'], 'int32')
        n_residues = keras.ops.shape(residue_size)[0]
        segment_ids = keras.ops.repeat(range(n_residues), residue_size)
        residue_state = self._readout_fn(
            data=inputs["node_state"],
            segment_ids=segment_ids,
            num_segments=None,
            sorted=False,
        )
        # Make shape known
        residue_state = keras.ops.reshape(
            residue_state, 
            (
                keras.ops.shape(residue_size)[0], 
                keras.ops.shape(inputs['node_state'])[-1]
            )
        )
        
        if keras.ops.shape(peptide_size)[0] == 1:
            # Single peptide in batch
            return residue_state[None]
        
        # Split and stack (with padding in the second dim)
        # Resulting shape: (n_peptides, n_residues, n_features)
        residues = keras.ops.split(residue_state, peptide_size[:-1])
        max_residue_size = keras.ops.max([len(r) for r in residues])
        return keras.ops.stack([
            keras.ops.pad(r, [(0, max_residue_size-keras.ops.shape(r)[0]), (0, 0)])
            for r in residues
        ])



