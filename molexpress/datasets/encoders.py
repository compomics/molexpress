from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple, Union

import numpy as np

from molexpress import types
from molexpress.datasets import featurizers
from molexpress.ops import chem_ops


class PeptideGraphEncoder:
    def __init__(
        self,
        atom_featurizers: list[featurizers.Featurizer],
        bond_featurizers: list[featurizers.Featurizer] = None,
        self_loops: bool = False,
    ) -> None:
        self.node_encoder = MolecularNodeEncoder(atom_featurizers)
        self.edge_encoder = MolecularEdgeEncoder(bond_featurizers, self_loops=self_loops)

    def __call__(self, residues: list[types.Molecule | types.SMILES | types.InChI]) -> np.ndarray:
        residue_graphs = []
        residue_sizes = []
        for residue in residues:
            residue_graph, residue_size = self._encode_residue(
                residue, self.node_encoder, self.edge_encoder
            )
            residue_graphs.append(residue_graph)
            residue_sizes.append(residue_size)

        disjoint_peptide_graph = self._merge_molecular_graphs(residue_graphs)
        disjoint_peptide_graph["residue_size"] = np.array(residue_sizes)
        disjoint_peptide_graph["peptide_size"] = np.array([len(residues)], dtype="int32")
        return disjoint_peptide_graph

    @staticmethod
    @lru_cache(maxsize=None)
    def _encode_residue(
        residue: types.Molecule | types.SMILES | types.InChI,
        node_encoder: MolecularNodeEncoder,
        edge_encoder: MolecularEdgeEncoder,
    ) -> Tuple[Dict[str, np.ndarray], int]:
        residue = chem_ops.get_molecule(residue)
        residue_graph = {**node_encoder(residue), **edge_encoder(residue)}
        return residue_graph, residue.GetNumAtoms()

    @staticmethod
    def collate_fn(
        data: list[Union[types.MolecularGraph, tuple[types.MolecularGraph, np.ndarray]]],
    ) -> tuple[types.MolecularGraph, np.ndarray]:
        """
        Merge list of graphs into a single disjoint graph.

        Data can be a list of MolecularGraphs or a list of tuples where the first element is a
        MolecularGraph and the second element is a label.

        """
        if isinstance(data[0], tuple):
            disjoint_peptide_graphs, y = list(zip(*data))
        else:
            disjoint_peptide_graphs = data
            y = None

        disjoint_peptide_batch_graph = PeptideGraphEncoder._merge_molecular_graphs(
            disjoint_peptide_graphs
        )
        disjoint_peptide_batch_graph["peptide_size"] = np.concatenate(
            [g["residue_size"].shape[:1] for g in disjoint_peptide_graphs]
        ).astype("int32")
        disjoint_peptide_batch_graph["residue_size"] = np.concatenate(
            [g["residue_size"] for g in disjoint_peptide_graphs]
        ).astype("int32")

        if y is None:
            return disjoint_peptide_batch_graph
        else:
            return disjoint_peptide_batch_graph, np.stack(y)

    @staticmethod
    def _merge_molecular_graphs(
        molecular_graphs: list[types.MolecularGraph],
    ) -> types.MolecularGraph:
        num_nodes = np.array([g["node_state"].shape[0] for g in molecular_graphs])

        disjoint_molecular_graph = {}

        disjoint_molecular_graph["node_state"] = np.concatenate(
            [g["node_state"] for g in molecular_graphs]
        )

        if "edge_state" in molecular_graphs[0]:
            disjoint_molecular_graph["edge_state"] = np.concatenate(
                [g["edge_state"] for g in molecular_graphs]
            )

        edge_src = np.concatenate([graph["edge_src"] for graph in molecular_graphs])
        edge_dst = np.concatenate([graph["edge_dst"] for graph in molecular_graphs])
        num_edges = np.array([graph["edge_src"].shape[0] for graph in molecular_graphs])
        indices = np.repeat(range(len(molecular_graphs)), num_edges)
        edge_incr = np.concatenate([[0], num_nodes[:-1]])
        edge_incr = np.take_along_axis(edge_incr, indices, axis=0)

        disjoint_molecular_graph["edge_src"] = edge_src + edge_incr
        disjoint_molecular_graph["edge_dst"] = edge_dst + edge_incr

        return disjoint_molecular_graph


class Composer:
    """Wraps a list of featurizers.

    While a Featurizer encodes an atom or bond based on a single property,
    the Composer encodes an atom or bond based on multiple properties.

    Args:
        featurizers:
            List of featurizers.
    """

    def __init__(self, featurizers: list[featurizers.Featurizer]) -> None:
        self.featurizers = featurizers
        assert all(
            self.featurizers[0].output_dtype == f.output_dtype for f in self.featurizers
        ), "'dtype' of features need to be consistent."

    def __call__(self, inputs: types.Atom | types.Bond) -> np.ndarray:
        return np.concatenate([f(inputs) for f in self.featurizers])

    @property
    def output_dim(self):
        return sum(f.output_dim for f in self.featurizers)

    @property
    def output_dtype(self):
        return self.featurizers[0].output_dtype


class MolecularEdgeEncoder:
    def __init__(
        self, featurizers: list[featurizers.Featurizer], self_loops: bool = False
    ) -> None:
        self.featurizer = Composer(featurizers)
        self.self_loops = self_loops
        self.output_dim = self.featurizer.output_dim
        self.output_dtype = self.featurizer.output_dtype

    def __call__(self, molecule: types.Molecule) -> np.ndarray:
        edge_src, edge_dst = chem_ops.get_adjacency(molecule, self_loops=self.self_loops)

        if self.featurizer is None:
            return {"edge_src": edge_src, "edge_dst": edge_dst}

        if molecule.GetNumBonds() == 0:
            edge_state = np.zeros(
                shape=(int(self.self_loops), self.output_dim + int(self.self_loops)),
                dtype=self.output_dtype,
            )
            return {
                "edge_src": edge_src,
                "edge_dst": edge_dst,
                "edge_state": edge_state,
            }

        bond_encodings = []

        for i, j in zip(edge_src, edge_dst):
            bond = molecule.GetBondBetweenAtoms(int(i), int(j))

            if bond is None:
                assert self.self_loops, "Found a bond to be None."
                bond_encoding = np.zeros(self.output_dim + 1, dtype=self.output_dtype)
                bond_encoding[-1] = 1
            else:
                bond_encoding = self.featurizer(bond)
                if self.self_loops:
                    bond_encoding = np.pad(bond_encoding, (0, 1))

            bond_encodings.append(bond_encoding)

        return {
            "edge_src": edge_src,
            "edge_dst": edge_dst,
            "edge_state": np.stack(bond_encodings),
        }


class MolecularNodeEncoder:
    def __init__(
        self,
        featurizers: list[featurizers.Featurizer],
    ) -> None:
        self.featurizer = Composer(featurizers)

    def __call__(self, molecule: types.Molecule) -> np.ndarray:
        node_encodings = np.stack([self.featurizer(atom) for atom in molecule.GetAtoms()], axis=0)
        return {
            "node_state": np.stack(node_encodings),
        }
