import numpy as np
from rdkit import Chem

from molexpress import types
from molexpress.ops import chem_ops
from molexpress import types


class MolecularGraphEncoder:

    def __init__(
        self,
        atom_featurizer: types.Featurizer,
        bond_featurizer: types.Featurizer = None,
        self_loops: bool = False, 
    ) -> None:
        self.node_encoder = MolecularNodeEncoder(atom_featurizer)
        self.edge_encoder = MolecularEdgeEncoder(
            bond_featurizer, self_loops=self_loops
        )

    def __call__(
        self, 
        molecule: types.Molecule | types.SMILES | types.InChI
    ) -> np.ndarray:
        molecule = chem_ops.get_molecule(molecule)
        return {**self.node_encoder(molecule), **self.edge_encoder(molecule)}

    @staticmethod
    def _collate_fn(
        data: list[tuple[types.MolecularGraph, np.ndarray]]
    ) -> tuple[types.MolecularGraph, np.ndarray]:
        
        """TODO: Not sure where to implement this collate function. 
                 Temporarily putting it here.

        Procedure:
            Merges list of graphs into a single disjoint graph.
        """

        x, y = list(zip(*data)) 
        
        num_nodes = np.array([
            graph['node_state'].shape[0] for graph in x
        ])
        
        disjoint_graph = {}

        disjoint_graph['node_state'] = np.concatenate([
            graph['node_state'] for graph in x
        ])

        if 'edge_state' in x[0]:
            disjoint_graph['edge_state'] = np.concatenate([
                graph['edge_state'] for graph in x
            ])

        edge_src = np.concatenate([graph['edge_src'] for graph in x])
        edge_dst = np.concatenate([graph['edge_dst'] for graph in x])
        num_edges = np.array([graph['edge_src'].shape[0] for graph in x])
        indices = np.repeat(range(len(x)), num_edges) 
        edge_incr = np.concatenate([[0], num_nodes[:-1]])
        edge_incr = np.take_along_axis(edge_incr, indices, axis=0)

        disjoint_graph['edge_src'] = edge_src + edge_incr
        disjoint_graph['edge_dst'] = edge_dst + edge_incr 
        disjoint_graph['graph_indicator'] = np.repeat(range(len(x)), num_nodes)

        return disjoint_graph, np.stack(y)


class MolecularEdgeEncoder:

    def __init__(
        self, 
        featurizer: types.Featurizer, 
        self_loops: bool = False
    ) -> None:
        self.featurizer = featurizer 
        self.self_loops = self_loops
        self.dim = featurizer.dim
        self.dtype = featurizer.dtype

    def __call__(self, molecule: Chem.Mol) -> np.ndarray:

        edge_src, edge_dst = chem_ops.get_adjacency(
            molecule, self_loops=self.self_loops)

        if self.featurizer is None:
            return {'edge_src': edge_src, 'edge_dst': edge_dst}

        if molecule.GetNumBonds() == 0:
            return np.zeros((0, self.dim), dtype=self.dtype)
        
        bond_encodings = []

        for i, j in zip(edge_src, edge_dst):
            
            bond = molecule.GetBondBetweenAtoms(int(i), int(j))

            if bond is None:
                assert self.self_loops, "Found a bond to be None."
                bond_encoding = np.zeros(self.dim + 1, dtype=self.dtype)
                bond_encoding[-1] = 1
            else:
                bond_encoding = self.featurizer(bond)
                if self.self_loops:
                    bond_encoding = np.pad(bond_encoding, (0, 1))

            bond_encodings.append(bond_encoding)

        return {
            'edge_src': edge_src, 
            'edge_dst': edge_dst, 
            'edge_state': np.stack(bond_encodings)
        }
    

class MolecularNodeEncoder:

    def __init__(
        self, 
        featurizer: types.Featurizer, 
    ) -> None:
        self.featurizer = featurizer 

    def __call__(self, molecule: Chem.Mol) -> np.ndarray:
        node_encodings = np.stack([
            self.featurizer(atom) for atom in molecule.GetAtoms()
        ], axis=0)
        return {'node_state': np.stack(node_encodings)}
    