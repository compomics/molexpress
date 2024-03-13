from abc import ABC 
from abc import abstractmethod

from rdkit.Chem import Lipinski
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdPartialCharges

import numpy as np
import math

from molexpress import types


DEFAULT_VOCABULARY = {
    'AtomType': {
        'H',  'He', 'Li', 'Be', 'B',  'C',  'N',  'O',  'F',  'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P',  'S',  'Cl', 'Ar', 'K',  'Ca',
        'Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y',  'Zr',
        'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
        'Sb', 'Te', 'I',  'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
        'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
        'Lu', 'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
        'Pa', 'U',  'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
        'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
        'Rg', 'Cn'
    },
    'Hybridization': {
        's', 'sp', 'sp2', 'sp3', 'sp3d', 'sp3d2', 'unspecified'
    },
    'CIPCode': {
        'R', 'S', 'None'
    },
    'FormalCharge': {
        -3, -2, -1, 0, 1, 2, 3, 4
    },
    'TotalNumHs': {
        0, 1, 2, 3, 4
    },
    'TotalValence': {
        0, 1, 2, 3, 4, 5, 6, 7, 8
    },
    'NumRadicalElectrons': {
        0, 1, 2, 3
    },
    'Degree': {
        0, 1, 2, 3, 4, 5, 6, 7, 8
    },
    'RingSize': {
        0, 3, 4, 5, 6, 7, 8
    },
    'BondType': {
        'single', 'double', 'triple', 'aromatic'
    },
    'Stereo': {
        'stereoe', 'stereoz', 'stereoany', 'stereonone'
    },
}


class Compose:

    """Wraps a list of features to featurize an atom or bond.
    
    While a Feature encodes an atom or bond based on a single feature,
    Compose encodes an atom or bond based on multiple features.

    Args:
        features:
            List of features.
    """

    def __init__(self, features: list['Feature']) -> None:
        self.features = features
        assert all(
            self.features[0].dtype == f.dtype for f in self.features
        ), "'dtype' of features need to be consistent."

    def __call__(self, inputs: types.Atom | types.Bond) -> np.ndarray:
        return np.concatenate([
            feature(inputs) for feature in self.features
        ])
    
    @property
    def dim(self):
        return sum(feature.dim for feature in self.features)

    @property
    def dtype(self):
        return self.features[0].dtype
    

class Feature(ABC):

    """Abstract feature.
    
    Represents a single feature of an atom or bond.
    """

    def __init__(self, dim: int = None, dtype: str = 'float32') -> None:
        self._dim = int(dim) if dim is not None else 1
        self._dtype = dtype
    
    @abstractmethod
    def call(self, x: types.Atom | types.Bond) -> types.Scalar:
        pass

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def dtype(self) -> str:
        return self._dtype


class OneHotFeature(Feature):

    """Abstract one-hot feature."""

    def __init__(
        self,
        vocab: list[str] | list[int] = None, 
        oov: bool = False,
        dtype: str = 'float32',
    ):
        if not vocab:
            vocab = DEFAULT_VOCABULARY.get(self.__class__.__name__)
            if vocab is None:
                raise ValueError("Need to supply a 'vocab'.")
        
        self.vocab = list(vocab) 
        self.vocab.sort(key=lambda x: x if x is not None else "")
        self.oov = oov

        super().__init__(
            dim=len(self.vocab) + int(self.oov), 
            dtype=dtype
        )

        if self.oov:
            self.vocab += ['<oov>']
            
        encodings = np.eye(self.dim, dtype=self.dtype)
        self.mapping = dict(zip(self.vocab, encodings))
    
    def __call__(self, x: types.Atom | types.Bond) -> np.ndarray:
        feature = self.call(x)
        encoding = self.mapping.get(
            feature, None if not self.oov else self.mapping['<oov>']
        )
        if encoding is not None:
            return encoding
        return np.zeros([self.dim], dtype=self.dtype)

    
class FloatFeature(Feature):

    """Abstract scalar floating point feature."""

    def __call__(self, x: types.Atom | types.Bond) -> np.ndarray:
        return np.array([self.call(x)], dtype=self.dtype)


class AtomType(OneHotFeature):
    def call(self, inputs: types.Atom) -> str:
        return inputs.GetSymbol() 


class Hybridization(OneHotFeature):
    def call(self, inputs: types.Atom) -> str:
        return inputs.GetHybridization().name.lower()
    

class CIPCode(OneHotFeature):
    def call(self, atom: types.Atom) -> str | None:
        if atom.HasProp("_CIPCode"):
            return atom.GetProp("_CIPCode")
        return 'None'


class ChiralCenter(FloatFeature):
    def call(self, atom: types.Atom) -> bool:
        return atom.HasProp("_ChiralityPossible")


class FormalCharge(OneHotFeature):
    def call(self, atom: types.Atom) -> int:
        return atom.GetFormalCharge()


class TotalNumHs(OneHotFeature):
    def call(self, atom: types.Atom) -> int:
        return atom.GetTotalNumHs()


class TotalValence(OneHotFeature):
    def call(self, atom: types.Atom) -> int:
        return atom.GetTotalValence()


class NumRadicalElectrons(OneHotFeature):
    def call(self, atom: types.Atom) -> int:
        return atom.GetNumRadicalElectrons()


class Degree(OneHotFeature):
    def call(self, atom: types.Atom) -> int:
        return atom.GetDegree()


class Aromatic(FloatFeature):
    def call(self, atom: types.Atom) -> bool:
        return atom.GetIsAromatic()


class Hetero(FloatFeature):
    def call(self, atom: types.Atom) -> bool:
        mol = atom.GetOwningMol()
        return atom.GetIdx() in [i[0] for i in Lipinski._Heteroatoms(mol)]


class HydrogenDonor(FloatFeature):
    def call(self, atom: types.Atom) -> bool:
        mol = atom.GetOwningMol()
        return atom.GetIdx() in [i[0] for i in Lipinski._HDonors(mol)]


class HydrogenAcceptor(FloatFeature):
    def call(self, atom: types.Atom) -> bool:
        mol = atom.GetOwningMol()
        return atom.GetIdx() in [i[0] for i in Lipinski._HAcceptors(mol)]


class RingSize(OneHotFeature):
    def call(self, atom: types.Atom) -> int:
        size = 0
        if atom.IsInRing():
            while not atom.IsInRingSize(size):
                size += 1
        return size


class Ring(FloatFeature):
    def call(self, atom: types.Atom) -> bool:
        return atom.IsInRing()


class CrippenLogPContribution(FloatFeature):
    def call(self, atom: types.Atom) -> float:
        mol = atom.GetOwningMol()
        val = Crippen._GetAtomContribs(mol)[atom.GetIdx()][0]
        if val is not None and math.isfinite(val):
            return val
        return 0.0


class CrippenMolarRefractivityContribution(FloatFeature):
    def call(self, atom: types.Atom) -> float:
        mol = atom.GetOwningMol()
        val = Crippen._GetAtomContribs(mol)[atom.GetIdx()][1]
        if val is not None and math.isfinite(val):
            return val
        return 0.0


class TPSAContribution(FloatFeature):
    def call(self, atom: types.Atom) -> float:
        mol = atom.GetOwningMol()
        val = rdMolDescriptors._CalcTPSAContribs(mol)[atom.GetIdx()]
        if val is not None and math.isfinite(val):
            return val
        return 0.0


class LabuteASAContribution(FloatFeature):
    def call(self, atom: types.Atom) -> float:
        mol = atom.GetOwningMol()
        val = rdMolDescriptors._CalcLabuteASAContribs(mol)[0][atom.GetIdx()]
        if val is not None and math.isfinite(val):
            return val
        return 0.0


class GasteigerCharge(FloatFeature):
    def call(self, atom: types.Atom) -> float:
        mol = atom.GetOwningMol()
        rdPartialCharges.ComputeGasteigerCharges(mol)
        val = atom.GetDoubleProp('_GasteigerCharge')
        if val is not None and math.isfinite(val):
            return val
        return 0.0


class BondType(OneHotFeature):
    def call(self, bond: types.Bond) -> str:
        return bond.GetBondType().name.lower()


class Stereo(OneHotFeature):
    def call(self, bond: types.Bond) -> str:
        return bond.GetStereo().name.lower()
    

class Conjugated(FloatFeature):
    def call(self, bond: types.Bond) -> bool:
        return bond.GetIsConjugated()


class Rotatable(FloatFeature):
    def call(self, bond: types.Bond) -> bool:
        mol = bond.GetOwningMol()
        atom_indices = tuple(
            sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]))
        return atom_indices in Lipinski._RotatableBonds(mol)
