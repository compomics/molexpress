from abc import ABC 
from abc import abstractmethod

from rdkit import Chem 
from rdkit.Chem import Lipinski
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdPartialCharges

import numpy as np
import math


ALLOWABLE_SETS = {
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
        'S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'UNSPECIFIED'
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
        'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'
    },
    'Stereo': {
        'STEREOE', 'STEREOZ', 'STEREOANY', 'STEREONONE'
    },
}


class Featurizer:

    """Wraps a list of features to featurize an atom or bond.
    
    While a Feature encodes an atom or bond based on a single feature,
    the Featurizer encodes an atom or bond based on multiple features.

    Args:
        features:
            List of features.
    """

    def __init__(self, features: list['Feature']) -> None:
        self.features = features
        assert all(
            self.features[0].dtype == f.dtype for f in self.features
        ), "'dtype' of features need to be consistent."

    def __call__(self, inputs: Chem.Atom | Chem.Bond) -> np.ndarray:
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
    
    Computes atom and bond features for the molecular graph. Atom and bond
    features correspond to the (initial) node and edge states, respectively.
    """

    def __init__(self, allowable_set=None, dtype='float32'):
        if allowable_set is None:
            allowable_set = ALLOWABLE_SETS.get(
                self.__class__.__name__
            )
        self.dtype = dtype
        self.allowable_set = allowable_set 

    @abstractmethod 
    def __call__(self, inputs):
        pass

    @property
    def dim(self):
        if not hasattr(self, '_dim'):
            return 1 
        return self._dim


class OneHotFeature(Feature):

    """Base class for one-hot features."""

    def __init__(
        self,
        allowable_set: list[str] | list[int] = None, 
        dtype: str = 'float32',
        oov: bool = False,
    ):
        super().__init__(allowable_set, dtype)

        self.oov = oov
        keys = list(self.allowable_set)
        keys.sort(key=lambda x: x if x is not None else "")

        self._dim = len(self.allowable_set)
        
        if self.oov:
            self._dim += 1
            keys += ['<oov>']
            
        values = np.eye(self.dim, dtype=self.dtype)
        self._mapping = dict(zip(keys, values))
    
    def __call__(self, inputs: Chem.Atom | Chem.Bond) -> np.ndarray:
        feature = self.call(inputs)
        encoding = self._mapping.get(feature)
        if encoding is not None:
            return encoding
        if self.oov:
            return self._mapping.get('<oov>')
        return np.zeros((self.dim,), dtype=self.dtype)
                
    @abstractmethod
    def call(self, inputs):
        pass

    
class FloatFeature(Feature):

    """Base class for scalar floating point features."""

    def __call__(self, inputs: Chem.Atom | Chem.Bond) -> np.ndarray:
        return np.array([self.call(inputs)], dtype=self.dtype)
    
    @abstractmethod
    def call(self, inputs):
        pass


class AtomType(OneHotFeature):
    def call(self, inputs: Chem.Atom) -> str:
        return inputs.GetSymbol() 


class Hybridization(OneHotFeature):
    def call(self, inputs: Chem.Atom) -> str:
        return inputs.GetHybridization().name
    

class CIPCode(OneHotFeature):
    def call(self, atom: Chem.Atom) -> str | None:
        if atom.HasProp("_CIPCode"):
            return atom.GetProp("_CIPCode")
        return 'None'


class ChiralCenter(FloatFeature):
    def call(self, atom: Chem.Atom) -> bool:
        return atom.HasProp("_ChiralityPossible")


class FormalCharge(OneHotFeature):
    def call(self, atom: Chem.Atom) -> int:
        return atom.GetFormalCharge()


class TotalNumHs(OneHotFeature):
    def call(self, atom: Chem.Atom) -> int:
        return atom.GetTotalNumHs()


class TotalValence(OneHotFeature):
    def call(self, atom: Chem.Atom) -> int:
        return atom.GetTotalValence()


class NumRadicalElectrons(OneHotFeature):
    def call(self, atom: Chem.Atom) -> int:
        return atom.GetNumRadicalElectrons()


class Degree(OneHotFeature):
    def call(self, atom: Chem.Atom) -> int:
        return atom.GetDegree()


class Aromatic(FloatFeature):
    def call(self, atom: Chem.Atom) -> bool:
        return atom.GetIsAromatic()


class Hetero(FloatFeature):
    def call(self, atom: Chem.Atom) -> bool:
        mol = atom.GetOwningMol()
        return atom.GetIdx() in [i[0] for i in Lipinski._Heteroatoms(mol)]


class HydrogenDonor(FloatFeature):
    def call(self, atom: Chem.Atom) -> bool:
        mol = atom.GetOwningMol()
        return atom.GetIdx() in [i[0] for i in Lipinski._HDonors(mol)]


class HydrogenAcceptor(FloatFeature):
    def call(self, atom: Chem.Atom) -> bool:
        mol = atom.GetOwningMol()
        return atom.GetIdx() in [i[0] for i in Lipinski._HAcceptors(mol)]


class RingSize(OneHotFeature):
    def call(self, atom: Chem.Atom) -> int:
        size = 0
        if atom.IsInRing():
            while not atom.IsInRingSize(size):
                size += 1
        return size


class Ring(FloatFeature):
    def call(self, atom: Chem.Atom) -> bool:
        return atom.IsInRing()


class CrippenLogPContribution(FloatFeature):
    def call(self, atom: Chem.Atom) -> float:
        mol = atom.GetOwningMol()
        val = Crippen._GetAtomContribs(mol)[atom.GetIdx()][0]
        if val is not None and math.isfinite(val):
            return val
        return 0.0


class CrippenMolarRefractivityContribution(FloatFeature):
    def call(self, atom: Chem.Atom) -> float:
        mol = atom.GetOwningMol()
        val = Crippen._GetAtomContribs(mol)[atom.GetIdx()][1]
        if val is not None and math.isfinite(val):
            return val
        return 0.0


class TPSAContribution(FloatFeature):
    def call(self, atom: Chem.Atom) -> float:
        mol = atom.GetOwningMol()
        val = rdMolDescriptors._CalcTPSAContribs(mol)[atom.GetIdx()]
        if val is not None and math.isfinite(val):
            return val
        return 0.0


class LabuteASAContribution(FloatFeature):
    def call(self, atom: Chem.Atom) -> float:
        mol = atom.GetOwningMol()
        val = rdMolDescriptors._CalcLabuteASAContribs(mol)[0][atom.GetIdx()]
        if val is not None and math.isfinite(val):
            return val
        return 0.0


class GasteigerCharge(FloatFeature):
    def call(self, atom: Chem.Atom) -> float:
        mol = atom.GetOwningMol()
        rdPartialCharges.ComputeGasteigerCharges(mol)
        val = atom.GetDoubleProp('_GasteigerCharge')
        if val is not None and math.isfinite(val):
            return val
        return 0.0


class BondType(OneHotFeature):
    def call(self, bond: Chem.Bond) -> str:
        return bond.GetBondType().name


class Conjugated(OneHotFeature):
    def call(self, bond: Chem.Bond) -> bool:
        return bond.GetIsConjugated()


class Rotatable(FloatFeature):
    def call(self, bond: Chem.Bond) -> bool:
        mol = bond.GetOwningMol()
        atom_indices = tuple(
            sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]))
        return atom_indices in Lipinski._RotatableBonds(mol)


class Stereo(OneHotFeature):
    def call(self, bond: Chem.Bond) -> str:
        return bond.GetStereo().name