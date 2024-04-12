from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
from rdkit.Chem import Crippen, Lipinski, rdMolDescriptors, rdPartialCharges

from molexpress import types

DEFAULT_VOCABULARY = {
    "AtomType": {
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
        "Sb",
        "Te",
        "I",
        "Xe",
        "Cs",
        "Ba",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Tl",
        "Pb",
        "Bi",
        "Po",
        "At",
        "Rn",
        "Fr",
        "Ra",
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
        "Lr",
        "Rf",
        "Db",
        "Sg",
        "Bh",
        "Hs",
        "Mt",
        "Ds",
        "Rg",
        "Cn",
    },
    "Hybridization": {"s", "sp", "sp2", "sp3", "sp3d", "sp3d2", "unspecified"},
    "CIPCode": {"R", "S", "None"},
    "FormalCharge": {-3, -2, -1, 0, 1, 2, 3, 4},
    "TotalNumHs": {0, 1, 2, 3, 4},
    "TotalValence": {0, 1, 2, 3, 4, 5, 6, 7, 8},
    "NumRadicalElectrons": {0, 1, 2, 3},
    "Degree": {0, 1, 2, 3, 4, 5, 6, 7, 8},
    "RingSize": {0, 3, 4, 5, 6, 7, 8},
    "BondType": {"single", "double", "triple", "aromatic"},
    "Stereo": {"stereoe", "stereoz", "stereoany", "stereonone"},
}


class Featurizer(ABC):
    """Abstract featurizer.

    Featurizes a single atom or bond based on a single property.
    """

    def __init__(self, output_dim: int = None, output_dtype: str = "float32") -> None:
        self._output_dim = int(output_dim) if output_dim is not None else 1
        self._output_dtype = output_dtype

    @abstractmethod
    def call(self, x: types.Atom | types.Bond) -> types.Scalar:
        pass

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def output_dtype(self) -> str:
        return self._output_dtype


class OneHotFeaturizer(Featurizer):
    """Abstract one-hot featurizer."""

    def __init__(
        self,
        vocab: list[str] | list[int] = None,
        oov: bool = False,
        output_dtype: str = "float32",
    ):
        if not vocab:
            vocab = DEFAULT_VOCABULARY.get(self.__class__.__name__)
            if vocab is None:
                raise ValueError("Need to supply a 'vocab'.")

        self.vocab = list(vocab)
        self.vocab.sort(key=lambda x: x if x is not None else "")
        self.oov = oov

        super().__init__(output_dim=len(self.vocab) + int(self.oov), output_dtype=output_dtype)

        if self.oov:
            self.vocab += ["<oov>"]

        encodings = np.eye(self.output_dim, dtype=self.output_dtype)
        self.mapping = dict(zip(self.vocab, encodings))

    def __call__(self, x: types.Atom | types.Bond) -> np.ndarray:
        feature = self.call(x)
        encoding = self.mapping.get(feature, None if not self.oov else self.mapping["<oov>"])
        if encoding is not None:
            return encoding
        return np.zeros([self.output_dim], dtype=self.output_dtype)


class FloatFeaturizer(Featurizer):
    """Abstract scalar floating point featurizer."""

    def __call__(self, x: types.Atom | types.Bond) -> np.ndarray:
        return np.array([self.call(x)], dtype=self.output_dtype)


class AtomType(OneHotFeaturizer):
    def call(self, inputs: types.Atom) -> str:
        return inputs.GetSymbol()


class Hybridization(OneHotFeaturizer):
    def call(self, inputs: types.Atom) -> str:
        return inputs.GetHybridization().name.lower()


class CIPCode(OneHotFeaturizer):
    def call(self, atom: types.Atom) -> str | None:
        if atom.HasProp("_CIPCode"):
            return atom.GetProp("_CIPCode")
        return "None"


class ChiralCenter(FloatFeaturizer):
    def call(self, atom: types.Atom) -> bool:
        return atom.HasProp("_ChiralityPossible")


class FormalCharge(OneHotFeaturizer):
    def call(self, atom: types.Atom) -> int:
        return atom.GetFormalCharge()


class TotalNumHs(OneHotFeaturizer):
    def call(self, atom: types.Atom) -> int:
        return atom.GetTotalNumHs()


class TotalValence(OneHotFeaturizer):
    def call(self, atom: types.Atom) -> int:
        return atom.GetTotalValence()


class NumRadicalElectrons(OneHotFeaturizer):
    def call(self, atom: types.Atom) -> int:
        return atom.GetNumRadicalElectrons()


class Degree(OneHotFeaturizer):
    def call(self, atom: types.Atom) -> int:
        return atom.GetDegree()


class Aromatic(FloatFeaturizer):
    def call(self, atom: types.Atom) -> bool:
        return atom.GetIsAromatic()


class Hetero(FloatFeaturizer):
    def call(self, atom: types.Atom) -> bool:
        mol = atom.GetOwningMol()
        return atom.GetIdx() in [i[0] for i in Lipinski._Heteroatoms(mol)]


class HydrogenDonor(FloatFeaturizer):
    def call(self, atom: types.Atom) -> bool:
        mol = atom.GetOwningMol()
        return atom.GetIdx() in [i[0] for i in Lipinski._HDonors(mol)]


class HydrogenAcceptor(FloatFeaturizer):
    def call(self, atom: types.Atom) -> bool:
        mol = atom.GetOwningMol()
        return atom.GetIdx() in [i[0] for i in Lipinski._HAcceptors(mol)]


class RingSize(OneHotFeaturizer):
    def call(self, atom: types.Atom) -> int:
        size = 0
        if atom.IsInRing():
            while not atom.IsInRingSize(size):
                size += 1
        return size


class Ring(FloatFeaturizer):
    def call(self, atom: types.Atom) -> bool:
        return atom.IsInRing()


class CrippenLogPContribution(FloatFeaturizer):
    def call(self, atom: types.Atom) -> float:
        mol = atom.GetOwningMol()
        val = Crippen._GetAtomContribs(mol)[atom.GetIdx()][0]
        if val is not None and math.isfinite(val):
            return val
        return 0.0


class CrippenMolarRefractivityContribution(FloatFeaturizer):
    def call(self, atom: types.Atom) -> float:
        mol = atom.GetOwningMol()
        val = Crippen._GetAtomContribs(mol)[atom.GetIdx()][1]
        if val is not None and math.isfinite(val):
            return val
        return 0.0


class TPSAContribution(FloatFeaturizer):
    def call(self, atom: types.Atom) -> float:
        mol = atom.GetOwningMol()
        val = rdMolDescriptors._CalcTPSAContribs(mol)[atom.GetIdx()]
        if val is not None and math.isfinite(val):
            return val
        return 0.0


class LabuteASAContribution(FloatFeaturizer):
    def call(self, atom: types.Atom) -> float:
        mol = atom.GetOwningMol()
        val = rdMolDescriptors._CalcLabuteASAContribs(mol)[0][atom.GetIdx()]
        if val is not None and math.isfinite(val):
            return val
        return 0.0


class GasteigerCharge(FloatFeaturizer):
    def call(self, atom: types.Atom) -> float:
        mol = atom.GetOwningMol()
        rdPartialCharges.ComputeGasteigerCharges(mol)
        val = atom.GetDoubleProp("_GasteigerCharge")
        if val is not None and math.isfinite(val):
            return val
        return 0.0


class BondType(OneHotFeaturizer):
    def call(self, bond: types.Bond) -> str:
        return bond.GetBondType().name.lower()


class Stereo(OneHotFeaturizer):
    def call(self, bond: types.Bond) -> str:
        return bond.GetStereo().name.lower()


class Conjugated(FloatFeaturizer):
    def call(self, bond: types.Bond) -> bool:
        return bond.GetIsConjugated()


class Rotatable(FloatFeaturizer):
    def call(self, bond: types.Bond) -> bool:
        mol = bond.GetOwningMol()
        atom_indices = tuple(sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]))
        return atom_indices in Lipinski._RotatableBonds(mol)
