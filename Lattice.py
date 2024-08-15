from netket.graph import Lattice
from typing import Union, Sequence
import numpy as np


def Rectangular(extent, *,
                pbc: Union[bool, Sequence[bool]] = True, **kwargs) -> Lattice:
    basis = np.array([[1, 0], [0, 1]])
    return Lattice(basis_vectors=basis,
                   extent=extent,
                   pbc=pbc,
                   **kwargs)
