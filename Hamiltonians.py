import numpy as np
import numba as nb
import pyarma as pa
import jax.numpy as jnp
import scipy.sparse as sp
from pympler import asizeof
from time import time
from scipy.ndimage import convolve, generate_binary_structure
from collections.abc import Sequence
from typing import Generator, Optional
from collections import defaultdict
from Hilbert import generate_basis_states
from Operator import s_z, calculate_interaction
import netket as nk
from netket.operator import GraphOperator, LocalOperator
from netket.operator.spin import sigmax, sigmay, sigmaz
from netket.hilbert import AbstractHilbert
from netket.graph import KitaevHoneycomb
from netket.utils.types import DType
from Bitarray import (integer_to_bitarray,
                      bitarray_to_integer, flip,
                      count_ones_and_zeros_difference,
                      shift_bits_of_integer)


class Kitaev(GraphOperator):
    def __init__(
            self,
            hilbert: AbstractHilbert,
            kitaev: KitaevHoneycomb,
            J: Sequence[float, float, float] = (1., 1., 1.),
            add_exchange: Optional[bool] = None,
            dtype: Optional[DType] = None,
    ):

        Jx, Jy, Jz = J
        self._Ji = J

        sx_sx = np.array([[0, 0, 0, 1],
                          [0, 0, 1, 0],
                          [0, 1, 0, 0],
                          [1, 0, 0, 0]])

        sy_sy = np.array([[0, 0, 0, -1],
                          [0, 0, 1, 0],
                          [0, 1, 0, 0],
                          [-1, 0, 0, 0]])

        sz_sz = np.array([[1, 0, 0, 0],
                          [0, -1, 0, 0],
                          [0, 0, -1, 0],
                          [0, 0, 0, 1]])

        exchange = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 2, 0],
                [0, 2, 0, 0],
                [0, 0, 0, 0],
            ]
        )

        bond_ops = [-Jx * (sx_sx + exchange if add_exchange else sx_sx),
                    -Jy * (sy_sy + exchange if add_exchange else sy_sy),
                    -Jz * (sz_sz + exchange if add_exchange else sz_sz)]
        # bond_ops = [-Jx * sx_sx, -Jy * sy_sy, -Jz * sz_sz]
        bond_ops_colors = [0, 1, 2]

        super().__init__(
            hilbert,
            kitaev,
            bond_ops=bond_ops,
            bond_ops_colors=bond_ops_colors,
            dtype=dtype
        )

    @property
    def Jx(self):
        return self._Ji[0]

    @property
    def Jy(self):
        return self._Ji[1]

    @property
    def Jz(self):
        return self._Ji[2]

    def __repr__(self):
        return (f'Kitaev(Jx={self._Ji[0]}, Jy={self._Ji[1]}, Jz={self._Ji[2]}, '
                f'dim={self.hilbert.size}, '
                f'#acting on={self.graph.n_edges} locations)')


class Sandvik:
    def __init__(self, L: int, J: float, delta: float, is_pbc: bool = False):
        self.L: int = L
        self.J: float = J
        self.delta: float = delta
        self.pbc: bool = is_pbc
        self.system_size: int = 2 ** L
        self.magnetization = None
        self.H_dict: defaultdict[tuple[int, int], float] = defaultdict(float)

    def reset_data(self):
        self.H_dict: defaultdict[tuple[int, int], float] = defaultdict(float)
        self.magnetization = None

    def choose_basis(self, magnetization: int = None):
        if magnetization is None:
            return range(self.system_size)
        else:
            return self.magnetization_basis(magnetization)

    def construct_hamiltonian(self,
                              magnetization: int = None,
                              convert_indices: bool = False):
        basis = self.choose_basis(magnetization)
        self.magnetization = magnetization
        for a in basis:
            a_bin = integer_to_bitarray(a, self.L)
            for i in range(self.L):
                j = i + 1
                if self.pbc:
                    j = (i + 1) % self.L
                else:
                    if j == self.L:
                        break
                if a_bin[i] == a_bin[j]:
                    self.H_dict[(a, a)] += 1
                else:
                    self.H_dict[(a, a)] -= 1
                    b = bitarray_to_integer(flip(a_bin, [i, j]))
                    self.H_dict[(a, b)] = self.J / 2
            self.H_dict[(a, a)] *= self.delta * self.J / 4
        if convert_indices:
            self.convert_matrix_coordinates_to_block()

    def magnetization_basis(self, magnetization: int) -> Generator:
        for state in range(self.system_size):
            if count_ones_and_zeros_difference(state, self.L) == magnetization:
                yield state

    def count_number_of_magnetization_block_states(self, magnetization: int) -> int:
        return len(list(self.magnetization_basis(magnetization)))

    def convert_matrix_coordinates_to_block(self):
        basis = list(self.choose_basis(self.magnetization))
        new_H_dict = defaultdict(float)
        for (i, j), value in self.H_dict.items():
            i_m = basis.index(i)
            if i == j:
                new_H_dict[(i_m, i_m)] = value
            else:
                new_H_dict[(i_m, basis.index(j))] = value
        self.H_dict = new_H_dict

    def get_hamiltonian_as_sparse_matrix(self):
        rows, cols, values = zip(*[(row, col, value) for (row, col), value in self.H_dict.items()])
        n_rows, n_cols = max(rows) + 1, max(cols) + 1
        return sp.coo_matrix((values, (rows, cols)), shape=(n_rows, n_cols)).tocsr()

    def get_hamiltonian_as_dense_matrix(self) -> np.ndarray:
        if self.magnetization is not None:
            basis_length = len(list(self.choose_basis(self.magnetization)))
            hamiltonian = np.zeros((basis_length, basis_length))
        else:
            hamiltonian = np.zeros((self.system_size, self.system_size))
        for (i, j), value in self.H_dict.items():
            hamiltonian[i, j] = value
        return hamiltonian

    def print_matrix(self):
        print(f'System size: {self.system_size} x {self.system_size}')
        print('Basis:')
        print(end=(9 - self.L) * ' ')
        for basis_state in generate_basis_states(
                0, self.system_size, self.L,
                block=False if self.magnetization is None else True,
                total_spin=self.magnetization / 2 if self.magnetization is not None else 0):
            print(basis_state, end=(9 - self.L) * ' ')
        print('\n')
        #     pa.mat(self.get_hamiltonian_as_dense_matrix()[m_basis][:, m_basis]).print()
        pa.mat(self.get_hamiltonian_as_dense_matrix()).print()


class MomentumStatesCalculator:

    def __init__(self, N: int, k: int = 0):
        self.k: int = k
        self.N: int = N

    def check_state_for_representatives(self, state: int) -> int:
        representative: int = -1
        temp_representative: int = state
        for i in range(1, self.N + 1):
            temp_representative = shift_bits_of_integer(temp_representative, self.N, -1)
            if temp_representative < state:
                continue
            elif temp_representative == state:
                if self.k % (self.N / i) != 0:
                    continue
                representative = i
        return representative

    def find_smallest_representative(self, state: int) -> tuple[int, int]:
        representative: int = state
        temp_representative: int = state
        num_of_translations: int = 0
        for i in range(1, self.N):
            temp_representative = shift_bits_of_integer(temp_representative, self.N, -1)
            if temp_representative < representative:
                representative = temp_representative
                num_of_translations = i
        return representative, num_of_translations


class Hamiltonian:

    def __init__(self, L: int, J: float, delta: float, is_pbc=False):
        self.L: int = L
        self.J: float = J
        self.delta: float = delta

        self.system_size: int = 2 ** L
        self.matrix: np.array = np.zeros((self.system_size, self.system_size))

        self.pbc: bool = is_pbc
        self.is_block_concatenated: bool = False
        self.block_value = None

    def model_parameters(self):
        params = {'L': self.L,
                  'J': self.J,
                  'delta': self.delta,
                  'size': self.system_size,
                  'basis': self.get_basis(convert_to_list=True),
                  'pbc': self.pbc,
                  'block_value': self.block_value}
        return params

    def get_basis(self, convert_to_list: bool = False):
        basis = generate_basis_states(0, 2 ** self.L, self.L,
                                      block=self.is_block_concatenated,
                                      total_spin=self.block_value)
        if convert_to_list:
            return list(basis)
        else:
            return basis

    def print_hamiltonian_data(self, return_msg=False):
        msg = f'L = {self.L}, J = {self.J}, delta = {self.delta}, ' \
              f'hamiltonian size: {self.system_size} x {self.system_size}'
        print(msg)
        if return_msg:
            return msg

    def print_basis(self):
        for basis_state in self.get_basis():
            print(f'|{basis_state}>')

    def print_matrix(self):
        print(f'Matrix size: {self.system_size} x {self.system_size}')
        if self.is_block_concatenated:
            print(f'Basis reduced to spin = {self.block_value}:')
        else:
            print('Basis:')
        print(end=(9 - self.L) * ' ')
        for basis_state in self.get_basis():
            print(basis_state, end=(9 - self.L) * ' ')
        print('\n')
        pa.mat(self.matrix).print()

    def plot_data(self, matrix=None, axes=None):
        if axes:
            axes.axis('off')

            matrix = np.round(np.array(matrix if matrix else self.matrix), 4)

            try:
                cell_height = 1 / matrix.shape[0]
                cell_width = 1 / matrix.shape[1]
            except IndexError:
                matrix = matrix[:, None]
                cell_height = 1 / matrix.shape[0]
                cell_width = 1 / matrix.shape[1]

            table = axes.table(cellText=matrix, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)

            for cell in table._cells.values():
                cell.set_edgecolor('none')
                cell.set_linewidth(0)
                cell.set_height(cell_height)
                cell.set_width(cell_width)

    def reset_hamiltonian(self):
        self.matrix = np.zeros((self.system_size, self.system_size))

    def block_concatenation(self, total_spin: int):
        self.is_block_concatenated = True
        self.block_value = total_spin
        self.system_size = len(self.get_basis(convert_to_list=True))
        self.matrix = np.zeros((self.system_size, self.system_size))

    def _hamiltonian_p_element(self, spin_configuration: str):
        element_sum = 0
        for i in range(self.L - 1):
            element_sum += s_z(i, spin_configuration) * s_z(i + 1, spin_configuration)
        if self.pbc:
            element_sum += s_z(-1, spin_configuration) * s_z(0, spin_configuration)
        return element_sum * self.J * self.delta

    def _prepare_hamiltonian_p(self):
        hamiltonian_p = np.zeros(self.system_size)
        for i, spin_configuration in enumerate(self.get_basis()):
            hamiltonian_p[i] = self._hamiltonian_p_element(spin_configuration)
        self.matrix += np.diag(hamiltonian_p)

    def _prepare_hamiltonian_k(self):
        hamiltonian_k = np.zeros((self.system_size, self.system_size))
        basis = self.get_basis(convert_to_list=True)
        for j, spin_configuration in enumerate(basis):
            combinations = calculate_interaction(spin_configuration, self.pbc)
            for result in combinations:
                if result != '':
                    i = basis.index(result)
                    hamiltonian_k[i, j] = self.J / 2
        self.matrix += hamiltonian_k

    def prepare_hamiltonian(self):
        self.reset_hamiltonian()
        self._prepare_hamiltonian_p()
        self._prepare_hamiltonian_k()


def KitaevHamiltonian(hilbert: nk.hilbert.Spin,
                      kitaev: KitaevHoneycomb,
                      J: Sequence[float, float, float]
                      ) -> LocalOperator:
    Jx, Jy, Jz = J
    Ji = {0: Jx, 1: Jy, 2: Jz}
    operators = {0: sigmax, 1: sigmay, 2: sigmaz}
    hamiltonian = LocalOperator(hilbert, dtype=jnp.complex128)
    for (i, j), color in zip(kitaev.edges(), kitaev.edge_colors):
        operator = operators[color]
        hamiltonian += -Ji[color] * (operator(hilbert, i) @ operator(hilbert, j))
    return hamiltonian


class IsingGrid:
    def __init__(self, L, J):
        self.L = L
        self.J = J
        self.lattice = None
        self.energy = 0

    def set_lattice(self, lattice):
        self.lattice = lattice.astype(np.int8)

    def initialize_grid(self, probability=0.5):
        probability_grid = np.random.random((self.L, self.L))
        self.lattice = np.where(probability_grid >= probability, 1, -1).astype(np.int8)

    def get_system_energy(self):
        kernel = generate_binary_structure(2, 1)
        kernel[1][1] = False
        convolution = -self.lattice * convolve(self.lattice, kernel, mode='constant', cval=0)
        self.energy = (convolution_sum := convolution.sum())
        return convolution_sum

    def metropolis(self, timestamp, T, probability=0.5, initialize=True):
        if initialize:
            self.initialize_grid(probability=probability)
        self.energy = self.get_system_energy()
        self.lattice, spin_sum, system_energy = metropolis_ising(
            self.lattice, timestamp, self.J, T, self.energy)
        return self.lattice, spin_sum, system_energy


@nb.njit()
def metropolis_ising(grid, timestamp, J, T, energy):
    N = grid.shape[0]
    spin_grid = grid.copy()
    spin_sum = np.zeros(timestamp)
    system_energy = np.zeros(timestamp)
    for t in range(timestamp):
        x, y = np.random.randint(0, high=N, size=2)
        spin_init = spin_grid[x, y]
        spin_flip = -spin_init
        E_init, E_flip = 0, 0
        if x > 0:
            E_init += -spin_init * spin_grid[x - 1, y]
            E_flip += -spin_flip * spin_grid[x - 1, y]
        if x < N - 1:
            E_init += -spin_init * spin_grid[x + 1, y]
            E_flip += -spin_flip * spin_grid[x + 1, y]
        if y > 0:
            E_init += -spin_init * spin_grid[x, y - 1]
            E_flip += -spin_flip * spin_grid[x, y - 1]
        if y < N - 1:
            E_init += -spin_init * spin_grid[x, y + 1]
            E_flip += -spin_flip * spin_grid[x, y + 1]
        dE = E_flip - E_init
        if dE > 0 and np.random.rand() < np.exp(-(J * dE) / T):
            spin_grid[x, y] = spin_flip
            energy += dE
        elif dE <= 0:
            spin_grid[x, y] = spin_flip
            energy += dE

        spin_sum[t] = spin_grid.sum()
        system_energy[t] = energy
    return spin_grid, spin_sum, system_energy


if __name__ == '__main__':
    L, J, delta = 6, 1, 1
    pbc = False
    m = 0
    k = 2

    # time_start = time()
    # # up to L = 14
    # h = Hamiltonian(L=L, J=J, delta=delta, is_pbc=pbc, s)
    # h.prepare_hamiltonian()
    # h.print_matrix()
    # time_end = time()
    # print(f'Time: {time_end - time_start:0.4f} seconds')
    # print(f'{asizeof.asizeof(h.matrix) / 1_000_000:0.4f} MB')

    time_start = time()
    # # up to L = 19 (9.5 s, 754 MB), compared with older one L = 14
    # #       L = 20 (21.3 s, 1568 MB)
    # # for m=0, L=20 (4.3 s, 311 MB)
    # #          L=22 (18.2 s, 1283 MB)
    hs = Sandvik(L=L, J=J, delta=delta, is_pbc=pbc)
    lst = list(hs.choose_basis(m, k))
    print(lst, len(lst))
    # hs.construct_hamiltonian(magnetization=m)
    # hs.print_matrix()
    time_end = time()
    print(f'Time: {time_end - time_start:0.4f} seconds')
    print(f'{asizeof.asizeof(hs.H_dict) / 1_000_000:0.4f} MB')
