import numpy as np
import numba as nb
import pyarma as pa
from pympler import asizeof
from time import time
from math import factorial
from typing import Union, List, Generator
from collections import defaultdict
from CommonFunctions import generate_basis_states
from OperatorFunctions import s_z, calculate_interaction
from BitFunctions import (integer_to_bitarray,
                          integer_to_bitstring,
                          bitarray_to_integer, flip,
                          count_ones_and_zeros_difference,
                          count_number_of_ones,
                          shift_bits_of_integer)


class HamiltonianSandvik:

    def __init__(self, L: int, J: float, delta: float, is_pbc: bool = False):
        self.L: int = L
        self.J: float = J
        self.delta: float = delta
        self.pbc: bool = is_pbc
        self.system_size: int = 2 ** L
        self.magnetization = None
        self.momentum = None
        self.H_dict: defaultdict[tuple[int, int], float] = defaultdict(float)

    def reset_data(self):
        self.H_dict: defaultdict[tuple[int, int], float] = defaultdict(float)
        self.magnetization = None

    def choose_basis(self, magnetization: int = None, momentum: int = None):
        if magnetization is None and momentum is None:
            return range(self.system_size)
        elif magnetization is not None and momentum is None:
            return self.magnetization_basis(magnetization)
        elif magnetization is None and momentum is not None:
            return self.momentum_basis(momentum)
        else:
            return self.momentum_basis(momentum, magnetization=magnetization)

    def construct_hamiltonian(self, magnetization: int = None, momentum: int = None):
        basis = self.choose_basis(magnetization, momentum)
        self.magnetization = magnetization
        self.momentum = momentum
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
                    if self.momentum is not None:
                        pass
                    else:
                        b = bitarray_to_integer(flip(a_bin, [i, j]))
                        self.H_dict[(a, b)] = self.J / 2
            self.H_dict[(a, a)] *= self.delta * self.J / 4

    def magnetization_basis(self, magnetization: int) -> Generator:
        for state in range(self.system_size):
            if count_ones_and_zeros_difference(state, self.L) == magnetization:
                yield state

    def count_number_of_magnetization_block_states(self, magnetization: int) -> int:
        return len(list(self.magnetization_basis(magnetization)))

    def momentum_basis(self, momentum: int, magnetization: int = 0, with_r: bool = False) -> Generator:
        for state in self.magnetization_basis(magnetization):
            representative = MomentumStatesCalculator(self.L, momentum).check_state_for_representatives(state)
            if representative >= 0:
                if with_r:
                    yield state, representative
                else:
                    yield state

    def count_number_of_momentum_block_states(self, k: int) -> int:
        return len(list(self.momentum_basis(k)))

    def convert_matrix_coordinates_to_block(self):
        basis = self.choose_basis(self.magnetization, self.momentum)
        new_H_dict = defaultdict(float)
        for (i, j), value in self.H_dict.items():
            i_m = basis.index(i)
            if i == j:
                new_H_dict[(i_m, i_m)] = value
            else:
                new_H_dict[(i_m, basis.index(j))] = value
        self.H_dict = new_H_dict

    def get_hamiltonian_as_dense_matrix(self) -> np.ndarray:
        if self.magnetization is not None or self.momentum is not None:
            basis_length = len(list(self.choose_basis(self.magnetization, self.momentum)))
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
        for basis_state in generate_basis_states(0, self.system_size, self.L,
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
    hs = HamiltonianSandvik(L=L, J=J, delta=delta, is_pbc=pbc)
    lst = list(hs.choose_basis(m, k))
    print(lst, len(lst))
    # hs.construct_hamiltonian(magnetization=m)
    # hs.print_matrix()
    time_end = time()
    print(f'Time: {time_end - time_start:0.4f} seconds')
    print(f'{asizeof.asizeof(hs.H_dict) / 1_000_000:0.4f} MB')

