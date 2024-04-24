import numpy as np
import pyarma as pa
from CommonFunctions import generate_basis_states
from OperatorFunctions import s_z, calculate_interaction


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