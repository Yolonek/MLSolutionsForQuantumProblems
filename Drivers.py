import jax
import jax.numpy as jnp
from netket.driver.abstract_variational_driver import AbstractVariationalDriver
from netket.utils.types import PyTree
from netket.stats import Stats
from Gradients import penalty_based_expect_and_grad


class PenaltyBasedVMC(AbstractVariationalDriver):

    def __init__(
        self,
        hamiltonian,
        optimizer,
        variational_state,
        preconditioner,
        state_list,
        penalty_list
    ):

        super().__init__(variational_state, optimizer, minimized_quantity_name='Energy')

        self._ham = hamiltonian.collect()
        self.preconditioner = preconditioner

        self._dp = None
        self._S = None
        self._sr_info = None
        self._state_list = state_list
        self._penalty_list = penalty_list

    def _forward_and_backward(self) -> PyTree:
        self.state.reset()
        for state in self._state_list:
            state.reset()

        self._loss_stats, self._loss_grad = penalty_based_expect_and_grad(
            self.state, self._ham, self._state_list,
            self._penalty_list, is_mutable=self.state.mutable
        )

        self._dp = self.preconditioner(self.state, self._loss_grad)
        self._dp = jax.tree_map(
            lambda x, target: (x if jnp.iscomplexobj(target) else x.real),
            self._dp,
            self.state.parameters
        )
        return self._dp

    @property
    def energy(self) -> Stats:
        return self._loss_stats

    def __repr__(self):
        return (
            "PenaltyBasedVMC("
            + f"\n  step_count = {self.step_count},"
            + f"\n  state = {self.state})"
        )




