import jax
import jax.numpy as jnp
import netket.jax as nkjax
from netket.stats import statistics
from netket.vqs.mc import get_local_kernel, get_local_kernel_arguments
from netket.utils import mpi
from functools import partial


def penalty_kernel(logpsi, pars1, pars2, sigma2):
    return jnp.exp(logpsi(pars1, sigma2) - logpsi(pars2, sigma2))


def penalty_based_expect_and_grad(
        vstate,
        operator,
        state_list,
        penalty_list,
        is_mutable=False
):
    sigma, sigma_args = get_local_kernel_arguments(vstate, operator)
    local_estimator_function = get_local_kernel(vstate, operator)

    sigma_list = []
    model_state_list = []
    parameters_list = []
    for state in state_list:
        sigma_list.append(state.samples)
        model_state_list.append(state.model_state)
        parameters_list.append(state.parameters)

    O, O_grad, new_model_state = penalty_based_grad_expect_hermitian(
        local_estimator_function,
        penalty_kernel,
        vstate._apply_fun,
        sigma,
        vstate.parameters,
        vstate.model_state,
        sigma_args,
        sigma_list,
        model_state_list,
        parameters_list,
        penalty_list,
        is_mutable=is_mutable
    )

    if is_mutable:
        vstate.model_state = new_model_state

    return O, O_grad


@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def penalty_based_grad_expect_hermitian(
        local_value_kernel,
        penalty_kernel,
        model_apply_function,
        sigma,
        parameters,
        model_state,
        local_value_args,
        sigma_list,
        model_state_list,
        parameters_list,
        penalty_list,
        is_mutable=False
):
    sigma = sigma.reshape((-1, sigma.shape[-1]))
    n_samples = sigma.shape[0] * mpi.n_nodes

    O_loc = local_value_kernel(
        model_apply_function,
        {'params': parameters, **model_state},
        sigma,
        local_value_args
    )

    E_loc = O_loc
    O = statistics(O_loc.reshape(sigma.shape[:-1]).T)
    O_loc -= O.mean

    _, vjp_function, *new_model_state = nkjax.vjp(
        lambda x: model_apply_function({'params': x, **model_state}, sigma, mutable=is_mutable),
        parameters,
        conjugate=True,
        has_aux=is_mutable
    )

    for penalty, sigma_i, params, previous_state in zip(
            penalty_list, sigma_list, parameters_list, model_state_list):
        if jnp.ndim(sigma_i) != 2:
            sigma_i = sigma_i.reshape((-1, sigma_i.shape[-1]))

        psi_loc_1 = penalty_kernel(
            model_apply_function,
            {'params': parameters, **model_state},
            {'params': params, **previous_state},
            sigma_i
        )
        psi_1 = statistics(psi_loc_1.reshape(sigma_i.shape[:-1]).T)

        psi_loc_2 = penalty_kernel(
            model_apply_function,
            {'params': params, **previous_state},
            {'params': parameters, **model_state},
            sigma
        )
        psi_2 = statistics(psi_loc_2.reshape(sigma_i.shape[:-1]).T)

        psi_loc_2 -= psi_2.mean
        psi_loc_2 *= penalty * psi_1.mean
        O_loc += psi_loc_2

        O_grad = vjp_function(jnp.conjugate(O_loc) / n_samples)[0]
        O_grad = jax.tree_map(
            lambda x, target: (x if jnp.iscomplexobj(target) else x.real).astype(target.dtype),
            O_grad, parameters
        )

        E = statistics(E_loc.reshape(sigma_i.shape[:-1]).T)
        new_model_state = new_model_state[0] if is_mutable else None

        return E, jax.tree_map(lambda x: mpi.mpi_sum_jax(x)[0], O_grad), new_model_state


