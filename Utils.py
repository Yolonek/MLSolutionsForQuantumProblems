import mplcyberpunk
import pickle
import jax
import netket as nk
import numpy as np
from itertools import product
from tqdm import tqdm
from flax.core import FrozenDict
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from netket.vqs import MCState
from Hamiltonians import Kitaev


def compute_energy_gap(hamiltonian):
    evals = nk.exact.lanczos_ed(hamiltonian, k=2, compute_eigenvectors=False)
    return evals[1] - evals[0]


def compute_kitaev_phase_diagram(kitaev_graph, kitaev_hi, J_scale):
    J_product = np.array(list(product(J_scale, repeat=3)))
    J_product /= np.sum(J_product, axis=1, keepdims=True)
    J_product = np.unique(J_product, axis=0)
    gaps = np.zeros(len(J_product))
    for i, J_vec in tqdm(enumerate(J_product)):
        hamiltonian = Kitaev(kitaev_hi, graph=kitaev_graph, J=J_vec)
        gaps[i] = compute_energy_gap(hamiltonian)
    return J_product, gaps


def enhance_plot(figure, axes, glow=False, alpha_gradient=0, lines=True, dpi=100):
    figure.set_facecolor('black')
    figure.set_dpi(dpi)
    axes.set_facecolor('black')
    for font in [axes.title, axes.xaxis.label, axes.yaxis.label]:
        font.set_fontweight('bold')
    if glow:
        if lines:
            mplcyberpunk.make_lines_glow(ax=axes)
        else:
            mplcyberpunk.make_scatter_glow(ax=axes)
    if 1 > alpha_gradient > 0:
        mplcyberpunk.add_gradient_fill(ax=axes, alpha_gradientglow=alpha_gradient)


def draw_kitaev_honeycomb(lattice, ax=None, node_color='powderblue', node_size=300,
                          figsize=None, curvature=0.2, font_size=12, font_color='k'):
    edge_colormap = dict(zip(lattice.edges(), lattice.edge_colors))
    edge_colors = {0: 'red', 1: 'blue', 2: 'green'}
    edge_labels = ['J_x', 'J_y', 'J_z']
    positions = lattice.positions
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize, layout='constrained')
    zorder, annotation = 1, None
    for edge in lattice.edges():
        x1, y1 = positions[edge[0]]
        x2, y2 = positions[edge[1]]
        arrowprops = dict(arrowstyle='-', shrinkA=0, shrinkB=0, patchA=None, patchB=None,
                          connectionstyle=f'arc3,rad={curvature}',
                          color=edge_colors[edge_colormap[edge]])
        annotation = ax.annotate('', xy=(x1, y1), xycoords='data',
                                 xytext=(x2, y2), textcoords='data',
                                 arrowprops=arrowprops)
    else:
        zorder = annotation.get_zorder() + 1
    ax.scatter(*positions.T, s=node_size, c=node_color,
               marker='o', zorder=zorder)
    for node in lattice.nodes():
        x1, y1 = positions[node]
        ax.text(x1, y1, str(node),
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=font_size,
                color=font_color,
                zorder=zorder)
    ax.set(xticks=[], yticks=[])
    legend_lines = [Patch(color=color, label=f'${label}$')
                    for color, label in zip(edge_colors.values(), edge_labels)]
    ax.legend(handles=legend_lines, loc='upper left')
    return ax


def plot_state_optimization(epochs, optimization_params, ax,
                            evals=None, colors=None,
                            title=None, **scatter_params):
    if evals is not None:
        ax.hlines(evals, 0, epochs, color='black', label='Exact states')
    optimization_iters, optimization_list = optimization_params
    for i, (iters, optimization) in enumerate(zip(optimization_iters, optimization_list)):
        ax.scatter(iters, optimization,
                   label=f'Ground state' if i == 0 else f'{i} excited state',
                   color=colors[i] if colors else None, **scatter_params)
    ax.set(xlabel='Iteration', ylabel='Energy', title=title)
    ax.legend()
    return ax


def plot_spectrum(evals, axes, title=None, linewidth=0.3, scatter=True, ylabel=True):
    axes.hlines(evals, 0, len(evals), color='blue', linewidth=linewidth)
    if scatter:
        axes.scatter(range(len(evals)), evals, color='black', s=1, zorder=3)
    if ylabel:
        axes.set(ylabel='$\epsilon$', title=title, xticks=[])
        axes.yaxis.label.set(rotation='horizontal', ha='right')


def save_variational_state_parameters(state_params, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(state_params, file)


def load_variational_state_parameters(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def save_variational_state(state, file_path):
    save_variational_state_parameters(FrozenDict(state.parameters), file_path)


def load_variational_state(file_path, sampler, machine, **state_params):
    state = MCState(sampler, machine, **state_params)
    state.init_parameters(jax.nn.initializers.normal(stddev=0.25))
    state.parameters = load_variational_state_parameters(file_path)
    return state

