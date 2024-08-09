import mplcyberpunk
from matplotlib import pyplot as plt
from matplotlib.patches import Patch


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