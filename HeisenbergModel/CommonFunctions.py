from OperatorFunctions import s_total
from BitFunctions import integer_to_bitstring
import mplcyberpunk


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


def generate_basis_states(start: int, stop: int, length: int,
                          block: bool = False, total_spin: int = 0):
    count = start
    while count < stop:
        binary_string = integer_to_bitstring(count, length)
        if block:
            if s_total(binary_string) == total_spin:
                yield binary_string
        else:
            yield binary_string
        count += 1
