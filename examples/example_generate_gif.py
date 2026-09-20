
import matplotlib.pyplot as plt

from nnShortcuts.common import CommonShortcuts

shortcut = CommonShortcuts()

sweep_values = [0.5, 1.0, 1.5, 2.0]
xs = list(range(10))

def plot_fig(sweep_val):
    fig, ax = plt.subplots()
    ys = [x * sweep_val for x in xs]
    ax.set_ylim(0, max(xs)*max(sweep_values))
    ax.plot(xs, ys)
    ax.set_title(f"sweep = {sweep_val}")


shortcut.generate_gif(
    plot_fig,
    sweep_values,
    "/Users/tsato/nextnano_Mac/example.gif",
)
