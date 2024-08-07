import numpy as np
import matplotlib.pyplot as plt
import unittest

from nnShortcuts.nextnano_colors import NextnanoColor

gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))


def plot_color_gradients(category, cmap_list):
	# Create figure and adjust figure height to number of colormaps
	nrows = len(cmap_list)
	figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
	fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
	fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
						left=0.2, right=0.99)
	axs[0].set_title(f'{category} colormaps', fontsize=14)

	for ax, name in zip(axs, cmap_list):
		ax.imshow(gradient, aspect='auto', cmap=NextnanoColor.cmap[name])
		ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=10,
				transform=ax.transAxes)

	# Turn off *all* ticks & spines, not just the ones with colormaps.
	for ax in axs:
		ax.set_axis_off()


class Test_nextnano_color(unittest.TestCase):
	plot_color_gradients('nextnano colormaps', ['divergent_dark', 'divergent_bright', 'linear'])
	plt.show()


if __name__ == '__main__':
	unittest.main()