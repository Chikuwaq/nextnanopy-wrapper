import numpy as np
import matplotlib.colors as pltc

"""
Description: Colormap based on nextnano corporate color.
"""

__author__ = "Yuta Goto, and Takuma Sato"
__copyright__ = "Copyright 2024, nextnano GmbH"


class NextnanoColor:
    cmap = dict()

    # --- color code ---
    # nn-blue
    r_in_nnblue = 19 / 256
    g_in_nnblue = 173 / 256
    b_in_nnblue = 181 / 256

    # nn-blue-dark
    r_in_nnblue_dark = r_in_nnblue / 2
    g_in_nnblue_dark = g_in_nnblue / 2
    b_in_nnblue_dark = b_in_nnblue / 2

    # golden
    r_in_golden = 255 / 256
    g_in_golden = 215 / 256
    b_in_golden = 0 / 256

    # violet
    r_in_violet = 212 / 256
    g_in_violet = 0 / 256
    b_in_violet = 85 / 256

    # violet-dark
    r_in_violet_dark = r_in_violet / 2
    g_in_violet_dark = g_in_violet / 2
    b_in_violet_dark = b_in_violet / 2

    # black
    r_in_black = 0
    g_in_black = 0
    b_in_black = 0

    # white
    r_in_white = 1
    g_in_white = 1
    b_in_white = 1

    N = 256

    # --- dark divergent map ---
    colors = np.ones((N * 4, 3))

    colors[:N, 0] = np.linspace(r_in_white, r_in_nnblue, N)
    colors[:N, 1] = np.linspace(g_in_white, g_in_nnblue, N)
    colors[:N, 2] = np.linspace(b_in_white, b_in_nnblue, N)

    colors[N:2 * N, 0] = np.linspace(r_in_nnblue, r_in_black, N)
    colors[N:2 * N, 1] = np.linspace(g_in_nnblue, g_in_black, N)
    colors[N:2 * N, 2] = np.linspace(b_in_nnblue, b_in_black, N)

    colors[2 * N:3 * N, 0] = np.linspace(r_in_black, r_in_violet, N)
    colors[2 * N:3 * N, 1] = np.linspace(g_in_black, g_in_violet, N)
    colors[2 * N:3 * N, 2] = np.linspace(b_in_black, b_in_violet, N)

    colors[3 * N:, 0] = np.linspace(r_in_violet, r_in_white, N)
    colors[3 * N:, 1] = np.linspace(g_in_violet, g_in_white, N)
    colors[3 * N:, 2] = np.linspace(b_in_violet, b_in_white, N)

    cmap['divergent_dark'] = pltc.ListedColormap(colors)

    # --- bright divergent map ---
    colors = np.ones((N * 2, 3))

    colors[:N, 0] = np.linspace(r_in_nnblue, r_in_white, N)
    colors[:N, 1] = np.linspace(g_in_nnblue, g_in_white, N)
    colors[:N, 2] = np.linspace(b_in_nnblue, b_in_white, N)

    colors[N:2 * N, 0] = np.linspace(r_in_white, r_in_violet, N)
    colors[N:2 * N, 1] = np.linspace(g_in_white, g_in_violet, N)
    colors[N:2 * N, 2] = np.linspace(b_in_white, b_in_violet, N)

    cmap['divergent_bright'] = pltc.ListedColormap(colors)

    # --- bright divergent map 2 ---
    colors = np.ones((N * 4, 3))

    colors[:N, 0] = np.linspace(r_in_nnblue_dark, r_in_nnblue, N)
    colors[:N, 1] = np.linspace(g_in_nnblue_dark, g_in_nnblue, N)
    colors[:N, 2] = np.linspace(b_in_nnblue_dark, b_in_nnblue, N)

    colors[N:2 * N, 0] = np.linspace(r_in_nnblue, r_in_white, N)
    colors[N:2 * N, 1] = np.linspace(g_in_nnblue, g_in_white, N)
    colors[N:2 * N, 2] = np.linspace(b_in_nnblue, b_in_white, N)

    colors[2 * N:3 * N, 0] = np.linspace(r_in_white, r_in_violet, N)
    colors[2 * N:3 * N, 1] = np.linspace(g_in_white, g_in_violet, N)
    colors[2 * N:3 * N, 2] = np.linspace(b_in_white, b_in_violet, N)

    colors[3 * N:, 0] = np.linspace(r_in_violet, r_in_violet_dark, N)
    colors[3 * N:, 1] = np.linspace(g_in_violet, g_in_violet_dark, N)
    colors[3 * N:, 2] = np.linspace(b_in_violet, b_in_violet_dark, N)

    cmap['divergent_bright2'] = pltc.ListedColormap(colors)

    # --- linear map ---
    colors = np.ones((N * 2, 3))

    colors[:N, 0] = np.linspace(r_in_white, r_in_nnblue, N)
    colors[:N, 1] = np.linspace(g_in_white, g_in_nnblue, N)
    colors[:N, 2] = np.linspace(b_in_white, b_in_nnblue, N)

    colors[N:2 * N, 0] = np.linspace(r_in_nnblue, r_in_black, N)
    colors[N:2 * N, 1] = np.linspace(g_in_nnblue, g_in_black, N)
    colors[N:2 * N, 2] = np.linspace(b_in_nnblue, b_in_black, N)

    cmap['linear'] = pltc.ListedColormap(colors)

    @staticmethod
    def get_nextnano_blue(dark_mode : bool):
        if dark_mode:
            return (NextnanoColor.r_in_nnblue_dark, NextnanoColor.g_in_nnblue_dark, NextnanoColor.b_in_nnblue_dark)
        else:
            return (NextnanoColor.r_in_nnblue, NextnanoColor.g_in_nnblue, NextnanoColor.b_in_nnblue)
