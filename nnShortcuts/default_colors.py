from nnShortcuts.nextnano_colors import NextnanoColor


class DefaultColors:
    bands = dict()
    bands_dark_background = dict()

    colormap = dict()
    lines_on_colormap = dict()

    current_voltage = ''
    light_voltage = ''
    current_under_illumination = ''
    responsivity = ''

    def __init__(self) -> None:
        self.bands = {
            'Gamma': 'tomato',
            'CB': 'tomato',
            'HH': 'royalblue',
            'LH': 'forestgreen',
            'SO': 'goldenrod',
            'kp6': 'blueviolet',
            'kp8': 'black'
        }
        # self.bands_dark_background = {
        #     'Gamma': 'darkgreen',
        #     'CB': 'darkgreen',
        #     'HH': 'lime', # 'cyan',
        #     'LH': 'lime',
        #     'SO': 'wheat',
        #     'kp6': 'magenta',
        #     'kp8': 'silver'
        # }
        self.bands_dark_background = {
            'Gamma': '#999999',  # (NextnanoColor.r_in_nnblue, NextnanoColor.g_in_nnblue, NextnanoColor.b_in_nnblue),
            'CB': '#999999',  # (NextnanoColor.r_in_nnblue, NextnanoColor.g_in_nnblue, NextnanoColor.b_in_nnblue),
            'HH': '#999999',
            'LH': '#999999',
            'SO': 'wheat',
            'kp6': 'magenta',
            'kp8': 'silver'
        }
        self.bandgap_fill = '#333333'  # 'grey' in the scale from 0 to f

        self.colormap['linear_bright_bg'] = NextnanoColor.cmap['linear']  # 'cividis'
        self.lines_on_colormap['bright_bg'] = ['black', 'orange']

        self.colormap['linear_dark_bg'] = NextnanoColor.cmap['linear'].reversed()  # 'cividis'
        self.lines_on_colormap['dark_bg'] = ['white', 'yellow']


        # TODO: load the CSV file at root of this repo and implement 'Fast' color map, which is divergent and Color Vision Deficiency (https://www.kennethmoreland.com/color-advice/)
        self.colormap['divergent_bright'] = 'bwr'  # 'seismic'

        self.colormap['divergent_dark'] = NextnanoColor.cmap['divergent_dark'].reversed()

        self.colormap['sequential'] = 'Greens'

        self.current_voltage = 'tab:blue'
        self.light_voltage = 'tab:red'
        self.current_under_illumination = 'tab:orange'
        self.responsivity = 'tab:purple'
