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

    def __init__(self, band_names) -> None:
        self.bands = {
            band_names['Gamma']: 'tomato',
            band_names['CB']: 'tomato',
            band_names['HH']: 'royalblue',
            band_names['LH']: 'forestgreen',
            band_names['SO']: 'goldenrod',
            band_names['kp6']: 'blueviolet',
            band_names['kp8']: 'black'
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
            band_names['Gamma']: '#999999',  # (NextnanoColor.r_in_nnblue, NextnanoColor.g_in_nnblue, NextnanoColor.b_in_nnblue),
            band_names['CB']: '#999999',  # (NextnanoColor.r_in_nnblue, NextnanoColor.g_in_nnblue, NextnanoColor.b_in_nnblue),
            band_names['HH']: '#999999',
            band_names['LH']: '#999999',
            band_names['SO']: 'wheat',
            band_names['kp6']: 'magenta',
            band_names['kp8']: 'silver'
        }
        self.bandgap_fill = '#333333'  # 'grey' in the scale from 0 to f

        self.colormap['linear_bright_bg'] = NextnanoColor.cmap['linear']  # 'cividis'
        self.lines_on_colormap['bright_bg'] = ['black', 'orange']

        self.colormap['linear_dark_bg'] = NextnanoColor.cmap['linear'].reversed()  # 'cividis'
        self.lines_on_colormap['dark_bg'] = ['white', 'yellow']

        # cividis runs from dark blue‑black to bright yellow‑white and is perceptually uniform, 
        # so a high‑luminance line stands out across the full range 
        # while remaining legible in grayscale and for color‑blind viewers.
        self.lines_on_colormap['cividis'] = [(0.95, 0.95, 0.95), 'black'] 


        # TODO: load the CSV file at root of this repo and implement 'Fast' color map, which is divergent and Color Vision Deficiency (https://www.kennethmoreland.com/color-advice/)
        self.colormap['divergent_bright'] = 'bwr'  # 'seismic'

        self.colormap['divergent_dark'] = NextnanoColor.cmap['divergent_dark'].reversed()

        self.colormap['sequential'] = 'Greens'

        self.current_voltage = 'tab:blue'
        self.light_voltage = 'tab:red'
        self.current_under_illumination = 'tab:orange'
        self.responsivity = 'tab:purple'

    def get_lines_on_colormap(self, dark_mode):
        """
        Return default color depending whether dark mode output is desired.
        
        Returns
        -------
            str
        """
        if not isinstance(dark_mode, bool):
            raise ValueError(f"'dark_mode' must be a bool, not {type(dark_mode)}")
        if dark_mode:
            return self.lines_on_colormap['dark_bg']
        else:
            return self.lines_on_colormap['bright_bg']

    def get_colormap(self, is_divergent, dark_mode):
        """
        Return default color depending whether the data is divergent, and whether dark mode output is desired.
        
        Returns
        -------
            str
        """
        if not isinstance(is_divergent, bool):
            raise ValueError(f"'dark_mode' must be a bool, not {type(is_divergent)}")
        if not isinstance(dark_mode, bool):
            raise ValueError(f"'dark_mode' must be a bool, not {type(dark_mode)}")
        
        if is_divergent:
            if dark_mode:
                return self.colormap['divergent_dark']
            else:
                return self.colormap['divergent_bright']
        else:
            if dark_mode:
                return self.colormap['linear_dark_bg']
            else:
                return self.colormap['linear_bright_bg']
            

    def get_linecolor_bandedges(self, CVD_aware, dark_mode):
        if CVD_aware:
            color_CB = self.lines_on_colormap['bright_bg'][0]
            color_HH = self.lines_on_colormap['bright_bg'][0]
            color_LH = color_HH
        else:
            if dark_mode:
                color_CB = self.bands_dark_background['CB']
                color_HH = self.bands_dark_background['HH']
                color_LH = self.bands_dark_background['LH']
            else:
                color_CB = self.bands['CB']
                color_HH = self.bands['HH']
                color_LH = self.bands['LH']
        return color_CB, color_HH, color_LH
    