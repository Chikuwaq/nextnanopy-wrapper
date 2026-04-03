import numpy as np
import matplotlib.ticker as ticker

import logging
import warnings

class ScientificPlotter:
    """
    We make the methods static because:
    - these utility functions do not depend on the class state but makes sense that they belong to the class
    - we want to make this method available without instantiation of an object.
    """

    # -------------------------------------------------------
    # Constructor
    # -------------------------------------------------------
    def __init__(self, loglevel=logging.INFO):
        pass


    @staticmethod
    def __count_nonzero_digits(number):
        flat_string = f"{number:.20f}"  # 20 decimal digits should be large enough for practical usage
        flat_string = flat_string.rstrip('0').rstrip('.')  # remove trailing zeros and decimal point
        flat_string = flat_string.lstrip('-')  # remove leading negative sign
        # flat_string = flat_string.lstrip('0').lstrip('.')  # remove leading zeros and decimal point
        # return len(flat_string.split('.')[-1])
        integers = list()
        for digit in flat_string:
            if digit == '.':
                continue
            integers.append(int(digit))
        return np.count_nonzero(integers)


    @staticmethod
    def __make_number_scientific(number, round_decimal):
        n_digits = ScientificPlotter.__count_nonzero_digits(number)
        n_decimals = min(round_decimal, n_digits - 1)
        return f"{number:.{n_decimals}e}"


    @staticmethod
    def __split_base_and_exponent(scientific):
        base, exponent = scientific.split('e')
        base = base.rstrip('0')  # remove trailing zeros
        base = base.rstrip('.')  # remove decimal dot if no decimals follow
        exponent = exponent.lstrip('+')  # remove leading plus sign
        exponent = exponent.lstrip('0')  # remove leading zeros
        return base, exponent


    @staticmethod
    def __polish_scientific_number(scientific):
        base, exponent = ScientificPlotter.__split_base_and_exponent(scientific)
        return f"{base}e{exponent}"


    @staticmethod
    def format_number_in_power_notation(number, round_decimal) -> str:
        scientific = ScientificPlotter.__make_number_scientific(number, round_decimal)
        base, exponent = ScientificPlotter.__split_base_and_exponent(scientific)
        if len(exponent) > 0:
            exponent = int(exponent)
            return rf"${base}\times 10^{{{exponent}}}$"
        else:
            return base


    @staticmethod
    def format_number_in_power_notation_fixed_exponent(number, desired_exponent) -> str:
        mantissa = number / (10 ** desired_exponent)
        return rf"${mantissa:g}\times 10^{{{desired_exponent}}}$"  # ':g' removes trailing zeros

    @staticmethod
    def __polish_nonscientific_number(nonscientific):
        number_str = str(nonscientific)
        if '.' in number_str:
            number_str = number_str.rstrip('0')  # remove trailing zeros
            number_str = number_str.rstrip('.')  # remove trailing decimal point
        return number_str


    # @staticmethod
    # def __format_number_scientific(number, round_decimal) -> str:
    #     """
    #     Return scientific expression of a value.
    #     """
    #     if isinstance(number, str):
    #         number = float(number)

    #     scientific = ScientificPlotter.__make_number_scientific(number, round_decimal)
    #     return ScientificPlotter.__polish_scientific_number(scientific)


    @staticmethod
    def format_number_for_axis(number, round_decimal, abs_max) -> str:
            too_many_digits = (abs_max > 1e4 or abs_max < 0.1)
            if too_many_digits:
                return ScientificPlotter.format_number_in_power_notation(number, round_decimal)
            else: #if abs_min > 1:
                return ScientificPlotter.__polish_nonscientific_number(number)
            # else: # use scientific format if the data range is complicated
            #     return CommonShortcuts.__format_number_scientific(number, round_decimal)


    @staticmethod
    def __get_abs_min_max(ax):
        min_value, max_value = ax.get_ylim()
        abs_min = min(abs(min_value), abs(max_value))
        abs_max = max(abs(min_value), abs(max_value))
        return abs_min, abs_max


    @staticmethod
    def format_major_ytick_labels(ax, round_decimal):
        """
        Retrieves major tick locations from ax and improves the label format.
        
        Args:
            ax: matplotlib Axes object
            round_decimal: argument for CommonShortcuts.format_number()
        """
        abs_min, abs_max = ScientificPlotter.__get_abs_min_max(ax)

        # Customize formatter 
        class MajorFormatter(ticker.Formatter):
            def __init__(self):
                super().__init__()

            def __call__(self, x, pos=None):
                return ScientificPlotter.format_number_for_axis(x, round_decimal, abs_max)
            
        # Apply to MAJOR ticks only
        formatter = MajorFormatter()
        ax.yaxis.set_major_formatter(formatter)


    @staticmethod
    def format_minor_ytick_labels(ax, first_minor, last_minor, round_decimal):
        """
        Adds two minor ytick labels to ax.
        """
        class MinMaxMinorFormatter(ticker.Formatter):
            def __init__(self, first, last):
                self.first = first
                self.last = last

            def __call__(self, y, pos=None):
                if abs(y - self.first) < 1e-10 or abs(y - self.last) < 1e-10:
                    abs_min, abs_max = ScientificPlotter.__get_abs_min_max(ax)
                    return ScientificPlotter.format_number_for_axis(y, round_decimal, abs_max)
                return ''

            # def format_ticks(self, values):
            #     """Properly format multiple ticks at once"""
            #     return [self(v, pos=None) for v in values]

        # Apply ONLY to minor ticks - major formatter completely untouched
        formatter = MinMaxMinorFormatter(first_minor, last_minor)
        ax.yaxis.set_minor_formatter(formatter)

    @staticmethod
    def add_min_max_minor_ytick_labels(ax, round_decimal):
        """
        Adds labels to first/last minor tick WITHOUT affecting existing major tick labels.
        Skips min minor if > min major, skips max minor if < max major.
        Improves data readability especially in a log plot.
        """
        y_min, y_max = ax.get_ylim()

        # Get tick locations in visible range
        major_locs = ax.get_yticks()
        visible_major_locs = major_locs[(major_locs >= y_min) & (major_locs <= y_max)]
        minor_locs = ax.yaxis.get_minorticklocs()
        visible_minor_locs = minor_locs[(minor_locs >= y_min) & (minor_locs <= y_max)]
        if len(visible_minor_locs) < 2:
            return

        first_minor = visible_minor_locs[0]
        last_minor = visible_minor_locs[-1]

        # Filter: label minors only if they enlarge the range of tick labels
        candidates = []
        if len(visible_major_locs) > 0:
            min_major = visible_major_locs[0]
            if len(visible_major_locs) == 1:
                max_major = min_major
            else:
                max_major = visible_major_locs[-1]

            if first_minor < min_major:
                candidates.append(first_minor)
            if last_minor > max_major:
                candidates.append(last_minor)
            
            if len(candidates) < 1:
                return
        else: # no major tick exists
            candidates = [first_minor, last_minor]
        first_candidate = candidates[0]
        last_candidate = candidates[-1]

        ScientificPlotter.format_minor_ytick_labels(ax, first_candidate, last_candidate, round_decimal)


    @staticmethod
    def format_number_for_filename(
            number, 
            round_decimal,
            should_polish=True
            ) -> str:
        """
        Return concise string expression of a value.
        Avoids lengthy file name (Windows cannot handle deeply-nested output files if nextnano input file name is too long).
        Unlike format_number_for_axis(), this method avoids the power notation.

        Parameters
        ----------
        should_polish : bool, optional
            Determines whether trailing characters should be removed.
        """
        if isinstance(number, str):
            number = float(number)

        is_integer_type = isinstance(number, (int, np.integer))
        is_float_type = isinstance(number, (float, np.floating))

        if is_integer_type or is_float_type:
            if number == 0:
                return '0'
            else:
                scientific = ScientificPlotter.__make_number_scientific(number, round_decimal)
                use_scientific = (len(scientific) < len(str(number)))  # do not use scientific format if the string would get longer

                if use_scientific:
                    return ScientificPlotter.__polish_scientific_number(scientific)
                else:
                    if should_polish:
                        return ScientificPlotter.__polish_nonscientific_number(number)
                    else:
                        return number
        else:
            raise TypeError(f"'number' must be str, int, or float, but is {type(number)}!")


    @staticmethod
    def get_maximum_points(quantity_arr, position_arr):
        if isinstance(quantity_arr, int) or isinstance(quantity_arr, float):
            warnings.warn(f"get_maximum_points(): Only one point exists in the array {quantity_arr}", category=RuntimeWarning)
            return position_arr[0], quantity_arr

        if len(quantity_arr) != len(position_arr):
            raise ValueError('Array size does not match!')
        ymax = np.amax(quantity_arr)
        if np.size(ymax) > 1:
            warnings.warn("Multiple maxima found. Taking the first...")
            ymax = ymax[0]
        xmaxIndex = np.where(quantity_arr == ymax)[0]
        xmax = position_arr[xmaxIndex.item(0)]             # type(xmaxIndex.item(0)) is 'int'

        return xmax, ymax


    @staticmethod
    def place_texts(ax, texts):
        """
        Locate first list of texts on the left and second on the right of the plot specified by matplotlib.Axes.
        """
        hPosition = 0.5
        for text in texts[0]:
            ax.text(0.04, hPosition, text, transform=ax.transAxes)
            hPosition -= 0.07
        hPosition = 0.5
        for text in texts[1]:
            ax.text(0.55, hPosition, text, transform=ax.transAxes)
            hPosition -= 0.07
        if len(texts) > 2:
            raise RuntimeError("Too many lists of texts requested.")


    @staticmethod
    def __findCell(arr, wanted_value):
        """
        Find the grid cell that contains given wanted position and return index.
        #TODO: numpy.array.argmin() can be applied also to non-monotonic arrays
        """
        num_nodes = len(arr)
        cnt = 0
        for i in range(num_nodes-1):
            if arr[i] <= wanted_value < arr[i + 1]:
                start_index = i
                end_index = i + 1
                cnt = cnt + 1

        if cnt == 0:
            raise RuntimeError(f'No grid cells found that contain the point x = {wanted_value}')
        if cnt > 1:
            raise RuntimeError(f'Multiple grid cells found that contain the point x = {wanted_value}')
        return start_index, end_index


    @staticmethod
    def get_value_at_position(quantity_arr, position_arr, wantedPosition):
        """
        Get value at given position.
        If the position does not match any of array elements due to inconsistent gridding, interpolate array and return the value at wanted position.
        """
        if len(quantity_arr) != len(position_arr):
            raise ValueError('Array size does not match!')

        start_idx, end_idx = ScientificPlotter.__findCell(position_arr, wantedPosition)

        # linear interpolation
        x_start = position_arr[start_idx]
        x_end   = position_arr[end_idx]
        y_start = quantity_arr[start_idx]
        y_end   = quantity_arr[end_idx]
        tangent = (y_end - y_start) / (x_end - x_start)
        return tangent * (wantedPosition - x_start) + y_start


    @staticmethod
    def draw_contour(
            ax, 
            X,
            Y,
            Z,
            contour_values,
            color,
            should_annotate=True,
            contour_symbol=None
            ):
        if not isinstance(contour_values, list):
            raise TypeError("'contour_values' must be a list!")
        contour = ax.contour(X, Y, Z, levels=contour_values, colors=color)
        if should_annotate:
            if contour_symbol is None:
                ax.clabel(contour, inline=True)
            else:
                ax.clabel(contour, inline=True, fmt=lambda x: f"{contour_symbol}={x}")
        return contour


    @staticmethod
    def __validate_data_shape_XYZ(X, Y, Z):
        """
        Add proper assertion about the shapes of X, Y, Z. It depends on 'shading' option of pcolormesh(). 
        See https://matplotlib.org/stable/gallery/images_contours_and_fields/pcolormesh_grids.html
        """
        Z_shape = Z.T.shape  # (Ny, Nx) after transpose
        X_shape = X.shape    # Should be (Ny, Nx) or (Ny+1, Nx+1)
        Y_shape = Y.shape    # Should be (Ny, Nx) or (Ny+1, Nx+1)

        assert X_shape == Y_shape, f"X {X_shape} and Y {Y_shape} shape mismatch"
        assert len(X_shape) == 2, f"X,Y must be 2D, got {X_shape}"

        if X_shape == Z_shape:
            logging.info("shading='auto' → 'nearest' (X,Y,Z same shape)")
        elif X_shape[0] == Z_shape[0] + 1 and X_shape[1] == Z_shape[1] + 1:
            logging.info("shading='auto' → 'flat' (X,Y one larger than Z)")
        else:
            raise ValueError(
                f"pcolormesh shape mismatch: Z.T {Z_shape}, "
                f"X/Y {X_shape}. Need (Ny,Nx) or (Ny+1,Nx+1)"
            )


    @staticmethod
    def draw_2D_color_plot(
            fig, 
            ax, 
            X, 
            Y, 
            Z, 
            is_divergent, 
            colormap, 
            label, 
            cbar_unit, 
            bias, 
            labelsize, 
            titlesize,
            ticksize, 
            ymin, 
            ymax, 
            zmin, 
            zmax, 
            showBias, 
            xlabel, 
            ylabel="Energy ($\mathrm{eV}$)", 
            cbar_label=None
            ):
        """
        ymin : float
            lower bound of y values. If None, it is automatically set to the minimum of the data 'Y'.
        ymax : float
            upper bound of y values. If None, it is automatically set to the maximum of the data 'Y'.
        zmin : float
            lower bound of y values. If None, it is automatically set to the minimum of the data 'Z'.
        zmax : float
            upper bound of y values. If None, it is automatically set to the maximum of the data 'Z'.
        cbar_label : str, optional
            If not None, this overwrites the colorbar label.
        """
        ScientificPlotter.__validate_data_shape_XYZ(X, Y, Z)

        from matplotlib import colors
        if is_divergent:
            pcolor = ax.pcolormesh(X, Y, Z.T, cmap=colormap, norm=colors.CenteredNorm(vcenter=0, halfrange=zmax), shading='auto')
        else:
            pcolor = ax.pcolormesh(X, Y, Z.T, vmin=zmin, vmax=zmax, cmap=colormap, shading='auto')

        if np.abs(Z).max() >= 1e4:
            cbar = fig.colorbar(pcolor, format='%.1e')
        else:
            cbar = fig.colorbar(pcolor)
        if cbar_label is None:
            if cbar_unit is None:
                    cbar_label = None
            else:
                    if showBias:
                        cbar_label = label + ' (' + cbar_unit + ')'
                    else:
                        cbar_label = '(' + cbar_unit + ')'
        else:
            if cbar_unit is None:
                    cbar_label = cbar_label
            else:
                    cbar_label = cbar_label  + ' (' + cbar_unit + ')'

        if cbar_label is not None:
            if len(cbar_label) > 35:
                    cbar.set_label(cbar_label, fontsize=labelsize*0.8)
            else:
                    cbar.set_label(cbar_label, fontsize=labelsize)
        cbar.ax.tick_params(labelsize=ticksize * 0.9)

        if not isinstance(xlabel, str):
            raise TypeError(f"'xlabel' must be str, not {type(xlabel)}")
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=labelsize)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=labelsize, labelpad=10)

        ax.set_xlim(np.amin(X), np.amax(X))
        if ymin is None:
            ymin = np.amin(Y)
        if ymax is None:
            ymax = np.amax(Y)
        ax.set_ylim(ymin, ymax)

        if showBias:
            ax.set_title(f'bias={bias}mV', fontsize=titlesize)
        else:
            ax.set_title(label, fontsize=titlesize)
        ax.tick_params(labelsize=ticksize)
        