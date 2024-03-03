
class DefaultColors:
   bands = dict()
   bands_dark_background = dict()

   colormap = ''
   lines_on_colormap = ''

   current_voltage = ''
   light_voltage = ''

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
      self.bands_dark_background = {
         'Gamma': 'salmon',
         'CB': 'salmon',
         'HH': 'cyan',
         'LH': 'lime',
         'SO': 'wheat',
         'kp6': 'magenta',
         'kp8': 'silver'
      }

      self.colormap = 'cividis'
      self.lines_on_colormap = 'white'

      self.current_voltage = 'tab:blue'
      self.light_voltage = 'tab:red'
