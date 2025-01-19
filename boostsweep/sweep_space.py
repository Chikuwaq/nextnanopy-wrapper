import numpy as np
import logging
import copy

class SweepSpace:
	"""
	Contains dict of numpy.ndarray of the form 
	{ 'sweep variable': array of values }
	"""
	data : dict = dict()

	def __init__(self, dictionary : dict) -> None:
		self.data = dictionary
	
	# "constructor overloading" in Python
	@classmethod
	def create_from_sweep_ranges(cls, sweep_ranges, round_decimal):
		dictionary = dict()

		if isinstance(sweep_ranges, dict):
			for var, range in sweep_ranges.items():
				if isinstance(range, tuple):  # min, max, and number of points have been given
					bounds, num_points = range
					if bounds[0] == bounds[1] and num_points > 1:
						raise RuntimeError(f"Sweep variable {var} has min = max, but more than one simulation is requested!")
					dictionary[var] = np.around(np.linspace(bounds[0], bounds[1], num_points), round_decimal)   # round to avoid lengthy filenames
				elif isinstance(range, list):  # list of values has been given
					dictionary[var] = np.around(np.array(range), round_decimal)   # round to avoid lengthy filenames
		elif isinstance(sweep_ranges, list):
			for i_variable, var in enumerate(sweep_ranges[0]):
				if not isinstance(var, str): 
					raise TypeError("First tuple of `sweep_ranges` must be str specifying variable names!")
				dictionary[var] = [sweep_ranges[1][i_variable]]  # tentative: store only the first value
			assert(len(dictionary.keys()) > 0)
		else:
			raise TypeError(f"__init__(): argument 'sweep_ranges' must be a either dict or list, but is {type(sweep_ranges)}")

		logging.debug("\nSweep space axes:")
		logging.debug(f"{ [ key for key in dictionary.keys() ] }")
		return cls(dictionary)

	def has_sweep_variable(self, key : str):
		return key in self.data
	
	def get_items(self):
		"""
		Returns (variable name)-(array of values) pairs to iterate over.
		"""
		return self.data.items()

	def get_variable_names(self):
		return self.data.keys()

	def get_values(self):
		return self.data.values()
	
	def get_values_by_variable_name(self, key):
		"""
		Returns array of values in the sweep space for given variable name.
		"""
		if key in self.data.keys():
			return self.data[key]
		else:
			raise KeyError(f"Unknown sweep variable '{key}'!")
		
	def get_nPoints_by_variable_name(self, key):
		"""
		Returns the number of values along the given axis.
		"""
		if key in self.data.keys():
			return len(self.data[key])
		else:
			raise KeyError(f"Unknown sweep variable '{key}'!")
		
	def get_dict(self):
		return self.data
	
	def extract_1D_line(self, sweep_var):
		"""
		Extract 1D line from multidimensional (d >= 1) sweep space.
		"""
		if not self.has_sweep_variable(sweep_var):
			raise KeyError(f"Unknown sweep variable '{sweep_var}'!")

		sweep_space_reduced = copy.deepcopy(self.data)

		# ask the values for other axes
		logging.info(f"Taking '{sweep_var}' for plot axis.")
		for var, array in self.get_items():
			if var == sweep_var: 
				continue

			print("\nRemaining sweep dimension: ", var)
			print("Simulation has been performed at: ")
			for i, val in enumerate(array):
				print(f"index {i}: {val}")
			if len(array) == 1:
				iChoice = 0
			else:
				while True:
					choice = input("Specify value for the plot with index: ")
					if choice == 'q':
						raise RuntimeError('Nextnanopy terminated.') from None
					try:
						iChoice = int(choice)
					except ValueError:
						print("Invalid input. (Type 'q' to quit)")
						continue
					if iChoice not in range(len(array)):
						print("Invalid input. (Type 'q' to quit)")
					else:
						break
			sweep_space_reduced[var] = [array[iChoice]]   # only one element, but has to be an Iterable for the use below
		return SweepSpace(sweep_space_reduced)


	def extract_2D_plane(self, sweep_var1, sweep_var2):
		"""
		Extract 2D plane from multidimensional (d >= 2) sweep space.
		"""
		if not self.has_sweep_variable(sweep_var1):
			raise KeyError(f"Unknown sweep variable {sweep_var1}!")
		if not self.has_sweep_variable(sweep_var2):
			raise KeyError(f"Unknown sweep variable {sweep_var2}!")
		
		sweep_space_reduced = copy.deepcopy(self.data)

		# ask the values for other axes
		logging.info(f"Taking '{sweep_var1}' and '{sweep_var2}' for plot axes.")
		for var, array in self.get_items():
			if var == sweep_var1 or var == sweep_var2: 
				continue

			print("\nRemaining sweep dimension: ", var)
			print("Simulation has been performed at: ")
			for i, val in enumerate(array):
				print(f"index {i}: {val}")
			if len(array) == 1:
				iChoice = 0
			else:
				while True:
					choice = input("Specify value for the plot with index: ")
					if choice == 'q':
						raise RuntimeError('Nextnanopy terminated.') from None
					try:
						iChoice = int(choice)
					except ValueError:
						print("Invalid input. (Type 'q' to quit)")
						continue
					if iChoice not in range(len(array)):
						print("Invalid input. (Type 'q' to quit)")
					else:
						break
			sweep_space_reduced[var] = [array[int(iChoice)]]   # only one element, but has to be an Iterable for the use below
		logging.debug("Extracted sweep_space", sweep_space_reduced)
		return SweepSpace(sweep_space_reduced)
	
	def has_sweep_point(self, coords):
		"""
		Returns True if the point belongs to the sweep space
		"""
		return all(coords[i] in self.get_values_by_variable_name(var) for i, var in enumerate(self.data.keys()))