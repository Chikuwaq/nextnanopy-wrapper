import os
import logging

class PathHandler:
	"""
	We make methods static because:
	- these utility functions do not depend on the class state but makes sense that they belong to the class
	- we want to make this method available without instantiation of an object.
	"""

	# -------------------------------------------------------
	# Constructor
	# -------------------------------------------------------
	def __init__(self, loglevel=logging.INFO):
		pass


	@staticmethod
	def separate_extension(filename):
		"""
		Separate file extension from file name.
		Returns the original filename and empty string if extension is absent.
		"""
		filename = os.path.split(filename)[1]   # remove paths if present
		filename_no_extension, extension = os.path.splitext(filename)

		if extension not in ['', '.in', '.xml', '.negf']: 
			logging.warning(f"separate_extension(): File extension {extension} is not supported by nextnano. Appending the extension to file name again...")
			filename_no_extension = filename
			extension = ''

		return filename_no_extension, extension


	@staticmethod
	def append_bias_to_path(output_folder, bias):
		bias_string = str(bias) + "mV"
		if bias_string in output_folder:
			return output_folder
		else:
			return os.path.join(output_folder, bias_string)


	@staticmethod
	def has_folder_starting_with(prefix, path='.'):
		"""
		Checks if any folder under the given path (recursively) starts with 'prefix'.

		Parameters
		----------
			prefix (str): The string to match at start of folder names.
			path (str): Directory to search from (default is current directory).

		Returns
		-------
			bool: True if at least one matching folder is found; False otherwise.
		"""
		for root, dirs, files in os.walk(path):
			for dirname in dirs:
				if dirname.startswith(prefix):
					return True
		return False
	

	@staticmethod
	def __get_folders_starting_with(path, prefix):
		"""
		Returns a list of full paths for folders in 'path' whose name starts with 'prefix'.
		
		Parameters
		----------
			path (str): The directory path in which to search.
			prefix (str): The prefix string to match at the start of folder names.
			
		Returns
		-------
			List[str]: List of full paths to folders starting with the prefix.
		"""
		folders = []
		# List all entries in the given directory
		try:
			for entry in os.listdir(path):
				full_path = os.path.join(path, entry)
				if os.path.isdir(full_path) and entry.startswith(prefix):
					folders.append(full_path)
		except FileNotFoundError:
			print(f"Directory not found: {path}")
		except PermissionError:
			print(f"Permission denied: {path}")
		if len(folders) == 0:
			raise FileNotFoundError(f"No folder found starting with {path}/{prefix}")
		return folders


	@staticmethod
	def __get_candidate_folder_paths(folder_path):
		parent_folder, rest = os.path.split(folder_path)
		if not PathHandler.has_folder_starting_with(rest, parent_folder):
			raise ValueError(f"Specified path {folder_path}* does not exist")
		candidate_folder_paths = PathHandler.__get_folders_starting_with(parent_folder, rest)
		if len(candidate_folder_paths) == 0:
			raise FileNotFoundError(f"No folder found starting with '{folder_path}'")
		return candidate_folder_paths


	@staticmethod
	# @must_use_result
	def expect_folders_to_exist(folder_path, allow_folder_name_suffix):
		"""
		Returns
		-------
			List[str]: List of full paths to folders starting with the prefix.

		Caution
		-------
			Do not forget to receive the return value of this method!
		"""
		if allow_folder_name_suffix:
			candidate_folder_paths = PathHandler.__get_candidate_folder_paths(folder_path)
		else:
			if not os.path.exists(folder_path): 
				raise ValueError(f"Specified path {folder_path} does not exist")
		return candidate_folder_paths


	@staticmethod
	def ask_user_to_choose_one(list_of_str) -> int:
		for count, file in enumerate(list_of_str):
			# filename = os.path.split(file)[1]
			print(f"Choice {count}: {file}")
		determined = False
		while not determined:
			choice = input('Enter the index of data you need: ')
			if choice == 'q':
				raise RuntimeError('Terminated nextnanopy.') from None
			try:
				choice = int(choice)
			except ValueError:
				print("Invalid input. (Type 'q' to quit)")
				continue
			else:
				if choice < 0 or choice >= len(list_of_str):
					print("Index out of bounds. Type 'q' to quit")
					continue
				else:
					determined = True
		return choice


	@staticmethod
	# @must_use_result
	def expect_single_folder_to_exist(folder_path, allow_folder_name_suffix):
		"""
		Returns
		-------
		folder_path : str
			Unique and existing folder path

		Caution
		-------
			Do not forget to receive the return value of this method!
		"""
		if allow_folder_name_suffix:
			candidate_folder_paths = PathHandler.__get_candidate_folder_paths(folder_path)
			if len(candidate_folder_paths) > 1:
				print("There are multiple candidates for the output folder paths.")
				choice = PathHandler.ask_user_to_choose_one(candidate_folder_paths)
				folder_path = candidate_folder_paths[choice]
			elif len(candidate_folder_paths) == 1:
				folder_path = candidate_folder_paths[0]
			else:
				raise FileNotFoundError(f"No folder found starting with '{folder_path}'")
		else:
			if not os.path.exists(folder_path): 
				raise ValueError(f"Specified path {folder_path} does not exist")
		return folder_path
	

	# -------------------------------------------------------
	# Access to output data
	# -------------------------------------------------------
	@staticmethod
	def get_sweep_output_folder_name(filename, *args):
		"""
		nextnanopy.sweep.execute_sweep() generates output folder with this name

		INPUT:
			filename
			args = SweepVariableString1, SweepVariableString2, ...

		RETURN:
			string of sweep output folder name

		"""
		filename_no_extension = PathHandler.separate_extension(filename)[0]
		output_folderName = filename_no_extension + '_sweep'

		for sweepVar in args:
			if not isinstance(sweepVar, str):
				raise TypeError(f'Argument {sweepVar} must be a string, but is {type(sweepVar)}')
			output_folderName += '__' + sweepVar

		return output_folderName


	@staticmethod
	def get_output_subfolder_path(sweep_output_folder_path, input_file_name):
		"""
		Return output folder path corresponding to the input file

		Parameters
		----------
		input_file_name : str
			input file name or path

		Returns
		-------
		str
			path to output folder

		"""
		subfolder_name = PathHandler.separate_extension(input_file_name)[0]
		return os.path.join(sweep_output_folder_path, subfolder_name)


	@staticmethod
	def compose_sweep_output_subfolder_name(filename, sweepCoordinates):
		"""
		nextnanopy.sweep.execute_sweep() generates output subfolders with this name

		INPUT:
			filename
			{sweepVariable1: value1, sweepVariable2: value2, ...}

		RETURN:
			string of sweep output subfolder name

		"""
		filename_no_extension = PathHandler.separate_extension(filename)[0]
		output_subfolderName = filename_no_extension + '__'

		for sweepVar, value in sweepCoordinates.items():
			if not isinstance(sweepVar, str):
				raise TypeError(f'Sweep variable must be a string, but is {type(sweepVar)}')
			try:
				val = str(value)
			except ValueError as e:
				raise Exception(f'value {value} cannot be converted to string!') from e
			else:
				output_subfolderName +=  sweepVar + '_' + val + '_'

		return output_subfolderName

