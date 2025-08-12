import os
import shutil
import logging
import subprocess

from nnShortcuts.common import CommonShortcuts

class SlurmData:
	max_num_jobs_per_user = 8  # may be larger (check cluster's policy - also affected by numCPU requested and number of physical cores?)
	jobname = 'nnSweep'  # used to inquire job status by the commands `sacct` and `squeue`
	sbatch_file_name = "run_"
	metascript_name = "submit_jobs.sh"

	def __init__(self, output_folder) -> None:
		self.partition = None
		self.nodelist = None
		self.suffix = None
		self.email = None
		self.num_CPU = 1
		self.memory_limit = None
		self.time_limit_hrs = 1

		self.output_folder = output_folder
		self.unique_tag = None

		self.metascript_paths = list()
		self.sbatch_script_paths = list()

		# cache file containing metascript paths
		# The user can recover SlurmData even after destructing the SweepManager object, by providing the same sweep range.
		# This is useful e.g. when Slurm simulations take long time.
		from pathlib import Path
		folderpath = Path(output_folder)
		folderpath.mkdir(parents=True, exist_ok=True)
		
		# Cache is not reliable when multiple SweepHelpers are submitting jobs.
		# self.cache_path = os.path.join(output_folder, "cache.txt")

		# if os.path.isfile(self.cache_path):
		# 	logging.info("Reading SlurmData cache...")
		# 	with open(self.cache_path, "r") as f_cache:
		# 		lines = f_cache.readlines()
		# 	for line in lines:
		# 		if "#!/bin/bash" in line or line == "":
		# 			continue
		# 		if self.sbatch_file_name in line:
		# 			self.sbatch_script_paths.append(line)
		# 		elif self.metascript_name in line:
		# 			self.metascript_paths.append(line)
		# 		elif "Node" in line:
		# 			self.partition = line
			
		# 	if self.partition is None:
		# 		logging.error("partition data wasn't found in cache!")


	def set(self, partition, nodelist, suffix, email, num_CPU, memory_limit, time_limit_hrs) -> None:
		# validate inputs
		if not isinstance(partition, str): TypeError(f"partition name must be a string, not {type(partition)}")
		if not isinstance(nodelist, str): TypeError(f"nodelist name must be a string, not {type(nodelist)}")
		if not isinstance(suffix, str): TypeError(f"Executable path must be a string, not {type(suffix)}")
		if not isinstance(num_CPU, int) or num_CPU < 0: ValueError(f"Illegal number of CPUs: {num_CPU}")
		if not isinstance(time_limit_hrs, int) or time_limit_hrs < 0: ValueError(f"Illegal time limit: '{time_limit_hrs} hours'")
	
		# Cache is not reliable when multiple SweepHelpers are submitting jobs.
		# if os.path.isfile(self.cache_path):
		# 	os.remove(self.cache_path)

		self.partition = partition
		self.nodelist = nodelist
		self.suffix = suffix
		self.email = email
		self.num_CPU = num_CPU
		self.memory_limit = memory_limit
		self.time_limit_hrs = time_limit_hrs

		self.unique_tag = "_on_" + partition + suffix  # TODO: use this to differentiate output folders, keeping consistency with SweepManager.outputs


	def create_sbatch_scripts(self, input_file_fullpaths : list, exe, output_folder, database, license, product_name):
		"""
		Generate sbatch files to submit the sweep simulations to cloud computers by Slurm.

		Write input and output paths data to a file, which is read in at SlurmData instantiation.

		Parameters
		----------
		input_file_fullpaths : list of str
		"""
		# validate arguments
		if len(input_file_fullpaths) == 0: RuntimeError("Trying to create sbatch scripts, but no input file paths have been provided!")
		if not isinstance(exe, str): TypeError(f"Executable path must be a string, not {type(exe)}")
		if not isinstance(output_folder, str): TypeError(f"Output folder must be a string, not {type(output_folder)}")
		if not isinstance(database, str): TypeError(f"Database path must be a string, not {type(database)}")
		if not isinstance(license, str): TypeError(f"License path must be a string, not {type(license)}")

		sbatch_scripts_folder = "./sbatch_temp"
		# DO NOT delete the existing sbatch scripts! Slurm may be pending the job until a partition becomes available
		if not os.path.isdir(sbatch_scripts_folder):
			os.mkdir(sbatch_scripts_folder)

		# refresh class variables
		self.metascript_paths = list()
		self.sbatch_script_paths = list()

		logging.info("Writing sbatch scripts...")

		sbatch_file_count = 0

		# initialize metascript (list of `sbatch` commands) to be run by the user
		metascript_path = os.path.join(sbatch_scripts_folder, self.metascript_name)
		self.metascript_paths = [metascript_path]
		with open(metascript_path, 'w') as f_meta:
			f_meta.write("#!/bin/bash\n\n")

		# prepare sbatch script paths
		num_sbatch_scripts = min(self.max_num_jobs_per_user, len(input_file_fullpaths))
		
		# unique ID to differentiate sbatch scripts because many SweepManager objects might generate scripts simultaneously.
		# UUID has 16^n patters, where n is the number of digits of the ID.
		# Here we have (26 + 26 + 10)^n patterns, making the file names shorter
		# NOTE: Assuming that upper and lower cases are distinguished in the filesystem (will not work on Windows).
		id = SlurmData.__random_lowercase_letter() + SlurmData.__random_lowercase_letter() + SlurmData.__random_lowercase_letter() + SlurmData.__random_lowercase_letter()
		for sbatch_file_count in range(1, num_sbatch_scripts + 1):
			scriptpath = os.path.join(sbatch_scripts_folder, f"{self.sbatch_file_name}{sbatch_file_count}_{id}.sh")  # differentiate shell script file names to avoid overwriting when multiple SweepHelpers are submitting jobs
			self.sbatch_script_paths.append(scriptpath)

		input_file_fullpaths_for_each_sbatch = [list() for _ in range(num_sbatch_scripts)]
		for i_input, input_file_fullpath in enumerate(input_file_fullpaths):
			input_file_fullpaths_for_each_sbatch[i_input % num_sbatch_scripts].append(input_file_fullpath)

		for scriptpath, input_file_fullpaths_for_this_sbatch in zip(self.sbatch_script_paths, input_file_fullpaths_for_each_sbatch):
			assert isinstance(input_file_fullpaths_for_this_sbatch, list)
			is_last_sbatch_file = (scriptpath == self.sbatch_script_paths[-1])
			logging.info((f"Writing sbatch file {scriptpath}..."))
			self.write_sbatch_script(sbatch_file_count, scriptpath, input_file_fullpaths_for_this_sbatch, exe, output_folder, database, license, product_name, is_last_sbatch_file)
			with open(metascript_path, 'a') as f_meta:
				f_meta.write(f"sbatch {scriptpath}\n")

		assert sbatch_file_count <= self.max_num_jobs_per_user
		
		# save SlurmData to file
		# Cache is not reliable when multiple SweepHelpers are submitting jobs.
		# with open(self.cache_path, 'w') as f_cache:
		# 	f_cache.write(f"Node = {self.partition}\n")
		# 	for sbatch in self.sbatch_script_paths:
		# 		f_cache.write(f"{sbatch}\n")
		# 	for metascript in self.metascript_paths:
		# 		f_cache.write(f"{metascript}\n")


	@staticmethod
	def __random_lowercase_letter():
		import random, string
		return random.choice(string.ascii_letters + '0123456789')


	def write_sbatch_script(self, sbatch_file_count, scriptpath, inputpaths, exe, output_folder_path, database, license, product_name, is_last_sbatch_file):
		"""
		Write a sbatch script for a single nextnano simulation.
		Stores output subfolder paths.

		Parameters
		----------
		inputpaths : list of str
			input file paths that should be executed in one sbatch script.

		Note
		----
			Currently, only supports nextnano++ and nextnano.NEGF++ command line syntax.
		"""
		# validate inputs
		if not isinstance(scriptpath, str): TypeError(f"Executable path must be a string, not {type(scriptpath)}")
		if not isinstance(inputpaths, list): TypeError(f"Input file path must be a list of string, not {type(inputpaths)}")
		if self.partition is None:
			logging.error("partition is undefined!")
		
		# initialize log file
		first_filename, extension = CommonShortcuts.separate_extension(inputpaths[0])
		logfile = os.path.join(output_folder_path, f"%j_{sbatch_file_count}_{first_filename}.log")  # differentiate log file names to avoid conflicts when multiple SweepHelpers are submitting jobs
		if os.path.isfile(logfile):
			os.remove(logfile)
		
		# write sbatch script
		with open(scriptpath, 'w') as f:
			f.write("#!/bin/bash\n\n")
			f.write(f"#SBATCH --partition={self.partition}\n")
			if self.nodelist is not None:
				f.write(f"#SBATCH --nodelist={self.nodelist}")
			f.write(f"#SBATCH --cpus-per-task={self.num_CPU}\n")
			f.write(f"#SBATCH --mem={self.memory_limit}\n")
			f.write("#SBATCH --nodes=1\n")  # multinode parallelism with MPI not implemented in nextnano
			f.write(f"#SBATCH --time={self.time_limit_hrs}:00:00\n")
			f.write(f"#SBATCH --hint=multithread\n")
			f.write(f"#SBATCH --output={logfile}\n")
			f.write("\n")
			f.write(f"#SBATCH --job-name={SlurmData.jobname}\n")
			f.write(f"#SBATCH --comment='Python Sweep simulation'\n")
			if self.email is not None and is_last_sbatch_file:
				f.write("#SBATCH --mail-type=end\n")
				f.write(f"#SBATCH --mail-user={self.email}\n")
			f.write("\n")
			f.write(r"echo 'Job started at: ' `date`")
			f.write("\n")

			for inputpath in inputpaths:
				filename, extension = CommonShortcuts.separate_extension(inputpath)
				
				output_subfolder_path = os.path.join(output_folder_path, filename)
				if not os.path.exists(output_subfolder_path): 
					os.makedirs(output_subfolder_path)


				if product_name == 'nextnano.NEGF++':
					f.write(f"{exe} -i \"{inputpath}\" -o \"{output_subfolder_path}\" -m \"{database}\" -c -l \"{license}\" -t {self.num_CPU} \n")  # add '-v splitfile' if needed
				elif product_name == 'nextnano++':
					f.write(f"{exe} -o \"{output_subfolder_path}\" -d \"{database}\" -l \"{license}\" -t {self.num_CPU} \"{inputpath}\"\n")
				elif product_name == 'nextnano3':
					f.write(f"{exe} -o \"{output_subfolder_path}\" -d \"{database}\" --check -license \"{license}\" -t {self.num_CPU} -inputfile \"{inputpath}\"\n")

				f.write(f"rm -f {inputpath}\n")  # delete temporary input file after simulation has ended

			f.write(r"echo 'Job ended at: ' `date`")


	def delete_sbatch_scripts(self):
		for script in self.sbatch_script_paths:
			if os.path.exists(script):
				os.remove(script)
		for metascript in self.metascript_paths:
			if os.path.exists(metascript):
				os.remove(metascript)


	def slurm_is_running(self):
		if SlurmData.jobname is None:
			logging.error("Slurm job name is undefined!")
		if self.partition is None:
			logging.error("partition is undefined!")

		# get the job status
		# TODO: maybe `squeue` command is better since jobs aren't deleted from the list at midnight everyday.
		commands = ['sacct', '|', 'grep', SlurmData.jobname, '|', 'grep', self.partition]
		result = subprocess.run(commands, capture_output=True, text=True)
		if result.stdout == "":
			logging.error("Could not read output of sacct command. I'm not sure if Slurm is still running.")
			return False
		return ('RUNNING' in result.stdout)
            