import os
import shutil
import logging

from nnShortcuts.common import CommonShortcuts

class SlurmData:
	max_num_jobs_per_user = 5  # may be larger (check cluster's policy - also affected by numCPU requested and number of physical cores?)
	jobname = 'nnSweep'  # used to inquire job status by the commands `sacct` and `squeue`

	def __init__(self) -> None:
		self.node = None
		self.suffix = None
		self.email = None
		self.num_CPU = 1
		self.memory_limit = None
		self.time_limit_hrs = 1

		self.output_subfolders = list()
		self.unique_tag = None

		self.metascript_paths = list()
		self.sbatch_script_paths = list()


	def set(self, node, suffix, email, num_CPU, memory_limit, time_limit_hrs) -> None:
		# validate inputs
		if not isinstance(node, str): TypeError(f"Node name must be a string, not {type(node)}")
		if not isinstance(suffix, str): TypeError(f"Executable path must be a string, not {type(suffix)}")
		if not isinstance(num_CPU, int) or num_CPU < 0: ValueError(f"Illegal number of CPUs: {num_CPU}")
		if not isinstance(time_limit_hrs, int) or time_limit_hrs < 0: ValueError(f"Illegal time limit: '{time_limit_hrs} hours'")
	
		self.node = node
		self.suffix = suffix
		self.email = email
		self.num_CPU = num_CPU
		self.memory_limit = memory_limit
		self.time_limit_hrs = time_limit_hrs

		self.unique_tag = "_on_" + node + suffix


	def create_sbatch_scripts(self, input_file_fullpaths, exe, output_folder, database, license, product_name):
		"""
		Generate sbatch files to submit the sweep simulations to cloud computers by Slurm.
		"""
		# validate arguments
		if not isinstance(exe, str): TypeError(f"Executable path must be a string, not {type(exe)}")
		if not isinstance(output_folder, str): TypeError(f"Output folder must be a string, not {type(output_folder)}")
		if not isinstance(database, str): TypeError(f"Database path must be a string, not {type(database)}")
		if not isinstance(license, str): TypeError(f"License path must be a string, not {type(license)}")

		sbatch_scripts_folder = "./sbatch_temp"

		# refresh sbatch_scripts_folder
		if os.path.isdir(sbatch_scripts_folder):
			shutil.rmtree(sbatch_scripts_folder)
		os.mkdir(sbatch_scripts_folder)

		logging.info("Writing sbatch scripts...")
		
		for iInputFile, input_file_fullpath in enumerate(input_file_fullpaths):
			filename, extension = CommonShortcuts.separate_extension(input_file_fullpath)
			scriptpath = os.path.join(sbatch_scripts_folder, "run_" + filename + ".sh")
			self.sbatch_script_paths.append(scriptpath)
			output_subfolder_path = os.path.join(output_folder, filename)  # TODO: may be replaced by SweepHelper.data['output_subfolder'] or its short version

			metascript_count = iInputFile // SlurmData.max_num_jobs_per_user  # integer division
			metascript_path = os.path.join(sbatch_scripts_folder, f"metascript{metascript_count}.sh")

			with open(metascript_path, 'a') as f_meta:
				if iInputFile % SlurmData.max_num_jobs_per_user == 0:  # first input file in the current metascript
					self.metascript_paths.append(metascript_path)
					f_meta.write("#!/bin/bash\n\n")
				f_meta.write(f"sbatch {scriptpath}\n")
			
				# individual script
				self.write_sbatch_script(scriptpath, input_file_fullpath, exe, output_subfolder_path, database, license, product_name, (iInputFile+1 == len(input_file_fullpaths)))



	def write_sbatch_script(self, scriptpath, inputpath, exe, output_subfolder_path, database, license, product_name, isLastInputFile):
		"""
		Write a sbatch script for a single nextnano simulation.
		Stores output subfolder paths.

		Note
		----
			Currently, only supports nextnano++ and nextnano.NEGF++ command line syntax.
		"""
		# validate inputs
		if not isinstance(scriptpath, str): TypeError(f"Executable path must be a string, not {type(scriptpath)}")
		if not isinstance(inputpath, str): TypeError(f"Input file path must be a string, not {type(inputpath)}")
		
		# prepare paths
		filename, extension = CommonShortcuts.separate_extension(inputpath)
		unique_name = filename + self.unique_tag
		if not os.path.exists(output_subfolder_path): 
			os.makedirs(output_subfolder_path)

		# initialize log file
		logfile = os.path.join(output_subfolder_path, unique_name + ".log")
		if os.path.isfile(logfile):
			os.remove(logfile)

		# write sbatch script
		with open(scriptpath, 'w') as f:
			f.write("#!/bin/bash\n\n")
			f.write(f"#SBATCH --partition={self.node}\n")
			f.write(f"#SBATCH --cpus-per-task={self.num_CPU}\n")
			f.write(f"#SBATCH --mem={self.memory_limit}\n")
			f.write("#SBATCH --nodes=1\n")  # multinode parallelism with MPI not implemented in nextnano
			f.write(f"#SBATCH --time={self.time_limit_hrs}:00:00\n")
			f.write(f"#SBATCH --hint=multithread\n")
			f.write(f"#SBATCH --output={logfile}\n")
			f.write("\n")
			f.write(f"#SBATCH --job-name={SlurmData.jobname}\n")
			f.write(f"#SBATCH --comment='Python Sweep simulation'\n")
			f.write("#SBATCH --mail-type=end\n")
			if self.email is not None and isLastInputFile:
				f.write(f"#SBATCH --mail-user={self.email}\n")
			f.write("\n")
			if product_name == 'nextnano.NEGF++':
				f.write(f"{exe} -i \"{inputpath}\" -o \"{output_subfolder_path}\" -m \"{database}\" -c -l \"{license}\" -t {self.num_CPU} -v splitfile")
			elif product_name == 'nextnano++':
				f.write(f"{exe} -o \"{output_subfolder_path}\" -d \"{database}\" -l \"{license}\" -t {self.num_CPU} \"{inputpath}\"")


	def delete_sbatch_scripts(self):
		for script in self.sbatch_script_paths:
			if os.path.exists(script):
				os.remove(script)
		for metascript in self.metascript_paths:
			if os.path.exists(metascript):
				os.remove(metascript)

