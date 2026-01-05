#!/bin/bash -l

#SBATCH --ntasks 1
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=2
#SBATCH -o standard_output_file.%J.out
#SBATCH -e standard_error_file.%J.err
#SBATCH -t 48:00:00

lscpu

python mtan_exec_time.py

#python gp_exec_time.py
