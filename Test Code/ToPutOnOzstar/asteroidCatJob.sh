#!/bin/bash
#
#SBATCH --job-name=TESS_S22_Asteroid_Cat
#SBATCH --output=/fred/oz335/bleicester/joblogs/%A_Asteroid_Cat_Logs.txt
#SBATCH --error=/fred/oz335/joblogs/bleicester/%A_asteroid_Cat_errors.txt
#
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G

python /fred/oz335/bleicester/Code/runAstrCat.py