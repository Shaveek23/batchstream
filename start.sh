#!/bin/bash

#SBATCH --job-name=stream
#SBATCH --cpus-per-task=54
#SBATCH --mail-type=END,FAIL
#SBATCH --gpus-per-node=0
#SBATCH --mail-user=01142171@pw.edu.pl
#SBATCH --mem=64G
#SBATCH -A stream-mining

source /home2/faculty/pgolik/.pyenv/versions/batchstream/bin/activate

python /home2/faculty/pgolik/batchstream/batchstream/main_rbf.py

