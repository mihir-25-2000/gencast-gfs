#! /bin/bash
#PBS -N graphcast_test
#PBS -o /home/mihir.more/myenv/out1.log
#PBS -e /home/mihir.more/myenv/err1.log
#PBS -l ncpus=1
#PBS -q gpu
#PBS -l host=gpu-h100
#PBS -k oe

module load /Datastorage/apps/utills/modulefiles/anaconda3-2022.5

conda init

source ~/.bashrc

conda activate /home/mihir.more/miniconda3/envs/myenv

cd /home/mihir.more/git-practice/

python3 tut.py