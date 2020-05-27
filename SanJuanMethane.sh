#!/bin/bash

#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 1-00:00
#SBATCH -p huce_amd
#SBATCH --mem-per-cpu=16000
#SBATCH -o logs/SanJuan.out
#SBATCH -e logs/SanJuan.err

date

# set user defined libraries 
source activate py37

python Driver2.py -d ang20190621t194713_rdn_v2u1_img_SanJuanMethane -n ang20190621t194713_rdn_v2u1_img_SanJuanMethane -s SanJuanMethane -x 108.391 -y 36.793 -m ang20190621t194713_rdn_v2u1_img_SanJuanMethane -t ang20190621t194713_rdn_v2u1_img_SanJuanMethane

source deactivate

date

