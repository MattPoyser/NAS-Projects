#!/bin/bash
# bash ./scripts-search/search-cifar.sh cifar10 ResNet110 CIFAR 0.57 777
set -e
echo script name: $0
echo $# arguments
#if [ "$#" -ne 5 ] ;then
#  echo "Input illegal number of parameters " $#
#  echo "Need 5 parameters for the dataset and the-model-name and the-optimizer and FLOP-ratio and the-random-seed"
#  exit 1
#fi
#if [ "$TORCH_HOME" = "" ]; then
#  echo "Must set TORCH_HOME envoriment variable for data dir saving"
#  exit 1
#else
#  echo "TORCH_HOME : $TORCH_HOME"
#fi

dataset=cifar10
model=ResNet110
optim=CIFARX
batch=256
gumbel_min=0.1
gumbel_max=5
expected_FLOP_ratio=0.47
rseed=14362
subset_size=$1
hardness=$2
mastery=$3
data_path="/home2/lgfm95/cifar10/"
#data_path="/hdd/PhD/data/cifar10/"

save_dir=./output/search-shape/${dataset}-${model}-${optim}-Gumbel_${gumbel_min}_${gumbel_max}-${expected_FLOP_ratio}

# normal training
xsave_dir=${save_dir}/seed-${rseed}-NMT
OMP_NUM_THREADS=4 python3 ./exps/basic-main.py --dataset ${dataset} \
--data_path ${data_path} \
--model_config ${save_dir}/seed-${rseed}-last.config \
--optim_config ./configs/opts/CIFAR-E300-W5-L1-COS.config \
--procedure    basic \
--save_dir     ${xsave_dir} \
--cutout_length -1 \
--batch_size 256 --rand_seed ${rseed} --workers 6 \
--eval_frequency 1 --print_freq 100 --print_freq_eval 200
# KD training
xsave_dir=${save_dir}/seed-${rseed}-KDT
OMP_NUM_THREADS=4 python3 ./exps/KD-main.py --dataset ${dataset} \
--data_path $TORCH_HOME/cifar.python \
--model_config  ${save_dir}/seed-${rseed}-last.config \
--optim_config  ./configs/opts/CIFAR-E300-W5-L1-COS.config \
--KD_checkpoint ./.latent-data/basemodels/${dataset}/${model}.pth \
--procedure    Simple-KD \
--save_dir     ${xsave_dir} \
--KD_alpha 0.9 --KD_temperature 4 \
--cutout_length -1 \
--batch_size 256 --rand_seed ${rseed} --workers 6 \
--eval_frequency 1 --print_freq 100 --print_freq_eval 200
