#!/bin/bash
# bash ./scripts/nas-infer-train.sh cifar10 SETN 256 -1
echo script name: $0
echo $# arguments
#if [ "$#" -ne 4 ] ;then
#  echo "Input illegal number of parameters " $#
#  echo "Need 4 parameters for dataset and the-model-name and epochs and LR and the-batch-size and the-random-seed"
#  exit 1
#fi
#if [ "$TORCH_HOME" = "" ]; then
#  echo "Must set TORCH_HOME envoriment variable for data dir saving"
#  exit 1
#else
#  echo "TORCH_HOME : $TORCH_HOME"
#fi

dataset=cifar10
model=SETN
batch=96
rseed=-1
subset_size=$1
hardness=$2
mastery=$3
root=/home2/lgfm95/cifar10/

if [ ${dataset} == 'cifar10' ] || [ ${dataset} == 'cifar100' ]; then
  xpath=$TORCH_HOME/cifar.python
  base=CIFAR
  workers=4
  cutout_length=16
elif [ ${dataset} == 'imagenet-1k' ]; then
  xpath=$TORCH_HOME/ILSVRC2012
  base=IMAGENET
  workers=28
  cutout_length=-1
else
  echo 'Unknown dataset: '${dataset}
fi

SAVE_ROOT="./output"

save_dir=${SAVE_ROOT}/nas-infer/${dataset}-${model}-${batch}

python3 --version

python3 ./exps/basic-main.py --dataset ${dataset} \
	--data_path ${root} --model_source nas \
	--name ${dataset} \
	--model_config ./configs/archs/NAS-${base}-${model}.config \
	--optim_config ./configs/opts/NAS-${base}.config \
	--procedure    basic \
	--save_dir     ${save_dir} \
	--cutout_length ${cutout_length} \
	--batch_size  ${batch} --rand_seed ${rseed} --workers ${workers} \
	--eval_frequency 1 --print_freq 500 --print_freq_eval 1000 \
	--init_train_epochs 5 \
	--ncc 1 \
	--dynamic 1 \
	--subset_size ${subset_size} \
	--hardness ${hardness} \
	--mastery ${mastery}
