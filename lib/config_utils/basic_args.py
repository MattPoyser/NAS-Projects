##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import os, sys, time, random, argparse
from .share_args import add_shared_args

def obtain_basic_args():
  parser = argparse.ArgumentParser(description='Train a classification model on typical image classification datasets.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--name', type=str, default="mnist", help='dataset to use')
  parser.add_argument('--resume'      ,     type=str,                   help='Resume path.')
  parser.add_argument('--init_model'  ,     type=str,                   help='The initialization model path.')
  parser.add_argument('--model_config',     type=str,                   help='The path to the model configuration')
  parser.add_argument('--optim_config',     type=str,                   help='The path to the optimizer configuration')
  parser.add_argument('--procedure'   ,     type=str,                   help='The procedure basic prefix.')
  parser.add_argument('--model_source',     type=str,  default='normal',help='The source of model defination.')
  add_shared_args( parser )
  # Optimization options
  parser.add_argument('--batch_size',       type=int,  default=2,       help='Batch size for training.')
  parser.add_argument('--subset_size',    type=int, help='subset size')
  parser.add_argument('--hardness',    type=float, help='hardness')
  parser.add_argument('--mastery',    type=float, help='mastery')
  parser.add_argument('--issave', type=bool, default=False, help='do we save indices for naswot')
  parser.add_argument('--dynamic', type=bool, default=False, help='are we doing dynamic dataset')
  parser.add_argument('--vanilla', type=bool, default=False, help='are we doing vanilla')
  parser.add_argument('--isbad', type=bool, default=False, help='are we using bad autoencoder (ablation)')
  parser.add_argument('--isTree', type=bool, default=False, help='do we use tree in dynamic dataloader (probably true)')
  parser.add_argument('--init_train_epochs', type=int, default=5,
                      help='minimum no. epochs to train before updating dynamic subset')
  parser.add_argument('--is_csv', type=bool, default=False, help='saving with csv?')
  parser.add_argument('--is_detection', type=bool, default=False, help='object detection?')
  parser.add_argument('--ncc', type=bool, default=False, help='are we on ncc?')
  parser.add_argument('--visualize', type=bool, default=False, help='are we visualizing results')

  args = parser.parse_args()

  if args.rand_seed is None or args.rand_seed < 0:
    args.rand_seed = random.randint(1, 100000)
  assert args.save_dir is not None, 'save-path argument can not be None'
  return args
