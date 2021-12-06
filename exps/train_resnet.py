import sys
from pathlib import Path
import os
import shutil

lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from models import obtain_model, load_net_from_checkpoint
from datasets import get_datasets, get_datasets_augment
import torch
import torch.nn as nn
from models.CifarResNet import CifarResNet

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--fashion', default=False, type=bool, help="are images fashionMnist?")

def main(kd_checkpoint, fashion=False):
    # kd_checkpoint = "/hdd/PhD/nas/tas/mnist110/checkpoint.pth.tar"
    model = load_net_from_checkpoint(kd_checkpoint)
    checkpoint = torch.load(kd_checkpoint)
    model_config = checkpoint['model-config']
    # model_config['dataset'] = 'mnist'
    # if fashion:
    #     model_config['dataset'] = 'fashion'
    # train_data, valid_data, xshape, class_num = get_datasets_augment("mnist", "/hdd/PhD/data/mnist", -1, kd=True)
    train_data, valid_data, xshape, class_num = get_datasets_augment("mnist", "/home2/lgfm95/mnist", -1, kd=True)
    if fashion:
        train_data, valid_data, xshape, class_num = get_datasets_augment("fashion", "/home2/lgfm95/fashionMnist", -1, kd=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True,
                                               num_workers=0, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=128, shuffle=False,
                                               num_workers=0, pin_memory=True)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 1e-3, momentum=0.9)

    # model.layers[0].conv = nn.Conv2d(1, 16, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    print(model.layers[0].conv)

    best = 0
    for i in range(50):
        train_losses = AverageMeter('Loss', ':.4e')
        train_accs = AverageMeter('Acc', ':.4e')
        valid_losses = AverageMeter('Loss', ':.4e')
        valid_accs = AverageMeter('Acc', ':.4e')
        for step, (images, labels) in enumerate(train_loader):
            outputs, outputs_e = model(images)
            loss = criterion(outputs, labels)

            train_losses.update(loss.item(), images.size(0))
            train_accs.update(accuracy(outputs, labels)[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                print(f"step {step} / {len(train_loader)}")
            # if step > 50:
            #     break
        print(f"epoch {i} / 50: train_accuracy: {train_accs.avg}, train_loss: {train_losses.avg}")

        for step, (images, labels) in enumerate(valid_loader):
            with torch.no_grad():
                outputs, outputs_e = model(images)
                loss = criterion(outputs, labels)
                valid_losses.update(loss.item(), images.size(0))
                valid_accs.update(accuracy(outputs, labels)[0])
                if step % 10 == 0:
                    print(f"step {step} / {len(valid_loader)}")
                # if step > 50:
                #     break
        if valid_accs.avg > best:
            best = valid_accs.avg
            print("saving")
            # save_checkpoint(model, "/hdd/PhD/nas/tas/mnist110/")
            if fashion:
                save_checkpoint(model, model_config, "/home2/lgfm95/nas/tas/fashion110/")
            else:
                save_checkpoint(model, model_config, "/home2/lgfm95/nas/tas/mnist110/")
                # save_checkpoint(model, model_config, "/hdd/PhD/nas/tas/mnist110/")
        print(f"epoch {i} / 50: valid_accuracy: {valid_accs.avg}, valid_loss: {valid_losses.avg}")



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res



def save_checkpoint(model, model_config, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save({'model-config': model_config,
                'base-model': model.state_dict()}, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)


if __name__ == '__main__':
    args = parser.parse_args()
    fashion = args.fashion
    if fashion:
        main("./.latent-data/basemodels/fashion/ResNet110.pth", fashion=True)
    else:
        main("./.latent-data/basemodels/mnist/ResNet110.pth")
