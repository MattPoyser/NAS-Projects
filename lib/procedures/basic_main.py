##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import os, sys, time, torch
from log_utils import AverageMeter, time_string
from utils import obtain_accuracy
import numpy as np
import torch.nn.functional as F


def basic_train(xloader, network, criterion, scheduler, optimizer, optim_config, extra_info, print_freq, logger, full_config):
    loss, acc1, acc5, hardness, correct = procedure(xloader, network, criterion, scheduler, optimizer, 'train',
                                                    optim_config, extra_info, print_freq, logger, full_config)
    return loss, acc1, acc5, hardness, correct


def basic_valid(xloader, network, criterion, optim_config, extra_info, print_freq, logger, full_config):
    with torch.no_grad():
        loss, acc1, acc5, _, _ = procedure(xloader, network, criterion, None, None, 'valid', None, extra_info, print_freq,
                                     logger, full_config)
    return loss, acc1, acc5


def procedure(xloader, network, criterion, scheduler, optimizer, mode, config, extra_info, print_freq, logger, full_config):
    data_time, batch_time, losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    if mode == 'train':
        network.train()
    elif mode == 'valid':
        network.eval()
    else:
        raise ValueError("The mode is not right : {:}".format(mode))

    # logger.log('[{:5s}] config ::  auxiliary={:}, message={:}'.format(mode, config.auxiliary if hasattr(config, 'auxiliary') else -1, network.module.get_message()))
    logger.log(
        '[{:5s}] config ::  auxiliary={:}'.format(mode, config.auxiliary if hasattr(config, 'auxiliary') else -1))
    end = time.time()

    hardness = [None for i in range(len(xloader))]
    correct = [None for i in range(len(xloader))]
    batch_size = full_config.batch_size

    for i, (inputs, targets) in enumerate(xloader):
        print(inputs.shape)
        if mode == 'train': scheduler.update(None, 1.0 * i / len(xloader))
        # measure data loading time
        data_time.update(time.time() - end)
        # calculate prediction and loss
        targets = targets.cuda(non_blocking=True)

        if mode == 'train': optimizer.zero_grad()

        # raise AttributeError(network)
        features, logits = network(inputs)
        if isinstance(logits, list):
            assert len(logits) == 2, 'logits must has {:} items instead of {:}'.format(2, len(logits))
            logits, logits_aux = logits
        else:
            logits, logits_aux = logits, None
        loss = criterion(logits, targets)
        if config is not None and hasattr(config, 'auxiliary') and config.auxiliary > 0:
            loss_aux = criterion(logits_aux, targets)
            loss += config.auxiliary * loss_aux

        if mode == 'train':
            new_hardness, new_correct = get_hardness(logits.cpu(), targets.cpu(), False)
            loss.backward()
            hardness[(i * batch_size):(i * batch_size) + batch_size] = new_hardness  # assumes batch 1 takes idx 0-8, batch 2 takes 9-16, etc.
            correct[(i * batch_size):(i * batch_size) + batch_size] = new_correct
            optimizer.step()

        # record
        prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 or (i + 1) == len(xloader):
            Sstr = ' {:5s} '.format(mode.upper()) + time_string() + ' [{:}][{:03d}/{:03d}]'.format(extra_info, i,
                                                                                                   len(xloader))
            if scheduler is not None:
                Sstr += ' {:}'.format(scheduler.get_min_info())
            Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(
                batch_time=batch_time, data_time=data_time)
            Lstr = 'Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})'.format(
                loss=losses, top1=top1, top5=top5)
            Istr = 'Size={:}'.format(list(inputs.size()))
            logger.log(Sstr + ' ' + Tstr + ' ' + Lstr + ' ' + Istr)

    logger.log(
        ' **{mode:5s}** Prec@1 {top1.avg:.2f} Prec@5 {top5.avg:.2f} Error@1 {error1:.2f} Error@5 {error5:.2f} Loss:{loss:.3f}'.format(
            mode=mode.upper(), top1=top1, top5=top5, error1=100 - top1.avg, error5=100 - top5.avg, loss=losses.avg))
    return losses.avg, top1.avg, top5.avg, hardness, correct


# low value for hardness means harder.
def get_hardness(output, target, is_multi):
    if not is_multi:
        # currently a binary association between correct classication => 0.8
        # we want it to be a softmax representation. if we instead take crossentropy loss of each individual cf target
        _, predicted = torch.max(output.data, 1)
        confidence = F.softmax(output, dim=1)
        hardness_scaler = np.where((predicted == target), 1, 0.1) # if correct, simply use confidence as measure of hardness
        # therefore if model can easily say yep this is object X, then confidence will be high. if it only just manages to identify
        # object X, confidence if lower
        # if object X is misclassified, hardness needs to be lower still.
        # assumes that it does not confidently misclassify.
        hardness = [(confidence[i][predicted[i]] * hardness_scaler[i]).item() for i in range(output.size(0))]
    else:
        output = torch.sigmoid(output.float()).detach()
        output[output>0.5] = 1
        output[output<=0.5] = 0
        confidence = F.softmax(output, dim=1)

        hardness_scaler = []
        hardness = []
        assert len(output) == len(target) # should both be equal to batch size
        for q in range(len(output)):
            assert len(output[q]) == len(target[q]) # should both be equal to num_classes eg 184
            correct_avg = (np.array(output[q]) == np.array(target[q])).sum() / len(output[q])
            if correct_avg > 0.5: # this could be another threshold we change, or have it == hardness threshold
                hardness_scaler.append(1)
            else:
                hardness_scaler.append(0.1)

            correct = np.where(np.array(output[q]) == np.array(target[q]))[0]
            hardness_value = [confidence[q][i] * 1 if i in correct else confidence[q][i] * 0.1 for i in range(len(output[q]))]
            hardness.append(sum(hardness_value) / len(output[q]))


        hardness_scaler = np.asarray(hardness_scaler)
        hardness = np.array(hardness)
        # raise AttributeError(output, target, hardness_scaler, hardness)

    return hardness, hardness_scaler
