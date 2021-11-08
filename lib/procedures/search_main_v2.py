##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import os, sys, time, torch
from log_utils import AverageMeter, time_string
from utils import obtain_accuracy
from models import change_key
import numpy as np
import torch.nn.functional as F


def get_flop_loss(expected_flop, flop_cur, flop_need, flop_tolerant):
    expected_flop = torch.mean(expected_flop)

    if flop_cur < flop_need - flop_tolerant:  # Too Small FLOP
        loss = - torch.log(expected_flop)
    # elif flop_cur > flop_need + flop_tolerant: # Too Large FLOP
    elif flop_cur > flop_need:  # Too Large FLOP
        loss = torch.log(expected_flop)
    else:  # Required FLOP
        loss = None
    if loss is None:
        return 0, 0
    else:
        return loss, loss.item()


def search_train_v2(train_loader, valid_loader, network, criterion, scheduler, base_optimizer, arch_optimizer, optim_config,
                    extra_info, print_freq, logger, args):
    data_time, batch_time = AverageMeter(), AverageMeter()
    base_losses, arch_losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    arch_cls_losses, arch_flop_losses = AverageMeter(), AverageMeter()
    epoch_str, flop_need, flop_weight, flop_tolerant = extra_info['epoch-str'], extra_info['FLOP-exp'], extra_info[
        'FLOP-weight'], extra_info['FLOP-tolerant']

    network.train()
    logger.log('[Search] : {:}, FLOP-Require={:.2f} MB, FLOP-WEIGHT={:.2f}'.format(epoch_str, flop_need, flop_weight))
    end = time.time()
    network.apply(change_key('search_mode', 'search'))

    hardness = [None for i in range(len(train_loader))]
    correct = [None for i in range(len(train_loader))]
    batch_size = args.batch_size
    for step, ((base_inputs, base_targets), (arch_inputs, arch_targets)) in enumerate(zip(train_loader, valid_loader)):
        scheduler.update(None, 1.0 * step / len(train_loader))
        # calculate prediction and loss
        base_targets = base_targets.cuda(non_blocking=True)
        arch_targets = arch_targets.cuda(non_blocking=True)
        # measure data loading time
        data_time.update(time.time() - end)

        # update the weights
        base_optimizer.zero_grad()
        logits, expected_flop = network(base_inputs)
        base_loss = criterion(logits, base_targets)

        new_hardness, new_correct = get_hardness(logits.cpu(), base_targets.cpu(), False)
        hardness[(step * batch_size):(step * batch_size) + batch_size] = new_hardness  # assumes batch 1 takes idx 0-8, batch 2 takes 9-16, etc.
        correct[(step * batch_size):(step * batch_size) + batch_size] = new_correct

        base_loss.backward()
        base_optimizer.step()
        # record
        prec1, prec5 = obtain_accuracy(logits.data, base_targets.data, topk=(1, 5))
        base_losses.update(base_loss.item(), base_inputs.size(0))
        top1.update(prec1.item(), base_inputs.size(0))
        top5.update(prec5.item(), base_inputs.size(0))

        # update the architecture
        arch_optimizer.zero_grad()
        print(base_inputs.shape, arch_inputs.shape, "grep here")
        logits, expected_flop = network(arch_inputs)
        flop_cur = network.module.get_flop('genotype', None, None)
        flop_loss, flop_loss_scale = get_flop_loss(expected_flop, flop_cur, flop_need, flop_tolerant)
        acls_loss = criterion(logits, arch_targets)
        arch_loss = acls_loss + flop_loss * flop_weight
        arch_loss.backward()
        arch_optimizer.step()

        # record
        arch_losses.update(arch_loss.item(), arch_inputs.size(0))
        arch_flop_losses.update(flop_loss_scale, arch_inputs.size(0))
        arch_cls_losses.update(acls_loss.item(), arch_inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % print_freq == 0 or (step + 1) == len(train_loader):
            Sstr = '**TRAIN** ' + time_string() + ' [{:}][{:03d}/{:03d}]'.format(epoch_str, step, len(train_loader))
            Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(
                batch_time=batch_time, data_time=data_time)
            Lstr = 'Base-Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})'.format(
                loss=base_losses, top1=top1, top5=top5)
            Vstr = 'Acls-loss {aloss.val:.3f} ({aloss.avg:.3f}) FLOP-Loss {floss.val:.3f} ({floss.avg:.3f}) Arch-Loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                aloss=arch_cls_losses, floss=arch_flop_losses, loss=arch_losses)
            logger.log(Sstr + ' ' + Tstr + ' ' + Lstr + ' ' + Vstr)
            # num_bytes = torch.cuda.max_memory_allocated( next(network.parameters()).device ) * 1.0
            # logger.log(Sstr + ' ' + Tstr + ' ' + Lstr + ' ' + Vstr + ' GPU={:.2f}MB'.format(num_bytes/1e6))
            # Istr = 'Bsz={:} Asz={:}'.format(list(base_inputs.size()), list(arch_inputs.size()))
            # logger.log(Sstr + ' ' + Tstr + ' ' + Lstr + ' ' + Vstr + ' ' + Istr)
            # print(network.module.get_arch_info())
            # print(network.module.width_attentions[0])
            # print(network.module.width_attentions[1])

    logger.log(
        ' **TRAIN** Prec@1 {top1.avg:.2f} Prec@5 {top5.avg:.2f} Error@1 {error1:.2f} Error@5 {error5:.2f} Base-Loss:{baseloss:.3f}, Arch-Loss={archloss:.3f}'.format(
            top1=top1, top5=top5, error1=100 - top1.avg, error5=100 - top5.avg, baseloss=base_losses.avg,
            archloss=arch_losses.avg))
    return base_losses.avg, arch_losses.avg, top1.avg, top5.avg, hardness, correct


# low value for hardness means harder.
def get_hardness(output, target, is_multi):
    if not is_multi:
        # currently a binary association between correct classication => 0.8
        # we want it to be a softmax representation. if we instead take crossentropy loss of each individual cf target
        _, predicted = torch.max(output.data, 1)
        confidence = F.softmax(output, dim=1)
        try:
            hardness_scaler = np.where((predicted == target), 1, 0.1) # if correct, simply use confidence as measure of hardness
        except RuntimeError:
            raise AttributeError(output.shape, target.shape, predicted.shape, confidence.shape)
        # therefore if model can easily say yep this is object X, then confidence will be high. if it only just manages to identify
        # object X, confidence if lower
        # if object X is misclassified, hardness needs to be lower still.
        # assumes that it does not confidently misclassify.
        # raise AttributeError(output.shape, target.shape, predicted.shape, confidence.shape, hardness_scaler)
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
