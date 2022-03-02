from __future__ import print_function, division

import sys
import time
from collections import OrderedDict

import torch
from copy import deepcopy as cp

from .util import AverageMeter, accuracy


def train_distill(epoch, train_loader, held_loader, module_list, criterion_list, s_optimizer, t_optimizer, opt):
    """One epoch distillation"""

    criterion_cls = criterion_list[0]
    criterion_kd = criterion_list[1]

    s_model = module_list[0]
    t_model = module_list[-1]

    batch_time = AverageMeter()

    assume_losses = AverageMeter()
    assume_top1 = AverageMeter()
    assume_top5 = AverageMeter()

    held_losses = AverageMeter()

    real_losses = AverageMeter()
    real_top1 = AverageMeter()
    real_top5 = AverageMeter()

    end = time.time()

    total_steps_one_epoch = len(train_loader)
    batches_buffer = []
    round_counter = 0

    for d_idx, d_data in enumerate(train_loader):

        batches_buffer.append((d_idx, d_data))

        if (d_idx + 1) % opt.num_meta_batches != 0 and (d_idx + 1) != total_steps_one_epoch:
            continue

        #########################################
        #           Step 1: Assume S'           #
        #########################################

        # Time machine!
        fast_weights = OrderedDict((name, param) for (name, param) in s_model.named_parameters())
        s_model_backup_state_dict, s_optimizer_backup_state_dict = cp(s_model.state_dict()), cp(s_optimizer.state_dict())

        s_model.train()
        t_model.eval()

        for idx, data in batches_buffer:

            input, target = data

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            logit_s = s_model(input, params=None if idx == 0 else fast_weights)
            logit_t = t_model(input)

            assume_loss_cls = criterion_cls(logit_s, target)
            assume_loss_kd = criterion_kd(logit_s, logit_t)

            assume_loss = opt.alpha * assume_loss_kd + (1 - opt.alpha) * assume_loss_cls

            acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
            assume_losses.update(assume_loss.item(), input.size(0))
            assume_top1.update(acc1[0], input.size(0))
            assume_top5.update(acc5[0], input.size(0))

            grads = torch.autograd.grad(assume_loss, s_model.parameters() if idx == 0 else fast_weights.values(),
                                        create_graph=True, retain_graph=True)

            fast_weights = OrderedDict(
                (name, param - opt.assume_s_step_size * grad) for ((name, param), grad) in
                zip(fast_weights.items(), grads))

        #########################################
        #  Step 2: Train T with S' on HELD set  #
        #########################################

        s_prime_loss = None

        t_model.train()
        held_batch_num = 0

        for idx, data in enumerate(held_loader):
            input, target = data

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            logit_s_prime = s_model(input, params=fast_weights)
            s_prime_step_loss = criterion_cls(logit_s_prime, target)

            if s_prime_loss is None:
                s_prime_loss = s_prime_step_loss
            else:
                s_prime_loss += s_prime_step_loss

            held_batch_num += 1

        s_prime_loss /= held_batch_num
        t_grads = torch.autograd.grad(s_prime_loss, t_model.parameters())

        for p, gr in zip(t_model.parameters(), t_grads):
            p.grad = gr

        held_losses.update(s_prime_loss.item(), 1)

        t_optimizer.step()

        # Manual zero_grad
        for p in t_model.parameters():
            p.grad = None

        for p in s_model.parameters():
            p.grad = None

        del t_grads
        del grads
        del fast_weights

        #########################################
        #        Step 3: Actually update S      #
        #########################################

        # We use the Time Machine!
        s_model.load_state_dict(s_model_backup_state_dict)
        s_optimizer.load_state_dict(s_optimizer_backup_state_dict)

        del s_model_backup_state_dict, s_optimizer_backup_state_dict

        s_model.train()
        t_model.eval()

        for idx, data in batches_buffer:

            input, target = data

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            logit_s = s_model(input)
            with torch.no_grad():
                logit_t = t_model(input)

            real_loss_cls = criterion_cls(logit_s, target)
            real_loss_kd = criterion_kd(logit_s, logit_t)

            real_loss = opt.alpha * real_loss_kd + (1 - opt.alpha) * real_loss_cls

            acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
            real_losses.update(real_loss.item(), input.size(0))
            real_top1.update(acc1[0], input.size(0))
            real_top5.update(acc5[0], input.size(0))

            real_loss.backward()

            s_optimizer.step()
            s_optimizer.zero_grad()

        round_counter += 1
        batches_buffer = []

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if round_counter % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Real_Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                loss=real_losses, top1=real_top1, top5=real_top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=real_top1, top5=real_top5))

    return real_top1.avg, real_losses.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg
