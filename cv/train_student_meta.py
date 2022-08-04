"""
the general training framework
"""

from __future__ import print_function

import os
import argparse
import socket
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn


from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser

from dataset.meta_cifar100 import get_cifar100_dataloaders

from helper.util import adjust_learning_rate

from distiller_zoo.KD import CustomDistillKL
from distiller_zoo.MSE import MSEWithTemperature
from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss
from crd.criterion import CRDLoss

from helper.meta_loops import train_distill as train, validate
from helper.pretrain import init


def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # optimization
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--teacher_lr', type=float, default=0.05, help='teacher learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--loss_type', type=str, choices=['mse', 'kl'])

    # held set
    parser.add_argument('--held_size', type=int, help="the size of held set")
    parser.add_argument('--num_held_samples', type=int, help="num of held samples used for one teacher update")
    parser.add_argument('--num_meta_batches', type=int, default=1, help="num of meta batches used for one teacher update")
    parser.add_argument('--assume_s_step_size', type=float, default=0.05, help="assume student grad update lr")

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19'])
    # TODO: Add more support
    # 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2', 'ResNet50', 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    # distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.lr = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/student_model'
        opt.tb_path = '/path/to/my/student_tensorboards'
    else:
        opt.model_path = './save/student_model'
        opt.tb_path = './save/student_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    opt.model_name = 'S:{}_T:{}_{}_{}_a:{}_{}'.format(opt.model_s, opt.model_t, opt.dataset, 'mlkd',
                                                                opt.alpha, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path)['model'])
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model'])
    print('==> done')
    return model


def main():
    best_acc = 0

    opt = parse_option()

    # tensorboard logger
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, held_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                         num_workers=opt.num_workers,
                                                                         held_size=opt.held_size,
                                                                         num_held_samples=opt.num_held_samples,
                                                                         )
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model_t = load_teacher(opt.path_t, n_cls)
    model_s = model_dict[opt.model_s](num_classes=n_cls)

    data = torch.randn(2, 3, 32, 32)
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()

    if opt.loss_type == 'mse':
        criterion_kd = MSEWithTemperature(T=opt.kd_T)
    elif opt.loss_type == 'kl':
        criterion_kd = CustomDistillKL(T=opt.kd_T)
    else:
        raise NotImplementedError()

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    # optimizer
    s_optimizer = optim.SGD(model_s.parameters(),
                            lr=opt.lr,
                            momentum=opt.momentum,
                            weight_decay=opt.weight_decay)
    t_optimizer = optim.SGD(model_t.parameters(),
                            lr=opt.teacher_lr,
                            momentum=opt.momentum,
                            weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # validate teacher accuracy
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc)
    
    #Fixed for first 150 epochs
    for param_group in t_optimizer.param_groups:
        param_group['lr'] = 0
    # routine
    for epoch in range(1, opt.epochs + 1):

        if epoch == 150:
            for param_group in t_optimizer.param_groups:
                param_group['lr'] = opt.teacher_lr
                opt.assume_s_step_size *= opt.lr_decay_rate
        if epoch == 180:
            for param_group in t_optimizer.param_groups:
                param_group['lr'] *= opt.lr_decay_rate
            opt.assume_s_step_size *= opt.lr_decay_rate
        if epoch == 210:
            for param_group in t_optimizer.param_groups:
                param_group['lr'] *= opt.lr_decay_rate
            opt.assume_s_step_size *= opt.lr_decay_rate
        adjust_learning_rate(epoch, opt, s_optimizer)
        # adjust_learning_rate(epoch, opt, t_optimizer)
        
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, held_loader, module_list, criterion_list, s_optimizer, t_optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'accuracy': test_acc,
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch. 
    print('best accuracy:', best_acc)

    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
