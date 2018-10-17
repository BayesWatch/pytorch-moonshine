''''Writing everything into one script..'''
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import json
import argparse
from torch.autograd import Variable
from models.wide_resnet import WideResNet, parse_options
import os
import imp
from tqdm import tqdm
from tensorboardX import SummaryWriter
import time
from funcs import *

os.mkdir('checkpoints/') if not os.path.isdir('checkpoints/') else None

parser = argparse.ArgumentParser(description='Student/teacher training')
parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet'], help='Choose between Cifar10/100/imagenet.')
parser.add_argument('mode', choices=['student','teacher'], type=str, help='Learn a teacher or a student')
parser.add_argument('--imagenet_loc', default='/disk/scratch_ssd/imagenet',type=str, help='folder containing imagenet train and val folders')
parser.add_argument('--cifar_loc', default='/disk/scratch/datasets/cifar',type=str, help='folder containing cifar train and val folders')
parser.add_argument('--workers', default=2, type=int, help='No. of data loading workers. Make this high for imagenet')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--GPU', default=None, type=str,help='GPU to use')
parser.add_argument('--student_checkpoint', '-s', default='wrn_40_2_student_KT',type=str, help='checkpoint to save/load student')
parser.add_argument('--teacher_checkpoint', '-t', default='wrn_40_2_T',type=str, help='checkpoint to load in teacher')

#network stuff
parser.add_argument('--wrn_depth', default=40, type=int, help='depth for WRN')
parser.add_argument('--wrn_width', default=2, type=float, help='width for WRN')
parser.add_argument('--module', default=None, type=str, help='path to file containing custom Conv and maybe Block module definitions')
parser.add_argument('--blocktype', default='Basic',type=str, help='blocktype used if specify a --conv')
parser.add_argument('--conv',
                    choices=['Conv','ConvB2','ConvB4','ConvB8','ConvB16','DConv', 'ACDC',
                             'Conv2x2','DConvB2','DConvB4','DConvB8','DConvB16','DConv3D','DConvG2','DConvG4','DConvG8','DConvG16'
                        ,'custom','DConvA2','DConvA4','DConvA8','DConvA16','G2B2','G2B4','G4B2','G4B4','G8B2','G8B4','G16B2','G16B4','A2B2','A4B2','A8B2','A16B2'],
                    default=None, type=str, help='Conv type')
parser.add_argument('--AT_split', default=1, type=int, help='group splitting for AT loss')

#learning stuff
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float, help='learning rate decay')
parser.add_argument('--temperature', default=4, type=float, help='temp for KD')
parser.add_argument('--alpha', default=0.0, type=float, help='alpha for KD')
parser.add_argument('--aux_loss', default='AT', type=str, help='AT or SE loss')
parser.add_argument('--beta', default=1e3, type=float, help='beta for AT')
parser.add_argument('--epoch_step', default='[60,120,160]', type=str,
                    help='json list with epochs to drop lr on')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--print_freq', default=10, type=int, help="print stats frequency")
parser.add_argument('--batch_size', default=128, type=int,
                    help='minibatch size')
parser.add_argument('--weightDecay', default=0.0005, type=float)

args = parser.parse_args()

writer = SummaryWriter()


def create_optimizer(lr,net):
    print('creating optimizer with lr = %0.5f' % lr)
    return torch.optim.SGD(net.parameters(), lr, 0.9, weight_decay=args.weightDecay)


def train_teacher(net):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.train()

    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs, _ = net(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        err1 = 100. - prec1
        err5 = 100. - prec5
        losses.update(loss.item(), inputs.size(0))
        top1.update(err1[0], inputs.size(0))
        top5.update(err5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Error@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, batch_idx, len(trainloader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    writer.add_scalar('train_loss', losses.avg, epoch)
    writer.add_scalar('train_top1', top1.avg, epoch)
    writer.add_scalar('train_top5', top5.avg, epoch)

    train_losses.append(losses.avg)
    train_errors.append(top1.avg)


def train_student(net, teach):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.train()
    teach.eval()

    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = Variable(inputs.cuda())
        targets = Variable(targets.cuda())
        outputs_student, ints_student = net(inputs)
        outputs_teacher, ints_teacher = teach(inputs)

        # If alpha is 0 then this loss is just a cross entropy.
        loss = distillation(outputs_student, outputs_teacher, targets, args.temperature, args.alpha)

        #Add an attention tranfer loss for each intermediate. Let's assume the default is three (as in the original
        #paper) and adjust the beta term accordingly.

        adjusted_beta = (args.beta*3)/len(ints_student)
        for i in range(len(ints_student)):
            loss += adjusted_beta * aux_loss(ints_student[i], ints_teacher[i])

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs_student.data, targets.data, topk=(1, 5))
        err1 = 100. - prec1
        err5 = 100. - prec5
        losses.update(loss.item(), inputs.size(0))
        top1.update(err1[0], inputs.size(0))
        top5.update(err5[0], inputs.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Error@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, batch_idx, len(trainloader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    writer.add_scalar('train_loss', losses.avg, epoch)
    writer.add_scalar('train_top1', top1.avg, epoch)
    writer.add_scalar('train_top5', top5.avg, epoch)

    train_losses.append(losses.avg)
    train_errors.append(top1.avg)


def validate(net, checkpoint=None):

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    net.eval()
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(valloader):

        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs, _ = net(inputs)
        if isinstance(outputs,tuple):
            outputs = outputs[0]

        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        err1 = 100. - prec1
        err5 = 100. - prec5

        losses.update(loss.item(), inputs.size(0))
        top1.update(err1[0], inputs.size(0))
        top5.update(err5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            print('validate: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Error@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                batch_idx, len(valloader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print(' * Error@1 {top1.avg:.3f} Error@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))



    val_losses.append(losses.avg)
    val_errors.append(top1.avg)


    if checkpoint:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
            'width': args.wrn_width,
            'depth': args.wrn_depth,
            'conv': args.conv,
            'blocktype': args.blocktype,
            'module': args.module,
            'train_losses': train_losses,
            'train_errors': train_errors,
            'val_losses': val_losses,
            'val_errors': val_errors,
        }
        print('SAVED!')
        torch.save(state, 'checkpoints/%s.t7' % checkpoint)



def what_conv_block(conv, blocktype, module):
    if conv is not None:
        Conv, Block = parse_options(conv, blocktype)
    elif module is not None:
        conv_module = imp.new_module('conv')
        with open(module, 'r') as f:
            exec(f.read(), conv_module.__dict__)
        Conv = conv_module.Conv
        try:
            Block = conv_module.Block
        except AttributeError:
            # if the module doesn't implement a custom block,
            # use default option
            _, Block = parse_options('Conv', args.blocktype)
    else:
        raise ValueError("You must specify either an existing conv option, or supply your own module to import")
    return Conv, Block

if __name__ == '__main__':
    # Stuff happens from here:
    Conv, Block = what_conv_block(args.conv, args.blocktype, args.module)

    if args.aux_loss == 'AT':
        aux_loss = at_loss
    elif args.aux_loss == 'SE':
        aux_loss = se_loss

    print(vars(args))
    if args.GPU is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

    use_cuda = torch.cuda.is_available()
    assert use_cuda, 'Error: No CUDA!'

    val_losses = []
    train_losses = []
    val_errors = []
    train_errors = []

    best_acc = 0
    start_epoch = 0
    epoch_step = json.loads(args.epoch_step)

    # Data and loaders
    print('==> Preparing data..')
    if args.dataset == 'cifar10':
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_validate = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root=args.cifar_loc,
                                                train=True, download=False, transform=transform_train)
        valset = torchvision.datasets.CIFAR10(root=args.cifar_loc,
                                               train=False, download=False, transform=transform_validate)
    elif args.dataset == 'cifar100':
        num_classes = 100
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
        ])
        transform_validate = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
        ])
        trainset = torchvision.datasets.CIFAR100(root=args.cifar_loc,
                                                train=True, download=True, transform=transform_train)
        validateset = torchvision.datasets.CIFAR100(root=args.cifar_loc,
                                               train=False, download=True, transform=transform_validate)

    elif args.dataset == 'imagenet':
        num_classes = 1000
        traindir = os.path.join(args.imagenet_loc, 'train')
        valdir = os.path.join(args.imagenet_loc, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_validate = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        trainset = torchvision.datasets.ImageFolder(traindir, transform_train)
        valset = torchvision.datasets.ImageFolder(valdir, transform_validate)



    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.workers,
                                              pin_memory = True if args.dataset == 'imagenet' else False)
    valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True if args.dataset == 'imagenet' else False)

    criterion = nn.CrossEntropyLoss()

    def load_network(loc):
        net_checkpoint = torch.load(loc)
        start_epoch = net_checkpoint['epoch']
        SavedConv, SavedBlock = what_conv_block(net_checkpoint['conv'],
                net_checkpoint['blocktype'], net_checkpoint['module'])
        net = WideResNet(args.wrn_depth, args.wrn_width, SavedConv, SavedBlock, num_classes=num_classes, dropRate=0).cuda()
        net.load_state_dict(net_checkpoint['net'])
        return net, start_epoch

    if args.mode == 'teacher':

        if args.resume:
            print('Mode Teacher: Loading teacher and continuing training...')
            teach, start_epoch = load_network('checkpoints/%s.t7' % args.teacher_checkpoint)
        else:
            print('Mode Teacher: Making a teacher network from scratch and training it...')
            teach = WideResNet(args.wrn_depth, args.wrn_width, Conv, Block, num_classes=num_classes, dropRate=0).cuda()


        get_no_params(teach)
        optimizer = optim.SGD(teach.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weightDecay)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=epoch_step, gamma=args.lr_decay_ratio)

        # Decay the learning rate depending on the epoch
        for e in range(0,start_epoch):
            scheduler.step()

        for epoch in tqdm(range(start_epoch, args.epochs)):
            scheduler.step()
            print('Teacher Epoch %d:' % epoch)
            print('Learning rate is %s' % [v['lr'] for v in optimizer.param_groups][0])
            writer.add_scalar('learning_rate', [v['lr'] for v in optimizer.param_groups][0], epoch)
            train_teacher(teach)
            validate(teach, args.teacher_checkpoint)


    elif args.mode == 'student':
        print('Mode Student: First, load a teacher network and convert for (optional) attention transfer')
        teach, _ = load_network('checkpoints/%s.t7' % args.teacher_checkpoint)
        # Very important to explicitly say we require no gradients for the teacher network
        for param in teach.parameters():
            param.requires_grad = False
        validate(teach)
        val_losses, val_errors = [], [] # or we'd save the teacher's error as the first entry

        if args.resume:
            print('Mode Student: Loading student and continuing training...')
            student, start_epoch = load_network('checkpoints/%s.t7' % args.student_checkpoint)
        else:
            print('Mode Student: Making a student network from scratch and training it...')
            student = WideResNet(args.wrn_depth, args.wrn_width, Conv, Block,
                    num_classes=num_classes, dropRate=0,
                    s=args.AT_split).cuda()

        optimizer = optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weightDecay)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=epoch_step, gamma=args.lr_decay_ratio)

        # Decay the learning rate depending on the epoch
        for e in range(0, start_epoch):
            scheduler.step()

        for epoch in tqdm(range(start_epoch, args.epochs)):
            scheduler.step()

            print('Student Epoch %d:' % epoch)
            print('Learning rate is %s' % [v['lr'] for v in optimizer.param_groups][0])
            writer.add_scalar('learning_rate', [v['lr'] for v in optimizer.param_groups][0], epoch)

            train_student(student, teach)
            validate(student, args.student_checkpoint)

