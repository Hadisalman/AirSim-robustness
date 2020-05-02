import argparse
import os
import sys
import shutil
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from IPython import embed
import numpy as np
from .datasets import ZippedDataset, splitTrainTest
from .utils import AverageMeter, accuracy, init_logfile, log, copy_code

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('outdir', type=str, help='path to output directory')
parser.add_argument('--img-size', default=32, type=int, metavar='N',
                    help='size of rgb image (assuming equal hight and width)')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

best_test_acc = 0

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def main():
    global args, best_test_acc
    args = parser.parse_args()

    # args.outdir = os.path.join(os.getenv('PT_OUTPUT_DIR', './'), args.outdir)
    copy_code(args.outdir)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    blockPrint()
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
    enablePrint()
    model.fc = torch.nn.Linear(512, 2)
 
    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_test_acc = checkpoint['best_test_acc']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    if '.zip' in args.data:
        dataset = ZippedDataset(args.data)
    else:
        dataset = datasets.ImageFolder(args.data)

    N_train = int(len(dataset)*0.8)
    N_test = len(dataset) - N_train

    transform_train = transforms.Compose([
                                        # transforms.RandomResizedCrop(224),
                                        transforms.Resize(args.img_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                        ])
    transform_test = transforms.Compose([
                                        transforms.Resize(args.img_size),
                                        transforms.ToTensor(),
                                        normalize
                                        ])
    train_data, test_data = splitTrainTest(dataset, (N_train, N_test), transform_train, transform_test, mode='random')

    print("Training data of size {}".format(len(train_data)))
    print("Test data of size {}".format(len(test_data)))
 
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    logfilename = os.path.join(args.outdir, 'log.txt')
    init_logfile(logfilename, "epoch\ttime\tlr\ttrainloss\ttrainacc\ttestloss\ttestacc")

    for epoch in range(args.start_epoch, args.epochs):
        before = time.time()
        # train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        test_loss, test_acc = validate(val_loader, model, criterion)
        after = time.time()
        scheduler.step(epoch)

        # Log results
        log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
            epoch, after - before,
            scheduler.get_lr()[0], train_loss, train_acc, test_loss, test_acc))

        # remember best test_acc and save checkpoint
        is_best = test_acc > best_test_acc
        best_test_acc = max(test_acc, best_test_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'test_acc': test_acc,
        }, is_best, outdir=args.outdir)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    output_class_distribution = [0,0]
    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        output_class_distribution[0] += (output.max(1)[1] == 0).sum().item()
        output_class_distribution[1] += (output.max(1)[1] == 1).sum().item()

        # measure accuracy and record loss
        prec1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0].item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
    print("Predicted distribution: ", output_class_distribution)
    return losses.avg, top1.avg

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    output_class_distribution = [0,0]

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            output_class_distribution[0] += (output.max(1)[1] == 0).sum().item()
            output_class_distribution[1] += (output.max(1)[1] == 1).sum().item()

            # measure accuracy and record loss
            prec1 = accuracy(output, target, topk=(1, ))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0].item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1))

        print(' * Accuracy {top1.avg:.3f} '
            .format(top1=top1))

        print("Predicted distribution: ", output_class_distribution)
        return losses.avg, top1.avg


def save_checkpoint(state, is_best, outdir='', filename='checkpoint.pth.tar'):
    filename = os.path.join(outdir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(outdir,'model_best.pth.tar'))



if __name__ == '__main__':
    main()