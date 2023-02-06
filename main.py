'''Train CIFAR10 with PyTorch.'''
from sklearn import datasets
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from sklearn.metrics import confusion_matrix

from madrys import MadrysLoss
import pickle

import os
import argparse
import numpy as np
from PIL import Image
import random

from models import *
from util import setup_logger, progress_bar
from models.vit import ViT
from models.MLP import MLP
from augmentations import *
from poison_loaders import *



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--poison_type', default=None, help='poison type')
parser.add_argument('--poison_path', default=None, help='path to the folder of poisoned images')
parser.add_argument('--poison_rate', default=1, type=float, help='poison rate (by a random seed)')

parser.add_argument('--grayscale', default=False, type=bool, help='grayscale compression')
parser.add_argument('--jpeg', default=None, type=int, help='JPEG quality factor')
parser.add_argument('--bdr', default=None, type=int, help='bit depth')
parser.add_argument('--AT', default=False, type=bool, help='PGD Adversarial training')
parser.add_argument('--AT_eps', default=0.031, type=float, help='poison_rate')
parser.add_argument('--TrainAUG', default='', help='Train augmentations')
parser.add_argument('--lowpass', default='', help='filtering')
parser.add_argument('--ISS_both_train_test', default=False)
parser.add_argument('--indices_path', default=None)
parser.add_argument('--adaptive_path', default=None)

parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--net', default='resnet18', help='models to train')

parser.add_argument('--exp_path', default='../EXPERIMENTS/TEMP/', help='exp_path')
parser.add_argument('--progress_bar_show', default=False, type=bool)


args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


if args.indices_path is not None:
    non_pois_np = np.load(args.indices)
else:
    non_pois_np = None

if not os.path.exists(args.exp_path):
    os.makedirs(args.exp_path)

log_file_path = os.path.join(args.exp_path, args.poison_type)
logger = setup_logger(name=args.poison_type, log_file=log_file_path + ".log")
logger.info("PyTorch Version: %s" % (torch.__version__))
logger.info('Poisons are: %s', args.poison_type)
logger.info('Poisons are at path: %s', args.poison_path)

logger.info('Mixup / Cutout / CutMix:  %s', str(args.TrainAUG))
logger.info('Grayscale compression for both train and test:  %s', str(args.grayscale))
logger.info('Bit Depth Reduction:  %s', str(args.bdr))
logger.info('JPEG compression quality:  %s', str(args.jpeg))
logger.info('Lowpass:  %s', str(args.lowpass))

logger.info('Training on:  %s', str(args.net))
logger.info('AT:  %s' 'with epsilon %s', str(args.AT), str(args.AT_eps))
logger.info('both train and test:  %s', str(args.ISS_both_train_test))
logger.info('poison rate:  %s', str(args.poison_rate))
logger.info('fixed indices at:  %s', str(args.indices_path))


# Data
print('==> Preparing data augmentation')

transform_train = aug_train(args.jpeg, args.grayscale, args.bdr, args.TrainAUG, args.lowpass)
transform_test = aug_test(args.ISS_both_train_test, args.jpeg, args.grayscale, args.bdr)

logger.info("Training transformation %s" % (transform_train))
logger.info("Test transformation %s" % (transform_test))


if args.poison_type == 'CLEAN':
    trainset = ST_load(T=transform_train, poison_rate=args.poison_rate, non_poison_indices=non_pois_np)
else:
    trainset = folder_load(path = args.poison_path, T=transform_train, poison_rate=args.poison_rate, non_poison_indices=non_pois_np)
testset = torchvision.datasets.CIFAR10(root='~/data', train=False, download=True, transform=transform_test)


if 'mixup' in args.TrainAUG:
    trainset = MixUp(trainset, num_class=10)
elif 'cutmix' in args.TrainAUG:
    trainset = CutMix(trainset, num_class=10)
else:
    pass


trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=4)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=4)

print('==> Building model..')

if args.net == 'resnet18':
    net = ResNet18(num_classes=10)
elif args.net == 'resnet34':
    net = ResNet34(num_classes=10)
elif args.net == 'vgg19':
    net = VGG('VGG19', num_classes=10)
elif args.net == 'densenet121':
    net = DenseNet121(num_classes=10)
elif args.net == 'mobilenetv2':
    net = MobileNetV2(num_classes=10)
elif args.net == 'mlp':
    net = MLP()    
elif args.net == 'vit':
    net = ViT(image_size = 32, patch_size = 4,
    num_classes = 10,
    dim = 512,
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1)
else:
    raise NotImplementedError

net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if 'mixup' in args.TrainAUG:
    criterion = cross_entropy
elif 'cutmix' in args.TrainAUG:
    criterion = CutMixCrossEntropyLoss()
else:
    criterion = nn.CrossEntropyLoss()


optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)

if args.AT:
    epochs = 100
else:
    epochs = 60
    
if args.net == 'vit':
    epochs = 200

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

best_acc = 0 
start_epoch = 0
progress_bar_show =False



# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        if args.AT:
            outputs, loss = MadrysLoss(cutmix=('cutmix' in args.TrainAUG), epsilon=args.AT_eps)(net,  inputs, targets, optimizer)
        else:
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if ('mixup' in args.TrainAUG) or ('cutmix' in args.TrainAUG):
            targets = torch.argmax(targets, dim=1)
        correct += predicted.eq(targets).sum().item()
        if progress_bar_show:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    logger.info("")
    logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)
    logger.info("Training Loss %.3f" % (train_loss/(batch_idx+1)))
    logger.info("Training Acc %.3f (%d/%d)" % (100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    outputlist = []
    targetlist = []


    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            outputlist.append(outputs.argmax(1).detach().cpu().numpy())
            targetlist.append(targets.detach().cpu().numpy())


            loss = nn.CrossEntropyLoss()(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if progress_bar_show:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        # if checkpoints are necessary
        # if not os.path.isdir(args.exp_path + '/checkpoint'):
        #     os.mkdir(args.exp_path + '/checkpoint')
        # torch.save(state, args.exp_path + '/checkpoint' + '/ckpt.pth')
        best_acc = acc

    cm = confusion_matrix(np.concatenate(targetlist, axis=0), np.concatenate(outputlist, axis=0))

    logger.info("")
    logger.info("="*20 + "Validation Epoch %d" % (epoch) + "="*20)
    logger.info("Validation Loss %.3f" % (test_loss/(batch_idx+1)))
    logger.info("Validation Acc  %.3f (%d/%d)" % (100.*correct/total, correct, total))
    logger.info("Best validation Acc %.3f" % (best_acc))
    logger.info(cm)
    


for epoch in range(start_epoch, start_epoch+epochs):
    train(epoch)
    test(epoch)
    scheduler.step()
