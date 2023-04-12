import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import *
import heapq
from torchstat import stat
import copy
import matplotlib.pyplot as plt

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=19,
                    help='depth of the vgg')
parser.add_argument('--percent', type=float, default=0.5,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='./proned', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('--selected-layers', default=4, type=int,
                    help='path to save pruned model (default: none)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if not os.path.exists(args.save):
    os.makedirs(args.save)

model = vgg(dataset=args.dataset, depth=args.depth)
if args.cuda:
    model.cuda()

if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.model))

print(model)

#********************************pre-pruning*********************************#
def find_pruned(layer, compression_ratio):
    conv_weight = layer.weight.data.cpu().numpy()
    num_filters = conv_weight.shape[0]

    L1_norm = np.sum(np.abs(conv_weight),axis=(1,2,3))
    filter_ranks = np.argsort(L1_norm)

    if compression_ratio < 2:
        num_prune = int(num_filters * 0.3)
        for i in range(num_prune):
            idx = filter_ranks[i]
            conv_weight[idx] = 0
    elif compression_ratio >= 2 & compression_ratio < 5:
            idx = filter_ranks[i]
            conv_weight[idx] = 0

    layer.weight.data = torch.from_numpy(conv_weight).to(layer.weight.device)

    return num_prune, (num_filters-num_prune)



def test(model):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

def save_checkpoint(state, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))

selected_layers = args.selected_layers
compression_ratio = 1
accuracy_proned_list = []
model_select_layer = model
for epoch in range(3):
    accuracy_list = []
    index_feature = []
    for k, m in enumerate(model.feature):
        if k > 0:
            if isinstance(m, nn.Conv2d):
                model_select_layer.load_state_dict(checkpoint['state_dict'])
                m = model_select_layer.feature[k]
                weight_copy = m.weight.data.abs().clone()
                # The channel mark Mask map to be retained
                num_prune, num_nonprune = find_pruned(m, compression_ratio)
                print('layer index: {:d} \t Proned channel: {:d} \t remaining channel: {:d}'.
                    format(k, num_prune, num_nonprune))
                accuracy = test(model_select_layer)
                accuracy_list.append(accuracy)
                x = range(len(accuracy_list))
                plt.plot(x, accuracy_list, label='Iteration:'+ str(epoch) + 'time')
                index_feature.append(k)
    max_num_index_list = map(accuracy_list.index, heapq.nlargest(selected_layers,accuracy_list))
    prone_index = list(max_num_index_list)
    print(max_num_index_list)
    for i in prone_index:
        k = index_feature[i]
        m = model.feature[k]
        num_prune, num_nonprune = find_pruned(m, compression_ratio)
    accuracy_proned = test(model)
    accuracy_proned_list.append(accuracy_proned)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
    }, filepath=args.save)
    checkpoint = torch.load('./proned/checkpoint.pth.tar')

print(accuracy_proned_list)
stat(model, (3,32,32))

plt.xlabel("Layers")
plt.ylabel("Accuracy")
plt.show()
