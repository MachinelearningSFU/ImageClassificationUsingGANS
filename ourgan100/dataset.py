import torch
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
import numpy as np


def MnistLabel(class_num):
    raw_dataset = datasets.MNIST('../data', train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     # transforms.Normalize((0.1307,), (0.3081,))
                                 ]))
    class_tot = [0] * 10
    print(class_tot)
    data = []
    labels = []
    positive_tot = 0
    tot = 0
    perm = np.random.permutation(raw_dataset.__len__())
    # print(raw_dataset.__len__())
    for i in range(raw_dataset.__len__()):
        datum, label = raw_dataset.__getitem__(perm[i])
        # print(class_tot)
        # print(class_tot[label])
        if class_tot[label] < 10:
            data.append(datum.numpy())
            labels.append(label)
            class_tot[label] += 1
            tot += 1
            # print(tot)
            if tot >= (100):
                print("breaindnsd",tot)
                break

    # print(tot)
    # print(class_num)
    data_tensor = torch.FloatTensor(np.array(data))
    target_tensor = torch.LongTensor(np.array(labels))
    print(data_tensor.shape)

    dataset = TensorDataset(data_tensor.repeat(600, 1, 1, 1), target_tensor.repeat(600))
    print("asdaadadsadsadsad")
    print(len(dataset))
    return dataset


def MnistUnlabel():
    raw_dataset = datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor()
    ]))
    print(len(raw_dataset))
    return raw_dataset


def MnistTest():
    return datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor()
    ]))


def Cifarlabel(class_num):
    raw_dataset = datasets.CIFAR10('../data', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       # transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    class_tot = [0] * 10
    data = []
    labels = []
    positive_tot = 0
    tot = 0
    perm = np.random.permutation(raw_dataset.__len__())
    for i in range(raw_dataset.__len__()):
        datum, label = raw_dataset.__getitem__(perm[i])
        if class_tot[label] < class_num:
            data.append(datum.numpy())
            labels.append(label)
            class_tot[label] += 1
            tot += 1
            if tot >= 10 * class_num:
                break
    data_tensor = torch.FloatTensor(np.array(data))
    target_tensor = torch.LongTensor(np.array(labels))

    dataset = TensorDataset(data_tensor.repeat(500, 1, 1, 1), target_tensor.repeat(500))
    return dataset


def CifarUnlabel():
    raw_dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor()
    ]))
    return raw_dataset


def CifarTest():
    return datasets.CIFAR10('../data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor()
    ]))


if __name__ == '__main__':
    print(dir(MnistTest()))
