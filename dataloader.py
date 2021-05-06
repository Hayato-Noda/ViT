#!/usr/bin/env python
# coding: utf-8

import random
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

class set_transforms():
  def __init__(self, size, batch_size, name='Cifar10'):
    super().__init__()
    self.batch_size = batch_size
    self.size = size
  

    if name == 'Cifar10':
      self._Cifar10download()
    else:
      self._Imagenet1kdownload()


  def _Cifar10download(self):
    train_transforms = transforms.Compose([
        transforms.Resize((self.size, self.size)),
        transforms.RandomResizedCrop(self.size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    val_transforms = transforms.Compose([
        transforms.Resize((self.size, self.size)),
        transforms.RandomResizedCrop(self.size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    test_transforms = transforms.Compose([
        transforms.Resize((self.size, self.size)),
        transforms.RandomResizedCrop(self.size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    self.trainset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=False, 
        transform=train_transforms)
    
    self.testset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=False,
        transform=val_transforms)
    
    return self

  def _Imagenet1kdownlaod():
    train_transforms = transforms.Compose([
        transforms.Resize((self.size, self.size)),
        transforms.RandomResizedCrop(self.size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    val_transforms = transforms.Compose([
        transforms.Resize((self.size, self.size)),
        transforms.RandomResizedCrop(self.size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    test_transforms = transforms.Compose([
        transforms.Resize((self.size, self.size)),
        transforms.RandomResizedCrop(self.size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    
    self.trainset = datasets.ImageNet(
        root='./data',
        train=True,
        download=False, 
        transform=self.train_transforms)
    
    self.testset = datasets.ImageNet(
        root='./data',
        train=False,
        download=False,
        transform=self.val_transforms)
    
    return self

  def _set_transforms(self):
    trainloader = DataLoader(
        self.trainset,
        batch_size=self.batch_size,
        shuffle=True,
        num_workers=2)

    testloader = DataLoader(
        self.testset,
        batch_size=self.batch_size,
        shuffle=False,
        num_workers=2)
    
    return trainloader, testloader