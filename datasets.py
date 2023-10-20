# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for Simple Continual Learning datasets
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------

import random
import logging

import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms

from timm.data import create_transform

from continual_datasets.continual_datasets import *

import utils

logging.basicConfig(level=logging.DEBUG, datefmt='%H:%M:%S', format='[%(levelname)s %(asctime)s : %(funcName)s] %(message)s')
device = 4

class Lambda(transforms.Lambda):
    def __init__(self, lambd, nb_classes):
        super().__init__(lambd)
        self.nb_classes = nb_classes
    
    def __call__(self, img):
        return self.lambd(img, self.nb_classes)

def target_transform(x, nb_classes):
    return x + nb_classes

def build_dataloader(args):
    train_transform = build_transform(True, args)
    val_transform = build_transform(False, args)
    dataset_train, dataset_val = get_dataset(args.dataset, train_transform, val_transform, args.data_path)
    args.nb_classes = len(dataset_val.classes)
    
    dataloader = list()

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    dataloader.append({'train': data_loader_train, 'val': data_loader_val})
    class_mask = None
    return dataloader, class_mask 


def build_continual_dataloader(args):
    """build dataloader for continual learning from args.dataset 
    (Split-CIFAR100 or 5-datasets or specify dataset with ,)

    Args:
        args (namesapce): arg parser  

    Returns:
        list: dataloader - [{'train' : train_loader, 'val' : val_loader},...]
        list: class_mask - [[1,2,3],[4,5,6]]
    """
    dataloader = list()
    class_mask = list() if args.task_inc or args.train_mask else None
    
    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)
    # args (SPLIT CIFAR)
    if args.dataset.startswith('Split-'):
        dataset_train, dataset_val = get_dataset(args.dataset.replace('Split-',''), transform_train, transform_val, args.data_path)

        args.nb_classes = len(dataset_val.classes)
        
        splited_dataset, class_mask = split_single_dataset(dataset_train, dataset_val, args)
    # or 5-datasets or others 
    else:
        if args.dataset == '5-datasets':
            dataset_list = ['SVHN', 'MNIST', 'CIFAR10', 'NotMNIST', 'FashionMNIST']
        else:
            dataset_list = args.dataset.split(',')
        
        if args.shuffle:
            random.shuffle(dataset_list)
        print(dataset_list)
    
        args.nb_classes = 0
    # for each task i
    for i in range(args.num_tasks):
        if args.dataset.startswith('Split-'):
            dataset_train, dataset_val = splited_dataset[i]

        else:
            dataset_train, dataset_val = get_dataset(dataset_list[i], transform_train, transform_val, args.data_path)

            transform_target = Lambda(target_transform, args.nb_classes)
            # if not Split-CIFAR100 set targets [0,1,2], [0,1,2] -> [0,1,2], [3,4,5] 
            if class_mask is not None:
                class_mask.append([i + args.nb_classes for i in range(len(dataset_val.classes))])
                args.nb_classes += len(dataset_val.classes)

            if not args.task_inc:
                dataset_train.target_transform = transform_target
                dataset_val.target_transform = transform_target
        
        if args.distributed and utils.get_world_size() > 1:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()

            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        dataloader.append({'train': data_loader_train, 'val': data_loader_val})

    return dataloader, class_mask

def get_dataset(dataset, transform_train, transform_val, data_path,):
    if dataset == 'CIFAR100':
        dataset_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR100(data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'CIFAR10':
        dataset_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR10(data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'MNIST':
        dataset_train = MNIST_RGB(data_path, train=True, download=True, transform=transform_train)
        dataset_val = MNIST_RGB(data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'FashionMNIST':
        dataset_train = FashionMNIST(data_path, train=True, download=True, transform=transform_train)
        dataset_val = FashionMNIST(data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'SVHN':
        dataset_train = SVHN(data_path, split='train', download=True, transform=transform_train)
        dataset_val = SVHN(data_path, split='test', download=True, transform=transform_val)
    
    elif dataset == 'NotMNIST':
        dataset_train = NotMNIST(data_path, train=True, download=True, transform=transform_train)
        dataset_val = NotMNIST(data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'Flower102':
        dataset_train = Flowers102(data_path, split='train', download=True, transform=transform_train)
        dataset_val = Flowers102(data_path, split='test', download=True, transform=transform_val)
    
    elif dataset == 'Cars196':
        dataset_train = StanfordCars(data_path, split='train', download=True, transform=transform_train)
        dataset_val = StanfordCars(data_path, split='test', download=True, transform=transform_val)
        
    elif dataset == 'CUB200':
        dataset_train = CUB200(data_path, train=True, download=True, transform=transform_train).data
        dataset_val = CUB200(data_path, train=False, download=True, transform=transform_val).data
    
    elif dataset == 'Scene67':
        dataset_train = Scene67(data_path, train=True, download=True, transform=transform_train).data
        dataset_val = Scene67(data_path, train=False, download=True, transform=transform_val).data

    elif dataset == 'TinyImagenet':
        dataset_train = TinyImagenet(data_path, train=True, download=True, transform=transform_train).data
        dataset_val = TinyImagenet(data_path, train=False, download=True, transform=transform_val).data
        
    elif dataset == 'Imagenet-R':
        dataset_train = Imagenet_R(data_path, train=True, download=True, transform=transform_train).data
        dataset_val = Imagenet_R(data_path, train=False, download=True, transform=transform_val).data
    
    else:
        raise ValueError('Dataset {} not found.'.format(dataset))
    
    return dataset_train, dataset_val

def split_single_dataset(dataset_train, dataset_val, args):
    '''Splits datset into (num_classes // args.num_tasks) classes per task
    Returns:
    list:split_datasets [[task1_train, task1_test], [task2_train, task2_test], ... ]
    list:mask [[1,4,5,...], [2,9,3,...]] list of targets for each task 
    '''
    nb_classes = len(dataset_val.classes)
    assert nb_classes % args.num_tasks == 0
    classes_per_task = nb_classes // args.num_tasks
    
    labels = [i for i in range(nb_classes)]
    
    split_datasets = list()
    mask = list()

    if args.shuffle:
        random.shuffle(labels)
    # split task  
    for _ in range(args.num_tasks):
        train_split_indices = []
        test_split_indices = []
        # current task
        scope = labels[:classes_per_task]
        # remaining tasks
        labels = labels[classes_per_task:]

        mask.append(scope)
        # current task index
        for k in range(len(dataset_train.targets)):
            if int(dataset_train.targets[k]) in scope:
                train_split_indices.append(k)
        
        for h in range(len(dataset_val.targets)):
            if int(dataset_val.targets[h]) in scope:
                test_split_indices.append(h)
        # subset datasets
        subset_train, subset_val =  Subset(dataset_train, train_split_indices), Subset(dataset_val, test_split_indices)
        
        split_datasets.append([subset_train, subset_val])
    
    return split_datasets, mask

def get_images(num, name='CIFAR10', data_path='./local_datasets'):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset_train, dataset_val = get_dataset(name, transform, transform, data_path)
    idx = torch.randint(0,len(dataset_train),(num,))
    c,h,w = dataset_train.__getitem__(0)[0].shape
    imgs = torch.empty((0,c,h,w))
    logging.debug(f'imgs shape : {imgs.shape}')
    for i in idx:
        imgs = torch.cat([torch.unsqueeze(dataset_train.__getitem__(i)[0],0), imgs],dim=0)
    imgs.to(device)
    
    return imgs
    
    
def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    
    return transforms.Compose(t)