# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for l2p implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
import sys
import os
import argparse
import datetime
import random
import numpy as np
import time
import logging

import torch
from torch.utils.data import DataLoader 
import torch.backends.cudnn as cudnn
from torchvision.utils import make_grid, save_image
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchsummary import summary
from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from datasets import *
from engine import *
import models
import utils
from image_prompt_loss import ImagePromptLoss
from pre_norm import PreNorm
from deep_inversion_feature_hook import DeepInversionFeatureHooK

import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(levelname)s %(asctime)s : [%(name)s - %(funcName)s] %(message)s')
logger = logging.getLogger(__name__)

def main(args):
    # utils.init_distributed_mode(args)
    device = torch.device(args.device)
    logger.debug(f'using device {args.device}:{device}')
    logger.debug(f'args : {args}')
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True
        
    #data_loader, class_mask = build_continual_dataloader(args)
    data_loader, class_mask = build_dataloader(args)
    # nor
    
    print(f"Creating original model: {args.model}")
    original_model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        prompt_length=args.length,
        embedding_key=args.embedding_key,
        prompt_init=args.initializer,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
    )
    original_model.to(device)
    model.to(device)  
    
    # save initial images 
    if args.initializer == 'train' and args.prompt_type == 'ImagePrompt':
        logger.info('save initial prompt imgs...')
        save_image(make_grid(model.prompt.prompt, nrow=args.size), f'{args.output_dir}/{args.exp_name}/initial_prompts.jpg')
    
    if args.freeze:
        # all parameters are frozen for original vit model
        for p in original_model.parameters():
            p.requires_grad = False
        
        # freeze args.freeze[blocks, patch_embed, cls_token] parameters
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False

#    logger.debug(f'original_model : {summary(original_model)}')
    logger.debug(f'model : {summary(model)}')
    
    for n, p in model.named_parameters():
        if p.requires_grad == True:
            logger.debug(f'{n} : {p.shape}')

    if args.eval:
        logger.info('eval mode')
        if args.continual:
            logger.info('continual')
            acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
            # validate for each task
            for task_id in range(args.num_tasks):
                checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
                if os.path.exists(checkpoint_path):
                    logger.info('Loading checkpoint from: %s', checkpoint_path)
                    checkpoint = torch.load(checkpoint_path)
                    model.load_state_dict(checkpoint['model'])
                else:
                    logger.info('No checkpoint found at: %s', checkpoint_path)
                    return
                _ = evaluate_till_now(model, original_model, data_loader, device, 
                                                task_id, class_mask, acc_matrix, args,)
                
            return
        else:
            logger.info('single task')
            acc_matrix = np.zeros(1)
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/checkpoint.pth')
            if os.path.exists(checkpoint_path):
                logger.info('Loading checkpoint from: %s', checkpoint_path)
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model'])
            else:
                logger.info('No checkpoint found at: %s', checkpoint_path)
                return
            _ = evaluate(model, original_model, data_loader, device, acc_matrix, args,)
    
    
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.debug('number of params: %s', n_parameters)
    
    # if args.unscale_lr:
    #     global_batch_size = args.batch_size
    # else:
    #     global_batch_size = args.batch_size * args.world_size
    # args.lr = args.lr * global_batch_size / 256.0
    
    optimizer = create_optimizer(args, model)
    
    # if args.sched == 'cosine':
    #     lr_scheduler, _ = create_scheduler(args, optimizer)
    # elif args.sched == 'constant':
    #     lr_scheduler = None
    if args.sched == 'constant':
        lr_scheduler = None
    elif args.sched == 'cosine':
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    # Image prompt loss
    prompt_criterion = None
    if args.prompt_type == 'ImagePrompt':
        # pre norm
        # data selection 100
        sample_loader = data_loader[0]['train']
        sample_size = 100
        sample = torch.empty(0,3,224,224)
        logger.info('load sample %d images for prenorm', sample_size)
        # todo: fix this (memory issue)
        for input, target in sample_loader:
            batch_size = input.shape[0]
            if batch_size >= sample_size:
                sample = torch.cat([sample, input[:sample_size]], dim=0)
                break
            else:
                sample = torch.cat([sample, input], dim=0)
                sample_size -= batch_size
        logger.info('sample shape for mean, var: %s', sample.shape) 
        
        sample = sample.to(device)
        # save mean, var
        original_model.eval()
        logger.debug('start prenorm')
        original_model(sample, is_pre=True)
        logger.debug('end prenorm')
        
        prenorm_values = list()
        for module in original_model.modules():
            if isinstance(module, PreNorm):
                prenorm_values.append({'mean' : module.mean, 'var' : module.var})
                #logger.debug(f'original model running : {module.mean} , {module.var}')
        # prompt loss
        r_feature_layers = list()
        idx = 0
        for module in model.modules():
            if isinstance(module, PreNorm):            
                module.set_mean_var(mean=prenorm_values[idx]['mean'], var=prenorm_values[idx]['var'])
                #logger.debug(f'model running: {module.mean}, {module.var}')
                r_feature_layers.append(DeepInversionFeatureHooK(module))
                idx += 1
    
        prompt_criterion = ImagePromptLoss(r_feature_layers=r_feature_layers, alpha_main=args.alpha_main, alpha_tv_l1=args.alpha_tv_l1, alpha_tv_l2=args.alpha_tv_l2, alpha_l2=args.alpha_l2, alpha_f=args.alpha_f)
    
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    if args.continual:
        train_and_evaluate_continual(model, original_model,
                    criterion, prompt_criterion, data_loader, optimizer, lr_scheduler,
                    device, class_mask, args)
    else:
        train_and_evaluate(model, original_model,
                                  criterion, prompt_criterion, data_loader, optimizer, lr_scheduler,
                                  device, args)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Image prompt')
    parser.add_argument('--continual', action='store_true', help='activate continual learning setting')
    parser.add_argument('--prompt_type', type=str, default='ImagePrompt')
    parser.add_argument('--exp_name', type=str)
    config = parser.parse_known_args()[-1][0]
    
    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_l2p':
        from configs.cifar100_l2p import get_args_parser
        config_parser = subparser.add_parser('cifar100_l2p', help='Split-CIFAR100 L2P configs')
    elif config == 'five_datasets_l2p':
        from configs.five_datasets_l2p import get_args_parser
        config_parser = subparser.add_parser('five_datasets_l2p', help='5-Datasets L2P configs')
    elif config == 'image_prompt':
        from configs.image_prompt import get_args_parser
        config_parser = subparser.add_parser('image_prompt', help='Image prompt configs')
    else:
        raise NotImplementedError
    
    get_args_parser(config_parser)

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir + '/' + args.exp_name).mkdir(parents=True, exist_ok=True)

    main(args)

    sys.exit(0)