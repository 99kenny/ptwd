import argparse
import json
import os
from torch.utils.tensorboard import SummaryWriter

def main(args):
    dir = args.dir
    writer = SummaryWriter(log_dir=dir)
    
    for filename in os.listdir(dir):
        f = os.path.join(dir, filename)
        # checking if it is a file
        if os.path.isfile(f):
            print(f)
        stat_dict = open(f, 'r')
        res = json.loads(stat_dict)
        
        writer.add_scalar('train_loss', res['train_Loss'],res['epoch'])
        writer.add_scalar('train_Prompt_Loss', res['train_Prompt_Loss'],res['epoch'])
        writer.add_scalar('LR', res['train_Lr'],res['epoch'])
        writer.add_scalar('train_acc', res['train_Acc@1'],res['epoch'])
        writer.add_scalar('test_loss', res['test_Loss'],res['epoch'])
        writer.add_scalar('test_Prompt_Loss', res['test_Prompt_Loss'],res['epoch'])
        writer.add_scalar('test_acc', res['test_Acc@1'],res['epoch'])
        
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--dir', type=str, help='summary directory')