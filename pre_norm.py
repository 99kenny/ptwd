import logging

import torch
import torch.nn as nn 

logging.basicConfig(level=logging.DEBUG, datefmt='%H:%M:%S', format='[%(levelname)s %(asctime)s : %(funcName)s] %(message)s')

class PreNorm(nn.Module):
    def __init__(self, patch_num):
        super().__init__()
        self.mean = 0.
        self.var = 0.
        self.features = list()
        self.patch_num = patch_num
    
    def set_mean_var(self, mean, var):
        self.mean = mean
        self.var = var
        
    @torch.no_grad()
    def forward(self, x, is_pre):
        if is_pre:
            # x = (batch, size, dim) 
            self.features.append(x)
            b, s, d = x.shape
            # Todo: 이렇게 말고 몇번쨰 forward인지만 저장했다가 곱하고 나누기 해주면 될듯?
            batch = torch.cat(self.features)
            
            self.mean = torch.mean(batch, dim=(0,2)) # (size,)
            self.var = torch.var(batch, dim=(0,2))   # (size,)
            
        return x
        