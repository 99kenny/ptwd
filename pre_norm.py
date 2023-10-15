import torch
import torch.nn as nn 

class PreNorm(nn.Module):
    def __init__(self, patch_num):
        super().__init__()
        self.mean = 0.
        self.var = 0.
        self.patch_num = patch_num
    
    def forward(self, x, is_pre):
        if is_pre:
            # x = (batch, size, dim) 
            stats = x[:,self.patch_num+1:-1,:]
            self.mean = torch.mean(stats, dim=(0,2)) # (size,)
            self.var = torch.var(stats, dim=(0,2))   # (size,)
            
        return x
        