import torch
import torch.nn as nn 

class PreNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = 0.
        self.var = 0.
    
    def forward(self, x, is_pre):
        if is_pre:
            # x = (batch, size, dim) 
            self.mean = torch.mean(x, dim=(0,2)) # (size,)
            self.var = torch.var(x, dim=(0,2))   # (size,)
            
        return x
        