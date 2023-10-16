import torch
import numpy as np

class DeepInversionFeatureHooK():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # batch, size (prompt_patch + cls + img_patch), dim
        # regularize only prompt_patch
        
        feature = input[0][:,:module.patch_num,:]
        nch = feature.shape[1] # img_patch
        mean = feature.mean([0,2])
        var = feature.permute(1,0,2).contiguous().view([nch, -1]).var(1, unbiased=False)
        # debug
        r_feature = torch.norm(module.var - var, 2) + torch.norm(module.mean - mean, 2)
        self.r_feature = r_feature
    
    def close(self):
        self.hook.remove()