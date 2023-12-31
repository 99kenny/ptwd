import logging

import torch
import numpy as np

logger = logging.getLogger(__name__)

class DeepInversionFeatureHooK():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # batch, size (prompt_patch + cls + img_patch), dim
        # regularize only prompt_patch
        
        feature = input[0][:,:module.patch_num+1,:]
        nch = feature.shape[1] # img_patch
        mean = feature.mean([0,2])
        var = feature.permute(1,0,2).contiguous().view([nch, -1]).var(1, unbiased=False)
        
        r_feature = torch.norm(module.var - var, 2) + torch.norm(module.mean - mean, 2)
        self.r_feature = r_feature
        logger.debug(f'feature {feature.shape}')
        logger.debug(f'nch : {nch}')
        logger.debug(f'mean : {mean.shape}')
        logger.debug(f'var : {var.shape}')
        logger.debug(f'r_feature : {r_feature}')
        
    def close(self):
        self.hook.remove()