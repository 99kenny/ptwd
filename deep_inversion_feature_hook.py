import torch

class DeepInversionFeatureHooK():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0,2])
        var = input[0].permute(1,0,2).contiguous().view([nch, -1]).var(1, unbiased=False)
        # debug
        print(module.var)
        print(module.mean)
        r_feature = torch.norm(module.var - var, 2) + torch.norm(module.mean - mean, 2)
        self.r_feature = r_feature
    
    def close(self):
        self.hook.remove()

