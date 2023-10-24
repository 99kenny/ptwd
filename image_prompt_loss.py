import logging
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image

logger = logging.getLogger(__name__)

class ImagePromptLoss(nn.Module):
    def __init__(self, r_feature_layers, alpha_main=1.0, alpha_tv_l1=0, alpha_tv_l2=2.5e-5, alpha_l2=3e-8, alpha_f=1.0):
        super().__init__()
        self.alpha_main = alpha_main
        self.alpha_tv_l1 = alpha_tv_l1
        self.alpha_tv_l2 = alpha_tv_l2
        self.alpha_l2 = alpha_l2
        self.alpha_f = alpha_f
        self.r_feature_layers = r_feature_layers
        logger.debug(f'alpha_main : {alpha_main}, alpha_tv_l2 : {alpha_tv_l1}, alpha_tv_l2 : {alpha_tv_l2}, alpha_l2 : {alpha_l2}, alpha_f : {alpha_f}')
        
    def r_prior(self, inputs):
        # COMPUTE total variation regularization loss
        diff1 = inputs[:, :, :, :-1] - inputs[:, :, :, 1:]
        diff2 = inputs[:, :, :-1, :] - inputs[:, :, 1:, :]
        diff3 = inputs[:, :, 1:, :-1] - inputs[:, :, :-1, 1:]
        diff4 = inputs[:, :, :-1, :-1] - inputs[:, :, 1:, 1:]

        loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
        loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
                diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
        loss_var_l1 = loss_var_l1 * 255.0
        #print(f'r_prior shape : {loss_var_l2.shape}')
        return loss_var_l1, loss_var_l2
    
    def r_feature(self,):
        ret = sum([mod.r_feature for (idx, mod) in enumerate(self.r_feature_layers)])
        #print(f'r_feature shaep : {ret.shape}' )
        return ret
    def r_l2(self, inputs):
        # batch 별 norm
        #print(f'norm inputs : {inputs.shape}')
        #for inpt in inputs:   
        norm = torch.norm(inputs)
        #print(f'l2 shape : {norm.shape}')
        return norm
    
    def forward(self, inputs):
        logger.debug(f'Prompt_loss input shape : {inputs.shape}')
        #save_image(make_grid(inputs, nrow=16), f'./output/prompt.jpg')
        loss_var_l1, loss_var_l2 = self.r_prior(inputs)
        loss = self.alpha_tv_l2 * loss_var_l2 + self.alpha_tv_l1 * loss_var_l1 + \
               self.alpha_l2 * self.r_l2(inputs) + \
               self.alpha_f * self.r_feature()
               
        return loss