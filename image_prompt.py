import logging
import torch
import torch.nn as nn
from torchvision import transforms

from datasets import get_images

logger = logging.getLogger(__name__)

class ImagePrompt(nn.Module):
    def __init__(self, patch_embed,embed_dim=768, size=32, embedding_key='mean', prompt_init='uniform', prompt_pool=False,
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform', channel=3):
        super().__init__()
        logger.info('ImagePrompt Configs : %s', self.__dict__)
        self.size = size
        self.patch_embed = patch_embed
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
            
        if self.prompt_pool:
            prompt_pool_shape = (pool_size, channel, size, size)
            logger.debug(f'prompt pool shape {prompt_pool_shape}')
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)
            elif prompt_init == 'train':
                size = pool_size
                logger.info('load %d sample images for training initialization', pool_size)
                self.prompt = nn.Parameter(get_images(size))
                logger.debug(f'prompt shape : {self.prompt.shape}')
                
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            if prompt_key_init == 'uniform' or prompt_key_init == 'train':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # if not use prompt_key, use image prompt embedding as prompt key
            self.prompt_key = None
        logger.debug(f'using prompt key : {self.prompt_key.shape}')
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        logger.debug('--------------image prompt forward------------------')
        out = dict()

        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        # x_embed (batch, patch_num, dim)
        # TODO: 224 should be changed to img size
        resized_prompt = transforms.RandomResizedCrop(224, scale=scale, ratio=ratio)(self.prompt) # (pool_size, channel, 224, 224)
        prompt_embed = self.patch_embed(resized_prompt) # (pool_size, patch_num, dim)
        logger.debug(f'prompt : {self.prompt.shape}')
        logger.debug(f'resized prompt : {resized_prompt.shape}')
        logger.debug(f'prompt_embed : {prompt_embed.shape}')
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0]
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys")
            logger.debug(f'cls_features : {cls_features.shape}')
            logger.debug(f'x_embed_mean : {x_embed_mean.shape}')
            if self.prompt_key is None:
                self.prompt_key = prompt_embed 
            prompt_norm = self.l2_normalize(self.prompt_key, dim=1) # pool_size, dim
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # batch, patch_num, dim  
            
            similarity = torch.matmul(x_embed_norm, prompt_norm.t())
            logger.debug(f'similarity : {similarity.shape}')
            if prompt_mask is None:
                _, idx = torch.topk(similarity, k=self.top_k, dim=1)
                logger.debug(f'topk idxs : {idx}')
                if self.batchwise_prompt:
                    prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                    if prompt_id.shape[0] < self.pool_size:
                        prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                        id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],),0, device=id_counts.device)])

                    major_prompt_id = prompt_id[major_idx]
                    idx = major_prompt_id.expand(x_embed.shape[0], -1)
                    logger.debug(f'batchwise : {idx}')
            else:
                idx = prompt_mask
                logger.debug(f'prompt mask : {prompt_mask}')
            # idx (batch_size, topk)
            batched_prompt_embed = prompt_embed[idx] # batch, top_k, patch_num, 768  
            # image prompt uses one prompt per image
            batched_prompt_embed = batched_prompt_embed.squeeze(dim=1) # batch, patch_num, 768
            #batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c)
            
            out['prompt_image'] = self.prompt[idx]
            out['prompt_idx'] = idx
            out['prompt_norm'] = prompt_norm
            out['x_embed_norm'] = x_embed_norm
            out['similarity'] = similarity
            
            batched_key_norm = prompt_norm[idx]
            out['selected_key'] = batched_key_norm
            x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
            sim = batched_key_norm * x_embed_norm
            reduce_sim = torch.sum(sim) / x_embed.shape[0]
            
            out['reduce_sim'] = reduce_sim
        else:
            # if self.prompt_init == 'zero':
            #     self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
            # elif self.prompt_init == 'uniform':
            #     self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
            #     nn.init.uniform_(self.prompt)
            # batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)
            # 
            pass
        out['total_prompt_len'] = batched_prompt_embed.shape[1]
        # todo: prompt concat channel
        out['prompted_embedding'] = torch.cat([batched_prompt_embed, x_embed], dim=1)
        
        return out