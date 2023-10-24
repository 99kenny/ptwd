#!/bin/bash
# conda activate ptwd
# A
CUDA_VISIBLE_DEVICES=4 python main.py --exp_name a image_prompt --model vit_base_patch16_224 --batch-size 16 --data_path ./local_datasets/ --output_dir ./output --epochs 100 | tee a.log 
# B
#pCUDA_VISIBLE_DEVICES=4 python main.py --exp_name b image_prompt --model vit_base_patch16_224 --batch-size 16 --data_path ./local_datasets/ --output_dir ./output --alpha_tv_l2 2.5e-5 --epochs 100 | tee b.log 
# C
#pCUDA_VISIBLE_DEVICES=4 python main.py --exp_name c image_prompt --model vit_base_patch16_224 --batch-size 16 --data-path ./local_datasets/ --output_dir ./output --alpha_tv_l2 2.5e-5 --alpha_f 1.0 --epochs 100 | tee c.log
# D
#pCUDA_VISIBLE_DEVICES=4 python main.py --exp_name d image_prompt --model vit_base_patch16_224 --batch-size 16 --data-path ./local_datasets/ --output_dir ./output --alpha_tv_l2 2.5e-5 --alpha_f 1.0 --alpha_l2 3e-8 --epochs 100 | tee d.log
