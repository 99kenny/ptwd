#!/bin/bash
# conda activate ptwd
# A
python main.py --exp_name A image_prompt --model vit_base_patch16_224 --batch-size 16 --data-path ./local_datasets/ --output_dir ./output 
# B
#python main.py image_prompt --model vit_base_patch16_224 --batch-size 16 --data-path ./local_datasets/ --output_dir ./output --exp_name B 
# C
#python main.py image_prompt --model vit_base_patch16_224 --batch-size 16 --data-path ./local_datasets/ --output_dir ./output --exp_name C
# D
#python main.py image_prompt --model vit_base_patch16_224 --batch-size 16 --data-path ./local_datasets/ --output_dir ./output --exp_name D 
