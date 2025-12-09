#!/bin/bash

python src/entropy.py --checkpoint_path outputs/rb_refind/-3/checkpoint-1500 --data_dir data/refind --num_samples 1000 --batch_size 32 --device cuda --overwrite
python src/entropy.py --checkpoint_path outputs/rb_refind/-2/checkpoint-1500 --data_dir data/refind --num_samples 1000 --batch_size 32 --device cuda --overwrite
python src/entropy.py --checkpoint_path outputs/rb_refind/-1/checkpoint-1500 --data_dir data/refind --num_samples 1000 --batch_size 32 --device cuda --overwrite
python src/entropy.py --checkpoint_path outputs/rb_refind/0/checkpoint-1500 --data_dir data/refind --num_samples 1000 --batch_size 32 --device cuda --overwrite
python src/entropy.py --checkpoint_path outputs/rb_refind/1/checkpoint-1500 --data_dir data/refind --num_samples 1000 --batch_size 32 --device cuda --overwrite
python src/entropy.py --checkpoint_path outputs/rb_refind/2/checkpoint-1500 --data_dir data/refind --num_samples 1000 --batch_size 32 --device cuda --overwrite
python src/entropy.py --checkpoint_path outputs/rb_refind/3/checkpoint-1500 --data_dir data/refind --num_samples 1000 --batch_size 32 --device cuda --overwrite
python src/entropy.py --checkpoint_path outputs/rb_refind/4/checkpoint-1500 --data_dir data/refind --num_samples 1000 --batch_size 32 --device cuda --overwrite
python src/entropy.py --checkpoint_path outputs/rb_refind/5/checkpoint-1500 --data_dir data/refind --num_samples 1000 --batch_size 32 --device cuda --overwrite
python src/entropy.py --checkpoint_path outputs/rb_refind/6/checkpoint-1500 --data_dir data/refind --num_samples 1000 --batch_size 32 --device cuda --overwrite
python src/entropy.py --checkpoint_path outputs/rb_refind/7/checkpoint-1500 --data_dir data/refind --num_samples 1000 --batch_size 32 --device cuda --overwrite
python src/entropy.py --checkpoint_path outputs/rb_refind/8/checkpoint-1500 --data_dir data/refind --num_samples 1000 --batch_size 32 --device cuda --overwrite
python src/entropy.py --checkpoint_path outputs/rb_refind/9/checkpoint-1500 --data_dir data/refind --num_samples 1000 --batch_size 32 --device cuda --overwrite
python src/entropy.py --checkpoint_path outputs/rb_refind/10/checkpoint-1500 --data_dir data/refind --num_samples 1000 --batch_size 32 --device cuda --overwrite
