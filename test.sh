CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py \
    --flagfile ./logs/DDPM_CIFAR10_EPS/flagfile.txt \
    --notrain \
    --eval \
    --parallel
