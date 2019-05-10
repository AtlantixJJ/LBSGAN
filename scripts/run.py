import os

logdir = "cifar10_"
compare_saw = ["swa", "noswa"]
compare_optim = ["adam", "sgd"]

python3 gan.py --dir=logs/cifar10_adam --dataset=CIFAR100 --data_path=data --model=VGG16 --epochs=200 --lr_init=0.001 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.01
python3 train.py --dir=logs/cifar10_sgd_swa --dataset=CIFAR100 --data_path=data --model=VGG16 --epochs=200 --lr_init=0.05 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.01