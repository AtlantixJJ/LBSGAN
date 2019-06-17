import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F
import torchvision
import models, lib

cfg = lib.config.BaseConfig()
cfg.parse()

print('Preparing model')
gen_model = cfg.gen_function(
    upsample=cfg.upsample,
    map_size=cfg.map_size,
    out_dim=cfg.out_dim)
disc_model = cfg.disc_function(
    downsample=cfg.downsample,
    in_dim=cfg.out_dim)
if cfg.num_gpu > 1:
    gen_model = torch.nn.DataParallel(gen_model)
    disc_model = torch.nn.DataParallel(disc_model)
gen_model.cuda()
disc_model.cuda()
print(gen_model)
print(disc_model)
print("=> Generator")
print(gen_model)
print("=> Discriminator")
print(disc_model)

if cfg.args.delayed_batch_size > -1:
    trainer = lib.train.DelayLBSTrainer(gen_model=gen_model, disc_model=disc_model, dataloader=cfg.dl, cfg=cfg)
else:
    trainer = lib.train.BaseGANTrainer(gen_model=gen_model, disc_model=disc_model, dataloader=cfg.dl, cfg=cfg)

trainer.train()
