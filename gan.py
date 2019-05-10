import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F
import torchvision
import models, utils, lib

cfg = lib.config.BaseConfig()
cfg.parse()

print('Preparing model')
gen_model = cfg.gen_function(upsample=cfg.upsample)
gen_model.cuda()
disc_model = cfg.disc_function(downsample=cfg.upsample)
disc_model.cuda()
print(gen_model)
print(disc_model)
print("=> Generator")
print(gen_model)
print("=> Discriminator")
print(disc_model)

trainer = lib.train.BaseGANTrainer(gen_model, disc_model, cfg.dl, cfg)
trainer.train()
