import os
import torch
from lib.train import summary_grid_image
import numpy as np
import torch.nn.functional as F
import torchvision.utils as vutils


def gan_eval(loader, gen_model, disc_model, criterion):
    loss_fake_sum, loss_real_sum = 0.0, 0.0

    gen_model.eval()
    disc_model.eval()

    z, label_valid, label_fake = None, None, None

    for i, (input, target) in enumerate(loader):
        x = input.cuda()
        target = target.cuda()
        if z is None:
            z = torch.zeros(x.size(0), 128); z = z.cuda()
            label_valid = torch.zeros_like(disc_fake).fill_(1.0).cuda()
            label_fake  = torch.zeros_like(disc_fake).fill_(0.0).cuda()
        z.normal_()
        fake_x = gen_model(z)
        disc_fake = disc_model(fake_x)
        disc_real = disc_model(x)
        loss_fake = criterion(disc_fake, label_fake)
        loss_real = criterion(disc_real, label_valid)
        loss_fake_sum += loss_fake.detach()[0] * input.size(0)
        loss_real_sum += loss_real.detach()[0] * input.size(0)

    return {
        'loss_fake': loss_fake_sum / len(loader.dataset),
        'loss_real': loss_real_sum / len(loader.dataset)
    }

class FixedNoiseEvaluator(object):
    def __init__(self, z_dim=10):
        self.fixed_noise = np.random.normal(size=(16, z_dim))

        self.fixed_noise = torch.Tensor(self.fixed_noise)

    def summary(self, loader, summary, crit, gen_model, disc_model, name, step):
        summary_grid_image(summary, gen_model(self.fixed_noise).detach(), name + "fixed_generate", step)
        res = gan_eval(loader, gen_model, disc_model, crit)
        summary.add_scalar(res['loss_fake'], name + "loss_fake", step)
        summary.add_scalar(res['loss_real'], name + "loss_real", step)
        return res