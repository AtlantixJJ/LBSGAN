import os, tqdm, time
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

import lib

def summary_grid_image(logger, x, name, step):
    if x.max() > 1:
        x = (x - x.max()) / (x.max() - x.min())
    else:
        x = (x + 1) / 2 # assume (-1, 1) input
    logger.add_image(name, vutils.make_grid(x[:16], nrow=4), step)

class BaseGANTrainer(object):
    def __init__(self, gen_model, disc_model, dataloader, cfg):
        self.cfg = cfg
        self.summary_writer = SummaryWriter(cfg.log_dir)
        self.evaluator = lib.evaluator.FixedNoiseEvaluator()
        self.summary_interval = cfg.args.summary_interval
        self.n_epoch = cfg.args.epochs
        self.gen_model = gen_model
        self.disc_model = disc_model
        self.dataloader = dataloader

        self.fixed_noise = torch.randn((16, 128)).cuda() * 2

        self.gen_optim = torch.optim.Adam(self.gen_model.parameters(),
            lr=1e-4, betas=(0.0, 0.9), weight_decay=1e-4)
        self.disc_optim = torch.optim.Adam(self.disc_model.parameters(),
            lr=4e-4, betas=(0.0, 0.9), weight_decay=1e-4)
        
        self.adv_crit = torch.nn.BCEWithLogitsLoss()
        self.adv_crit.cuda()

        self.reset()
    
    def reset(self):
        self.iter = 0
    
    def train(self):
        self.start_time = time.clock()
        self.compute_time = 0
        self.read_time = 0
        #self.forward_time = 0
        #self.backward_time = 0
        #self.update_time = 0
        for i in range(self.n_epoch):
            self.dataloader.reset()
            self.train_epoch()
            if i % self.cfg.args.save_freq == 1:
                self.save()

    def save(self):
        savepath = self.cfg.args.log_dir
        torch.save(dict(
            gen_model=self.gen_model.state_dict(),
            disc_model=self.disc_model.state_dict()
        ), os.path.join(savepath, "model.pytorch"))

    def train_epoch(self):
        self.gen_model.train()
        self.disc_model.train()

        z, label_valid, label_fake = None, None, None

        b2 = time.clock()
        for i, (input, target) in enumerate(tqdm.tqdm(self.dataloader)):
            b1 = time.clock()
            self.read_time += b1 - b2
            x = input.cuda()
            target = target.cuda()

            # optimze G
            self.gen_optim.zero_grad()
            #self.gen_model.train()
            #self.disc_model.eval()
            if z is None:
                z = torch.zeros(x.size(0), 128)
                z = z.cuda()
            z = z.normal_() * 2
            fake_x = self.gen_model(z)
            disc_fake = self.disc_model(fake_x)
            if label_valid is None:
                label_valid = torch.zeros_like(disc_fake).fill_(1.0).cuda()
                label_fake  = torch.zeros_like(disc_fake).fill_(0.0).cuda()
            g_loss = self.adv_crit(disc_fake, label_valid)
            g_loss.backward()
            self.gen_optim.step()
            g_loss_val = g_loss.detach().cpu().numpy()
            self.summary_writer.add_scalar("g_loss", g_loss_val, self.iter)

            # optimize D
            #self.disc_model.train()
            #self.gen_model.eval()
            self.disc_optim.zero_grad()
            disc_real = self.disc_model(x)
            disc_fake = self.disc_model(fake_x.detach())
            loss_real = self.adv_crit(disc_real, label_valid)
            loss_fake = self.adv_crit(disc_fake, label_fake)
            d_loss = loss_real + loss_fake
            d_loss.backward()
            d_loss_val = d_loss.detach().cpu().numpy()
            self.disc_optim.step()
            self.summary_writer.add_scalar("d_loss", d_loss_val, self.iter)

            b2 = time.clock()

            self.compute_time += b2 - b1

            self.iter += 1

            if self.iter % self.summary_interval == 0:
                summary_grid_image(self.summary_writer, self.gen_model(self.fixed_noise).detach(),
                    "generate_image", self.iter)
                summary_grid_image(self.summary_writer, x,
                    "real_image", self.iter)