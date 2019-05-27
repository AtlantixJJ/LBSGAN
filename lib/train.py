import os, tqdm, time
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

import lib
from lib import utils

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
        self.fid_evaluator = lib.evaluator.FIDEvaluator(cfg.ref_path, cuda=True)
        self.summary_interval = cfg.args.summary_interval
        self.n_epoch = cfg.args.n_epoch
        self.gen_model = gen_model
        self.disc_model = disc_model
        self.dataloader = dataloader

        self.fixed_noise = torch.randn((16, 128)).cuda() * 2

        self.gen_optim = torch.optim.Adam(self.gen_model.parameters(),
            lr=cfg.g_lr, betas=(0.0, 0.9), weight_decay=1e-5)
        self.disc_optim = torch.optim.Adam(self.disc_model.parameters(),
            lr=cfg.d_lr, betas=(0.0, 0.9), weight_decay=1e-5)
        
        self.adv_crit = torch.nn.BCEWithLogitsLoss()
        self.adv_crit.cuda()

        self.reset()
    
    def reset(self):
        self.iter = 0
    
    def train(self):
        self.epoch = 0
        self.start_time = time.clock()
        self.compute_time = 0
        self.read_time = 0
        for i in range(self.n_epoch):
            self.epoch = i
            self.dataloader.reset()
            self.train_epoch()
            print("=> Average Time: %.3f" % (self.compute_time / (self.epoch + 1)))
            if i % self.cfg.args.save_freq == 0 and i != 0:
                self.save()
            if i % self.cfg.args.eval_freq == 0 and i != 0:
                self.dataloader.train = False
                self.dataloader.reset()
                with torch.no_grad():
                    result = self.eval()
                print("=> Evaluate epoch %d" % self.epoch)
                for k, v in result.items():
                    print("=> %s\t%.3f" % (k, v))
                    self.summary_writer.add_scalar("eval_" + k, v, self.epoch)
                self.dataloader.train = True
    
    def eval(self):
        save_path = os.path.join(self.cfg.log_dir, "eval_epoch_%d" % self.epoch)
        os.system("mkdir %s" % save_path)
        self.gen_model.eval()
        self.disc_model.eval()
        fid_dist = self.fid_evaluator.fid(self.gen_model)

        summary_grid_image(self.summary_writer, self.gen_model(self.fixed_noise).detach(),
            "eval_generate_image", self.epoch)
            
        return {
            "fid" : fid_dist,
            "avgtime" : self.compute_time / (self.epoch + 1)
            }

    def save(self):
        savepath = self.cfg.log_dir
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
            if type(input) == int: break
            b1 = time.clock()
            self.read_time += b1 - b2
            x = input.cuda()
            target = target.cuda()
            if z is None or z.size(0) != x.size(0):
                z = torch.zeros(x.size(0), 128)
                z = z.cuda()
            z = z.normal_() * 2

            # optimze G
            self.gen_optim.zero_grad()
            fake_x = self.gen_model(z)
            disc_fake = self.disc_model(fake_x)
            if label_valid is None or label_valid.size(0) != x.size(0):
                label_valid = torch.zeros_like(disc_fake).fill_(1.0).cuda()
                label_fake  = torch.zeros_like(disc_fake).fill_(0.0).cuda()
            g_loss = self.adv_crit(disc_fake, label_valid)
            g_loss.backward()
            self.gen_optim.step()
            g_loss_val = g_loss.detach().cpu().numpy()
            self.summary_writer.add_scalar("g_loss", g_loss_val, self.iter)

            # optimize D
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
            if self.iter == 0:
                summary_grid_image(self.summary_writer, x,
                    "real_image", self.iter)