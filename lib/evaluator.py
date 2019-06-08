import os, glob, pathlib
import torch
import lib
from lib.train import summary_grid_image
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from lib.fid.inception_modified import inception_v3

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

class FileEvaluator(object):
    def __init__(self, save_dir, file_num=50000, batch_size=50, cuda=False):
        self.save_dir = save_dir
        self.file_num = file_num
        self.cuda = cuda
        self.batch_size = batch_size
    
    def __call__(self, gen_model, epoch):
        epoch_dir = os.path.join(self.save_dir, "eval_epoch_%d" % epoch)
        print("=> Save image to %s" % epoch_dir)
        os.system("mkdir %s" % epoch_dir)
        z = torch.Tensor(self.batch_size, 128)
        if self.cuda: z = z.cuda()
        for i in range(self.file_num // self.batch_size):
            z = z.normal_() * 2
            lib.utils.save_4dtensor_image(
                epoch_dir + "/%05d.jpg",
                i * self.batch_size,
                (gen_model(z) + 1) * 127.5)

class FIDEvaluator(object):
    def __init__(self, ref_datapath, save_path=None, batch_size=50, cuda=True):
        self.ref_datapath = ref_datapath
        self.save_path = save_path
        self.batch_size = batch_size
        self.cuda = cuda

        self.model = inception_v3(pretrained=True, aux_logits=False, transform_input=False)
        self.model.eval()
        if cuda: self.model.cuda()
        
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def calculate_statistics_given_path(self, path, save_npy=True):
        npylist = glob.glob(os.path.join(path, "*.npy"))
        if len(npylist) > 0:
            path = npylist[0]
            print("=> Use npy file %s" % path)
            f = np.load(path).tolist()
            m, s = f['mu'][:], f['sigma'][:]
            return m, s
        else:
            print("=> Calc from path %s" % path)
            op = pathlib.Path(path)
            ds = lib.dataset.SimpleDataset(path, (299, 299), self.transform_test)
            np.random.RandomState(1314).shuffle(ds.files)
            ds.files = ds.files[:50000]
            dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=1, pin_memory=True)
            feature = lib.fid.fid_score.get_feature(self.model, dl, self.cuda)
            mu = np.mean(feature, axis=0)
            sigma = np.cov(feature, rowvar=False)
            if save_npy:
                np.save(os.path.join(path, "mu_sigma.npy"), {"mu": mu, "sigma": sigma})
            return mu, sigma

    def __call__(self, gs, epoch):
        m1, s1 = self.calculate_statistics_given_path(self.ref_datapath)
        m2, s2 = lib.fid.fid_score.calculate_statistics_given_iterator(
            self.model,
            gs.iterator() if self.save_path is None else gs.iterator(self.save_path + ("/eval_epoch_%d" % epoch)),
            self.cuda)
        return lib.fid.fid_score.calculate_frechet_distance(m1, s1, m2, s2)